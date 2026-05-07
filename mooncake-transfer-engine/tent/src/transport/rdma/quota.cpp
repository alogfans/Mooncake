// Copyright 2025 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tent/transport/rdma/quota.h"
#include "tent/transport/rdma/shared_quota.h"
#include "tent/common/utils/random.h"
#include "tent/common/utils/os.h"

#include <algorithm>
#include <iostream>
#include <iomanip>

namespace mooncake {
namespace tent {
Status DeviceQuota::loadTopology(std::shared_ptr<Topology>& local_topology) {
    local_topology_ = local_topology;
    for (size_t dev_id = 0; dev_id < local_topology->getNicCount(); ++dev_id) {
        auto entry = local_topology->getNicEntry(dev_id);
        if (!entry || entry->type != Topology::NIC_RDMA) continue;
        DeviceInfo& info = devices_[dev_id];
        info.dev_id = dev_id;
        info.bw_gbps = kDefaultBwGbps;
        info.numa_id = entry->numa_node;
        info.ewma_bandwidth_bps.store(info.getTheoreticalBandwidth(),
                                      std::memory_order_relaxed);
    }
    return Status::OK();
}

Status DeviceQuota::enableSharedQuota(const std::string& shm_name) {
    shared_quota_ = std::make_shared<SharedQuotaManager>(this);
    auto status = shared_quota_->attach(shm_name);
    if (!status.ok()) shared_quota_.reset();
    return status;
}

Status DeviceQuota::allocate(uint64_t total_length, uint32_t num_slices,
                             uint64_t slice_bytes,
                             const std::string& location,
                             std::vector<int>& slice_dev_ids,
                             uint64_t device_mask) {
    slice_dev_ids.clear();
    slice_dev_ids.reserve(num_slices);
    auto entry = local_topology_->getMemEntry(location);
    if (!entry) return Status::InvalidArgument("Unknown location" LOC_MARK);

    if (!enable_quota_) {
        for (size_t rank = 0; rank < Topology::DevicePriorityRanks; ++rank) {
            thread_local std::vector<int> tl_eligible;
            tl_eligible.clear();
            for (int dev_id : entry->device_list[rank]) {
                if (!devices_.count(dev_id)) continue;
                if ((device_mask & (1ULL << dev_id)) == 0) continue;
                tl_eligible.push_back(dev_id);
            }
            if (tl_eligible.empty()) break;
            uint64_t offset = 0;
            for (uint32_t i = 0; i < num_slices; ++i) {
                thread_local uint64_t tl_rr_counter = 0;
                int dev_id = tl_eligible[tl_rr_counter % tl_eligible.size()];
                tl_rr_counter++;
                slice_dev_ids.push_back(dev_id);
                uint64_t this_slice_bytes = std::min(slice_bytes, total_length - offset);
                offset += this_slice_bytes;
                devices_[dev_id].total_bytes.fetch_add(
                    this_slice_bytes, std::memory_order_relaxed);
            }
            return Status::OK();
        }
        return Status::DeviceNotFound("no eligible devices");
    }

    std::vector<DeviceQuota::Candidate> tl_candidates;
    Status status = buildCandidates(entry, slice_bytes, device_mask,
                                    tl_candidates);
    if (!status.ok()) return status;
    if (num_slices == 1) {
        selectSinglePath(tl_candidates, num_slices, total_length,
                         slice_dev_ids);
    } else {
        thread_local uint64_t tl_call_count = 0;
        bool explore_mode = ((++tl_call_count % 100) == 0);
        selectMultiPath(tl_candidates, num_slices, total_length, slice_dev_ids,
                        explore_mode);
    }
    return Status::OK();
}

int DeviceQuota::getDeviceRank(const std::string& location, int dev_id) const {
    auto entry = local_topology_->getMemEntry(location);
    if (!entry) return 0;
    for (size_t rank = 0; rank < Topology::DevicePriorityRanks; ++rank) {
        for (int id : entry->device_list[rank]) {
            if (id == dev_id) return static_cast<int>(rank);
        }
    }
    return 0;
}

Status DeviceQuota::buildCandidates(const Topology::MemEntry* entry,
                                    uint64_t slice_bytes, uint64_t device_mask,
                                    std::vector<Candidate>& candidates) {
    for (size_t rank = 0; rank < Topology::DevicePriorityRanks; ++rank) {
        for (int dev_id : entry->device_list[rank]) {
            if (!devices_.count(dev_id)) continue;
            if ((device_mask & (1ULL << dev_id)) == 0) continue;

            auto& dev = devices_[dev_id];
            uint64_t inflight = dev.getInflightBytes();
            double ewma_bw = dev.getEwmaBandwidth();
            double predicted_time = static_cast<double>(inflight + slice_bytes) / ewma_bw;
            double rank_weight = sched_params_.rank_weights[rank];
            double score = predicted_time * rank_weight;
            score += (SimpleRandom::Get().next(10) * 1e-9);

            Candidate c;
            c.dev_id = dev_id;
            c.score = score;
            c.is_cross_numa = (rank > 0);
            c.dev_priority = getDevicePriority(dev_id);
            candidates.push_back(c);
        }
    }

    if (candidates.empty()) {
        return Status::DeviceNotFound("no eligible devices");
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  if (std::abs(a.score - b.score) > 1e-9)
                      return a.score < b.score;
                  return a.dev_id < b.dev_id;
              });

    return Status::OK();
}

void DeviceQuota::selectSinglePath(const std::vector<Candidate>& candidates,
                                   uint32_t num_slices, uint64_t total_length,
                                   std::vector<int>& slice_dev_ids) {
    if (candidates.empty()) return;

    const Candidate& best = candidates[0];
    int dev_id = best.dev_id;
    auto& dev = devices_[dev_id];

    dev.addInflight(total_length);
    dev.total_bytes.fetch_add(total_length, std::memory_order_relaxed);

    for (uint32_t i = 0; i < num_slices; ++i) {
        slice_dev_ids.push_back(dev_id);
    }
}

void DeviceQuota::selectMultiPath(const std::vector<Candidate>& candidates,
                                  uint32_t num_slices, uint64_t total_length,
                                  std::vector<int>& slice_dev_ids,
                                  bool explore_mode) {
    if (candidates.empty()) return;
    uint64_t slice_bytes = (total_length + num_slices - 1) / num_slices;
    if (explore_mode) {
        for (uint32_t i = 0; i < num_slices; ++i) {
            const Candidate& c = candidates[i % candidates.size()];
            slice_dev_ids.push_back(c.dev_id);
            devices_[c.dev_id].addInflight(slice_bytes);
            devices_[c.dev_id].total_bytes.fetch_add(
                slice_bytes, std::memory_order_relaxed);
        }
    } else {
        double total_weight = 0.0;
        double max_weight = -1.0;
        int best_dev_idx = -1;
        for (size_t i = 0; i < candidates.size(); ++i) {
            double w = 1.0 / (candidates[i].score + 1e-12);
            total_weight += w;
            if (w > max_weight) {
                max_weight = w;
                best_dev_idx = static_cast<int>(i);
            }
        }
        if (best_dev_idx == -1 || num_slices == 0 || total_weight <= 0.0)
            return;
        uint32_t remaining_slices = num_slices;
        for (size_t i = 0; i < candidates.size(); ++i) {
            double w = 1.0 / (candidates[i].score + 1e-12);
            uint32_t assigned =
                static_cast<uint32_t>((w / total_weight) * num_slices);
            if (assigned > 0) {
                if (assigned > remaining_slices) assigned = remaining_slices;
                remaining_slices -= assigned;
                const Candidate& c = candidates[i];
                for (uint32_t s = 0; s < assigned; ++s) {
                    slice_dev_ids.push_back(c.dev_id);
                }
                uint64_t total_assigned_bytes =
                    static_cast<uint64_t>(slice_bytes) * assigned;
                devices_[c.dev_id].addInflight(total_assigned_bytes);
                devices_[c.dev_id].total_bytes.fetch_add(
                    total_assigned_bytes, std::memory_order_relaxed);
            }
        }
        if (remaining_slices > 0) {
            const Candidate& c = candidates[best_dev_idx];
            for (uint32_t s = 0; s < remaining_slices; ++s) {
                slice_dev_ids.push_back(c.dev_id);
            }
            uint64_t total_assigned_bytes =
                static_cast<uint64_t>(slice_bytes) * remaining_slices;
            devices_[c.dev_id].addInflight(total_assigned_bytes);
            devices_[c.dev_id].total_bytes.fetch_add(total_assigned_bytes,
                                                     std::memory_order_relaxed);
        }
    }
}

Status DeviceQuota::allocate(uint64_t length, const std::string& location,
                             int& chosen_dev_id) {
    std::vector<int> slice_dev_ids;
    Status status = allocate(length, 1, length, location, slice_dev_ids, ~0ULL);
    if (!status.ok()) return status;
    if (slice_dev_ids.empty()) {
        return Status::DeviceNotFound("allocation failed");
    }
    chosen_dev_id = slice_dev_ids[0];
    return Status::OK();
}

Status DeviceQuota::release(int dev_id, uint64_t length, double latency) {
    auto it = devices_.find(dev_id);
    if (it == devices_.end())
        return Status::InvalidArgument("device not found");

    auto& dev = it->second;
    dev.releaseInflight(length);

    if (!enable_quota_ || !sched_params_.enable_ewma_learning) {
        return Status::OK();
    }

    double observed_bw = static_cast<double>(length) / latency;
    double current_ewma = dev.getEwmaBandwidth();

    double alpha = sched_params_.ewma_alpha;
    double new_ewma = alpha * current_ewma + (1.0 - alpha) * observed_bw;

    double theoretical_bw = dev.getTheoreticalBandwidth();
    new_ewma = std::max(0.1 * theoretical_bw, std::min(10.0 * theoretical_bw, new_ewma));

    dev.ewma_bandwidth_bps.store(new_ewma, std::memory_order_relaxed);
    dev.addSample(length, latency);

    return Status::OK();
}

void DeviceQuota::updateStats(int dev_id, uint64_t bytes) {
    auto it = devices_.find(dev_id);
    if (it != devices_.end()) {
        it->second.total_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
}

void DeviceQuota::printTrafficStats() {
    std::cout << "=== Device Traffic Statistics ===" << std::endl;
    for (const auto& [dev_id, dev] : devices_) {
        uint64_t total = dev.total_bytes.load(std::memory_order_relaxed);
        double ewma_bw_gbps = dev.getEwmaBandwidth() / 1e9 * 8.0;
        uint64_t inflight = dev.getInflightBytes();
        std::cout << "Dev " << dev_id << ": "
                  << "Total=" << (total / 1024.0 / 1024.0 / 1024.0) << " GB, "
                  << "EWMA BW=" << std::fixed << std::setprecision(2) << ewma_bw_gbps << " Gbps, "
                  << "Inflight=" << inflight << " bytes"
                  << std::endl;
    }
}

void DeviceQuota::fillDevicePriorities() {
    sched_params_.device_base_priorities.clear();
    for (const auto& [dev_id, dev] : devices_) {
        sched_params_.device_base_priorities.push_back(dev_id);
    }
}

int DeviceQuota::getDevicePriority(int dev_id) const {
    if (!sched_params_.enable_device_priority) return 0;

    auto it = std::find(sched_params_.device_base_priorities.begin(),
                        sched_params_.device_base_priorities.end(), dev_id);
    if (it == sched_params_.device_base_priorities.end()) return 0;

    return static_cast<int>(std::distance(sched_params_.device_base_priorities.begin(), it));
}

bool DeviceQuota::isCrossNumaNodeIdle(int dev_numa) const {
    for (const auto& [dev_id, dev] : devices_) {
        if (dev.numa_id == dev_numa) {
            uint64_t inflight = dev.getInflightBytes();
            if (inflight < 1024 * 1024) {
                return true;
            }
        }
    }
    return false;
}

}  // namespace tent
}  // namespace mooncake