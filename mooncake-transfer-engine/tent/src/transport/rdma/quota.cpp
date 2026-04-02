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

#include <algorithm>
#include <iostream>
#include <iomanip>

namespace mooncake {
namespace tent {

thread_local uint64_t tl_rr_counter = 0;

static std::vector<int> collectEligibleDevices(
    const Topology::MemEntry* entry,
    const std::unordered_map<int, DeviceQuota::DeviceInfo>& devices,
    uint64_t device_mask) {
    std::vector<int> result;
    for (size_t rank = 0; rank < Topology::DevicePriorityRanks; ++rank) {
        for (int dev_id : entry->device_list[rank]) {
            if (!devices.count(dev_id)) continue;
            if ((device_mask & (1ULL << dev_id)) == 0) continue;
            result.push_back(dev_id);
        }
    }
    return result;
}

Status DeviceQuota::loadTopology(std::shared_ptr<Topology>& local_topology) {
    local_topology_ = local_topology;
    std::unordered_set<int> used_numa_id;
    for (size_t dev_id = 0; dev_id < local_topology->getNicCount(); ++dev_id) {
        auto entry = local_topology->getNicEntry(dev_id);
        if (!entry || entry->type != Topology::NIC_RDMA) continue;
        DeviceInfo& info = devices_[dev_id];
        info.dev_id = dev_id;
        info.bw_gbps = entry->bw_gbps;
        info.numa_id = entry->numa_node;
        info.ewma_bandwidth_bps.store(info.getTheoreticalBandwidth(),
                                      std::memory_order_relaxed);
        used_numa_id.insert(entry->numa_node);
    }
    if (used_numa_id.size() == 1) allow_cross_numa_ = true;
    return Status::OK();
}

Status DeviceQuota::enableSharedQuota(const std::string& shm_name) {
    shared_quota_ = std::make_shared<SharedQuotaManager>(this);
    auto status = shared_quota_->attach(shm_name);
    if (!status.ok()) shared_quota_.reset();
    return status;
}

Status DeviceQuota::allocate(uint64_t total_length, uint32_t num_slices,
                             const std::string& location,
                             std::vector<int>& slice_dev_ids, int priority,
                             uint64_t device_mask) {
    slice_dev_ids.clear();
    auto entry = local_topology_->getMemEntry(location);
    if (!entry) return Status::InvalidArgument("Unknown location");

    uint64_t slice_bytes = (total_length + num_slices - 1) / num_slices;

    struct Candidate {
        int dev_id;
        double bw_bps;
        uint64_t inflight;
        double score;
        bool is_cross_numa;
    };
    std::vector<Candidate> candidates;

    bool found_rank0 = false;
    for (size_t rank = 0; rank < Topology::DevicePriorityRanks; ++rank) {
        for (int dev_id : entry->device_list[rank]) {
            if (!devices_.count(dev_id) || !(device_mask & (1ULL << dev_id)))
                continue;

            auto& dev = devices_[dev_id];
            double effective_bw = dev.getEwmaBandwidth();
            bool is_cross = (rank > 0);
            if (is_cross) effective_bw *= sched_params_.cross_numa_penalty;

            uint64_t current_inflight = dev.getInflightBytes();
            double score = effective_bw / (current_inflight + slice_bytes);

            candidates.push_back(
                {dev_id, effective_bw, current_inflight, score, is_cross});
            if (rank == 0) found_rank0 = true;
        }
        if (found_rank0 && sched_params_.cross_numa_max_ratio <= 0) break;
    }

    if (candidates.empty())
        return Status::DeviceNotFound("no eligible devices");

    bool use_multi_path = (num_slices >= sched_params_.batch_threshold) &&
                          (candidates.size() > 1);
    if (!use_multi_path) {
        double max_score = 0;
        for (auto& c : candidates) max_score = std::max(max_score, c.score);
        std::vector<size_t> best_indices;
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (candidates[i].score >= max_score * 0.9)
                best_indices.push_back(i);
        }
        size_t idx =
            best_indices[SimpleRandom::Get().next(best_indices.size())];
        const auto& target = candidates[idx];
        for (uint32_t i = 0; i < num_slices; ++i)
            slice_dev_ids.push_back(target.dev_id);
        updateStats(target.dev_id, total_length, target.is_cross_numa);
    } else {
        std::vector<double> cumulative_scores;
        double total_score = 0;
        for (auto& c : candidates) {
            total_score += c.score;
            cumulative_scores.push_back(total_score);
        }
        thread_local uint64_t tl_batch_offset = 0;
        double offset = (tl_batch_offset++ * slice_bytes) / total_score;
        for (uint32_t i = 0; i < num_slices; ++i) {
            double pos =
                fmod(offset + i * (total_score / num_slices), total_score);
            auto it = std::lower_bound(cumulative_scores.begin(),
                                       cumulative_scores.end(), pos);
            size_t idx = std::distance(cumulative_scores.begin(), it);
            if (idx >= candidates.size()) idx = candidates.size() - 1;

            slice_dev_ids.push_back(candidates[idx].dev_id);
            updateStats(candidates[idx].dev_id, slice_bytes,
                        candidates[idx].is_cross_numa);
        }
    }
    return Status::OK();
}

void DeviceQuota::updateStats(int dev_id, uint64_t bytes, bool is_cross) {
    auto& dev = devices_[dev_id];
    dev.addInflight(bytes);
    dev.total_bytes.fetch_add(bytes, std::memory_order_relaxed);
    if (is_cross) {
        total_cross_numa_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    } else {
        total_local_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    }
}

static void printTrafficStats(
    std::unordered_map<int, DeviceQuota::DeviceInfo>& devices) {
    static std::atomic<uint64_t> last_print_time_ns{0};

    uint64_t now = getCurrentTimeInNano();
    uint64_t expected_last = last_print_time_ns.load(std::memory_order_relaxed);

    if (expected_last == 0) {
        last_print_time_ns.store(now, std::memory_order_relaxed);
        return;
    }

    // Print every 1 second (only one thread will succeed)
    if (now - expected_last >= 1000000000ULL) {
        // Try to claim the print slot
        if (last_print_time_ns.compare_exchange_strong(
                expected_last, now, std::memory_order_relaxed,
                std::memory_order_relaxed)) {
            std::ostringstream oss;
            oss << "[RDMA Traffic] ";
            bool first = true;
            for (auto& [dev_id, dev] : devices) {
                uint64_t total =
                    dev.total_bytes.load(std::memory_order_relaxed);
                uint64_t last =
                    dev.last_second_bytes.load(std::memory_order_relaxed);
                uint64_t delta = total - last;
                double mbps = delta * 8.0 / 1e6;  // Mb/s

                dev.last_second_bytes.store(total, std::memory_order_relaxed);

                // Skip devices with zero traffic
                if (mbps < 0.01) continue;

                if (!first) oss << " | ";
                oss << "dev" << dev_id << ": " << std::fixed
                    << std::setprecision(2) << mbps << " Mb/s";
                first = false;
            }

            // Only print if there's actual traffic
            if (!first) {
                LOG(INFO) << oss.str();
            }
        }
    }
}

Status DeviceQuota::release(int dev_id, uint64_t length, double latency,
                            int priority) {
    // Print traffic stats periodically
    printTrafficStats(devices_);

    if (!enable_quota_) return Status::OK();

    auto it = devices_.find(dev_id);
    if (it == devices_.end())
        return Status::InvalidArgument("device not found");

    auto& dev = it->second;

    // Release inflight bytes
    dev.subInflight(length);

    // Early exit if EWMA learning disabled
    if (!sched_params_.enable_ewma_learning) return Status::OK();

    // Filter out small transfers (noise reduction)
    if (length < 4096 || latency < 1e-6) return Status::OK();

    // Calculate observed bandwidth
    double observed_bps = length / latency;

    // Update EWMA bandwidth: new = alpha * old + (1 - alpha) * observed
    double alpha = sched_params_.ewma_alpha;
    double current_ewma = dev.getEwmaBandwidth();
    double new_ewma = alpha * current_ewma + (1.0 - alpha) * observed_bps;

    // Clamp to reasonable bounds (10% to 300% of theoretical)
    double min_bw = dev.getTheoreticalBandwidth() * 0.1;
    double max_bw = dev.getTheoreticalBandwidth() * 3.0;
    new_ewma = std::clamp(new_ewma, min_bw, max_bw);

    dev.ewma_bandwidth_bps.store(new_ewma, std::memory_order_relaxed);

    // Periodic diffusion for shared quota (if enabled)
    if (shared_quota_) {
        thread_local uint64_t tl_last_ts = 0;
        uint64_t now = getCurrentTimeInNano();
        uint64_t diffusion_interval = 10 * 1000000ull;  // 10ms
        if (now - tl_last_ts > diffusion_interval) {
            tl_last_ts = now;
            return shared_quota_->diffusion();
        }
    }

    return Status::OK();
}

}  // namespace tent
}  // namespace mooncake