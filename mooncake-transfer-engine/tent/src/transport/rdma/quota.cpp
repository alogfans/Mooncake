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

#include <assert.h>
#include <unordered_set>

namespace mooncake {
namespace tent {
Status DeviceQuota::loadTopology(std::shared_ptr<Topology>& local_topology) {
    local_topology_ = local_topology;
    std::unordered_set<int> used_numa_id;
    for (size_t dev_id = 0; dev_id < local_topology->getNicCount(); ++dev_id) {
        auto entry = local_topology->getNicEntry(dev_id);
        if (entry->type != Topology::NIC_RDMA) continue;
        DeviceInfo& info = devices_[dev_id];
        info.dev_id = dev_id;
        info.bw_gbps = entry->bw_gbps;
        info.numa_id = entry->numa_node;
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

struct TlsDeviceInfo {
    uint64_t active_bytes{0};
    double ewma_bandwidth_bps{0.0};
};

thread_local std::unordered_map<int, TlsDeviceInfo> tl_device_info;
thread_local std::vector<int> tl_eligible;
thread_local size_t tl_rr_counter = 0;

Status DeviceQuota::allocate(uint64_t total_length, uint32_t num_slices,
                             const std::string& location,
                             std::vector<int>& slice_dev_ids,
                             uint64_t device_mask) {
    slice_dev_ids.clear();
    slice_dev_ids.reserve(num_slices);

    auto entry = local_topology_->getMemEntry(location);
    if (!entry) return Status::InvalidArgument("Unknown location" LOC_MARK);

    uint64_t block_size = calculateBlockSize(total_length, num_slices, 1);
    uint64_t slice_bytes = (total_length + num_slices - 1) / num_slices;

    if (!enable_quota_) {
        tl_eligible.clear();
        for (size_t rank = 0; rank < Topology::DevicePriorityRanks; ++rank) {
            for (int dev_id : entry->device_list[rank]) {
                if (!devices_.count(dev_id)) continue;
                if ((device_mask & (1ULL << dev_id)) == 0) continue;
                tl_eligible.push_back(dev_id);
            }
            if (!tl_eligible.empty()) break;
        }

        if (tl_eligible.empty()) {
            return Status::DeviceNotFound("no eligible devices for " + location);
        }

        uint64_t offset = 0;
        for (uint32_t i = 0; i < num_slices; ++i) {
            int dev_id = tl_eligible[tl_rr_counter % tl_eligible.size()];
            tl_rr_counter++;
            slice_dev_ids.push_back(dev_id);
            uint64_t actual_slice_bytes = getSliceLength(total_length, offset, block_size);
            offset += actual_slice_bytes;
            tl_device_info[dev_id].active_bytes += actual_slice_bytes;
            if (local_weight_ < 1 - 1e-6) {
                devices_[dev_id].active_bytes.fetch_add(
                    actual_slice_bytes, std::memory_order_relaxed);
            }
        }
        return Status::OK();
    }

    static constexpr double penalty[] = {1.0, 3.0, 10.0};
    const double w = local_weight_;
    std::vector<Candidate> candidates;
    bool found_device = false;

    for (size_t rank = 0; rank < Topology::DevicePriorityRanks; ++rank) {
        if (rank == Topology::DevicePriorityRanks - 1 && !allow_cross_numa_ &&
            found_device)
            continue;
        for (int dev_id : entry->device_list[rank]) {
            if (!devices_.count(dev_id)) continue;
            if ((device_mask & (1ULL << dev_id)) == 0) continue;
            auto& dev = devices_[dev_id];
            auto& tl_dev = tl_device_info[dev_id];
            uint64_t overall_active_bytes =
                dev.diffusion_active_bytes.load(std::memory_order_relaxed) +
                dev.active_bytes.load(std::memory_order_relaxed);
            double weighted_active = w * tl_dev.active_bytes +
                                     (1.0 - w) * overall_active_bytes + slice_bytes;

            double bw_bps = dev.bw_gbps * 1e9 / 8;
            double predicted_time = weighted_active / bw_bps;
            double score = penalty[rank] * predicted_time;
            candidates.push_back({dev_id, score, dev.bw_gbps});
            found_device = true;
        }
    }

    if (!found_device) {
        return Status::DeviceNotFound("no eligible devices for " + location);
    }

    bool use_multi_path = (num_slices >= batch_threshold_) && (candidates.size() > 1);

    if (!use_multi_path) {
        selectSinglePath(candidates, num_slices, total_length, slice_dev_ids);
    } else {
        selectMultiPath(candidates, num_slices, total_length, slice_dev_ids);
    }

    return Status::OK();
}

void DeviceQuota::selectSinglePath(const std::vector<Candidate>& candidates,
                                   uint32_t num_slices, uint64_t total_length,
                                   std::vector<int>& slice_dev_ids) {
    double max_score = 0;
    for (const auto& c : candidates) max_score = std::max(max_score, c.score);
    std::vector<size_t> best_indices;
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (candidates[i].score >= max_score * 0.9) best_indices.push_back(i);
    }

    uint64_t block_size = calculateBlockSize(total_length, num_slices, 1);
    uint64_t offset = 0;
    for (uint32_t i = 0; i < num_slices; ++i) {
        size_t idx = best_indices[tl_rr_counter++ % best_indices.size()];
        slice_dev_ids.push_back(candidates[idx].dev_id);
        uint64_t actual_slice_bytes = getSliceLength(total_length, offset, block_size);
        offset += actual_slice_bytes;
        tl_device_info[candidates[idx].dev_id].active_bytes += actual_slice_bytes;
        if (local_weight_ < 1 - 1e-6) {
            devices_[candidates[idx].dev_id].active_bytes.fetch_add(
                actual_slice_bytes, std::memory_order_relaxed);
        }
    }
}

void DeviceQuota::selectMultiPath(const std::vector<Candidate>& candidates,
                                  uint32_t num_slices, uint64_t total_length,
                                  std::vector<int>& slice_dev_ids) {
    uint64_t block_size = calculateBlockSize(total_length, num_slices, 1);

    std::vector<double> cumulative_scores;
    double total_score = 0;
    for (const auto& c : candidates) {
        double inverted_score = 1.0 / (c.score + 1e-9);
        total_score += inverted_score;
        cumulative_scores.push_back(total_score);
    }

    thread_local uint64_t tl_batch_offset = 0;
    uint64_t slice_bytes = (total_length + num_slices - 1) / num_slices;
    double weight_offset = (tl_batch_offset++ * slice_bytes) / total_score;

    uint64_t offset = 0;
    for (uint32_t i = 0; i < num_slices; ++i) {
        double pos = fmod(weight_offset + i * (total_score / num_slices), total_score);
        auto it = std::lower_bound(cumulative_scores.begin(),
                                   cumulative_scores.end(), pos);
        size_t idx = std::distance(cumulative_scores.begin(), it);
        if (idx >= candidates.size()) idx = candidates.size() - 1;

        const auto& selected = candidates[idx];
        slice_dev_ids.push_back(selected.dev_id);
        uint64_t actual_slice_bytes = getSliceLength(total_length, offset, block_size);
        offset += actual_slice_bytes;
        tl_device_info[selected.dev_id].active_bytes += actual_slice_bytes;
        if (local_weight_ < 1 - 1e-6) {
            devices_[selected.dev_id].active_bytes.fetch_add(
                actual_slice_bytes, std::memory_order_relaxed);
        }
    }
}

Status DeviceQuota::release(int dev_id, uint64_t length, double latency) {
    if (!enable_quota_) return Status::OK();
    auto it = devices_.find(dev_id);
    if (it == devices_.end())
        return Status::InvalidArgument("device not found");

    auto& dev = it->second;
    auto& tl_dev = tl_device_info[dev_id];

    if (local_weight_ < 1 - 1e-6)
        dev.active_bytes.fetch_sub(length, std::memory_order_relaxed);
    tl_dev.active_bytes -= length;

    if (!update_quota_params_) return Status::OK();

    double obs_bw_bps = (latency > 1e-9) ? (length / latency) : 0.0;

    double& ewma_bw = tl_dev.ewma_bandwidth_bps;
    double hw_bw_bps = dev.bw_gbps * 1e9 / 8;

    if (ewma_bw < 1e-9) {
        ewma_bw = hw_bw_bps;
    } else {
        double adapt_alpha = alpha_;
        ewma_bw = (1.0 - adapt_alpha) * ewma_bw + adapt_alpha * obs_bw_bps;
        ewma_bw = std::clamp(ewma_bw, hw_bw_bps * 0.5, hw_bw_bps * 2.0);
    }

    if (shared_quota_) {
        thread_local uint64_t tl_last_ts = 0;
        uint64_t now = getCurrentTimeInNano();
        if (now - tl_last_ts > diffusion_interval_) {
            tl_last_ts = now;
            return shared_quota_->diffusion();
        }
    }
    return Status::OK();
}

}  // namespace tent
}  // namespace mooncake