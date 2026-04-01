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
#include <limits>
#include <unordered_set>
#include <thread>

namespace mooncake {
namespace tent {

// ========== Thread-local device state ==========
struct TlsDeviceInfo {
    uint64_t active_bytes_by_prio[3] = {0, 0, 0};  // [high, medium, low]
    double beta0{0.01};                            // Fixed overhead (μs)
    double beta1{1.00};  // Bandwidth correction factor

    // Adaptive bounds for beta1
    double beta1_min_observed{0.5};
    double beta1_max_observed{5.0};
    uint32_t sample_count{0};
    static constexpr double kDecayFactor = 0.98;
    static constexpr uint32_t kWarmupSamples = 50;
};

thread_local std::unordered_map<int, TlsDeviceInfo> tl_device_info;

// ========== Per-location thread-local state ==========
struct LocationState {
    uint64_t location_hash{0};
    int local_count{0};
    int first_local_dev{-1};
    int preferred_dev{-1};
    bool initialized{false};
    bool overloaded{false};  // cached overload state
    uint64_t last_overload_check_ns{0};
};

Status DeviceQuota::loadTopology(std::shared_ptr<Topology> &local_topology) {
    local_topology_ = local_topology;
    std::unordered_set<int> used_numa_id;
    for (size_t dev_id = 0; dev_id < local_topology->getNicCount(); ++dev_id) {
        auto entry = local_topology->getNicEntry(dev_id);
        if (!entry || entry->type != Topology::NIC_RDMA) continue;
        DeviceInfo &info = devices_[dev_id];
        info.dev_id = dev_id;
        info.bw_gbps = entry->bw_gbps;
        info.numa_id = entry->numa_node;
        used_numa_id.insert(entry->numa_node);
    }
    if (used_numa_id.size() == 1) allow_cross_numa_ = true;
    return Status::OK();
}

Status DeviceQuota::enableSharedQuota(const std::string &shm_name) {
    shared_quota_ = std::make_shared<SharedQuotaManager>(this);
    auto status = shared_quota_->attach(shm_name);
    if (!status.ok()) shared_quota_.reset();
    return status;
}

Status DeviceQuota::allocate(uint64_t length, const std::string &location,
                             int &chosen_dev_id, int priority,
                             uint64_t device_mask) {
    auto entry = local_topology_->getMemEntry(location);
    if (!entry) return Status::InvalidArgument("Unknown location" LOC_MARK);
    static constexpr double kPenalty[] = {1.0, 5.0, 50.0};
    static constexpr double kTieEpsilon = 0.01;  // 1% tolerance for ties
    struct Candidate {
        int dev_id;
        double score;
    };
    thread_local std::vector<Candidate> candidates;
    candidates.clear();
    double best_score = 1e300;
    double local_best_score = 1e300;
    for (size_t rank = 0; rank < Topology::DevicePriorityRanks; ++rank) {
        for (int dev_id : entry->device_list[rank]) {
            if (!devices_.count(dev_id)) continue;
            if ((device_mask & (1ULL << dev_id)) == 0) continue;
            auto &dev = devices_[dev_id];
            auto &tl = tl_device_info[dev_id];
            uint64_t total = tl.active_bytes_by_prio[0] +
                             tl.active_bytes_by_prio[1] +
                             tl.active_bytes_by_prio[2] + length;
            double theory_time_us = total / dev.getBandwidthBytesPerSec() * 1e6;
            double pred_time_us = tl.beta0 + tl.beta1 * theory_time_us;
            double score = kPenalty[rank] * pred_time_us;
            if (rank == 0 && pred_time_us < local_best_score) {
                local_best_score = pred_time_us;
            }
            if (rank > 0 && local_best_score < 1e200) {
                constexpr double kCrossNumaThreshold = 0.7;
                if (score >= local_best_score * kCrossNumaThreshold) continue;
            }
            if (score < best_score) {
                best_score = score;
                candidates.clear();
                candidates.push_back({dev_id, score});
            } else if (score <= best_score * (1.0 + kTieEpsilon)) {
                candidates.push_back({dev_id, score});
            }
        }
    }
    if (candidates.empty())
        return Status::DeviceNotFound("no eligible devices");
    uint32_t idx = SimpleRandom::Get().next(candidates.size());
    chosen_dev_id = candidates[idx].dev_id;
    tl_device_info[chosen_dev_id].active_bytes_by_prio[priority] += length;
    if (local_weight_ < 1 - 1e-6)
        devices_[chosen_dev_id].addBytes(priority, length);
    return Status::OK();
}

Status DeviceQuota::release(int dev_id, uint64_t length, double latency,
                            int priority) {
    if (!enable_quota_) return Status::OK();

    auto it = devices_.find(dev_id);
    if (it == devices_.end())
        return Status::InvalidArgument("device not found");

    auto &dev = it->second;
    auto &tl_dev = tl_device_info[dev_id];

    // Update active bytes
    tl_dev.active_bytes_by_prio[priority] -= length;
    if (local_weight_ < 1 - 1e-6) dev.subBytes(priority, length);

    // Early exit if latency learning disabled
    if (!sched_params_.enable_latency_learning || !update_quota_params_)
        return Status::OK();

    // Filter out small transfers (noise reduction)
    if (length < 4096 || latency < 1e-6) return Status::OK();

    // Update beta parameters with exponential smoothing
    constexpr double US = 1e6;
    double obs_time_us = latency * US;
    double theory_time_us =
        (tl_dev.active_bytes_by_prio[0] + tl_dev.active_bytes_by_prio[1] +
         tl_dev.active_bytes_by_prio[2]) /
        dev.getBandwidthBytesPerSec() * US;
    double pred_time = tl_dev.beta0 + tl_dev.beta1 * theory_time_us;
    double err = obs_time_us - pred_time;
    double rel_err = (theory_time_us > 1e-3) ? (err / theory_time_us) : 0.0;

    double alpha = sched_params_.alpha;
    double new_beta1 = tl_dev.beta1 * (1.0 + alpha * rel_err);
    tl_dev.beta0 += alpha * err;

    // Adaptive bounds tracking
    tl_dev.sample_count++;
    if (tl_dev.sample_count <= TlsDeviceInfo::kWarmupSamples) {
        tl_dev.beta1_min_observed = std::min(tl_dev.beta1_min_observed, new_beta1);
        tl_dev.beta1_max_observed = std::max(tl_dev.beta1_max_observed, new_beta1);
    } else {
        tl_dev.beta1_min_observed =
            TlsDeviceInfo::kDecayFactor * tl_dev.beta1_min_observed +
            (1 - TlsDeviceInfo::kDecayFactor) *
                std::min(tl_dev.beta1_min_observed, new_beta1);
        tl_dev.beta1_max_observed =
            TlsDeviceInfo::kDecayFactor * tl_dev.beta1_max_observed +
            (1 - TlsDeviceInfo::kDecayFactor) *
                std::max(tl_dev.beta1_max_observed, new_beta1);
    }

    // Clamp to adaptive bounds
    double beta1_min = std::max(0.1, tl_dev.beta1_min_observed * 0.5);
    double beta1_max = tl_dev.beta1_max_observed * 1.5;
    tl_dev.beta0 = std::clamp(tl_dev.beta0, 0.0, 500.0);
    tl_dev.beta1 = std::clamp(new_beta1, beta1_min, beta1_max);

    // Update global state
    if (local_weight_ < 1 - 1e-6) {
        dev.last_update_ns.store(getFastTimeNanos(), std::memory_order_relaxed);
        dev.beta0.store(tl_dev.beta0, std::memory_order_relaxed);
        dev.beta1.store(tl_dev.beta1, std::memory_order_relaxed);

        // Periodic diffusion for shared quota
        if (shared_quota_) {
            thread_local uint64_t tl_last_ts = 0;
            uint64_t now = getCurrentTimeInNano();
            if (now - tl_last_ts > diffusion_interval_) {
                tl_last_ts = now;
                return shared_quota_->diffusion();
            }
        }
    }

    return Status::OK();
}

}  // namespace tent
}  // namespace mooncake