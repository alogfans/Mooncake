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
        info.bw_gbps = 200.0;
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
    double beta0{0.0};
    double beta1{1.0};

    // Sample window for percentile calculation
    static constexpr uint32_t kSampleWindow = 100;
    double beta0_samples[kSampleWindow]{0};
    double beta1_samples[kSampleWindow]{0};
    uint32_t sample_idx{0};
    bool window_full{false};

    // Helper: get both p05 and p95 percentiles in one pass (assumes
    // window_full)
    void getBeta0Percentiles(double& out_p05, double& out_p95) {
        if (!window_full) {
            out_p05 = 5e-4;
            out_p95 = 5e-4;
            return;
        }
        size_t idx_p05 = (kSampleWindow - 1) * 0.05;
        size_t idx_p95 = (kSampleWindow - 1) * 0.95;
        std::nth_element(beta0_samples, beta0_samples + idx_p05,
                         beta0_samples + kSampleWindow);
        out_p05 = beta0_samples[idx_p05];
        std::nth_element(beta0_samples + idx_p05 + 1, beta0_samples + idx_p95,
                         beta0_samples + kSampleWindow);
        out_p95 = beta0_samples[idx_p95];
    }

    void getBeta1Percentiles(double& out_p05, double& out_p95) {
        if (!window_full) {
            out_p05 = 2.0;
            out_p95 = 2.0;
            return;
        }
        size_t idx_p05 = (kSampleWindow - 1) * 0.05;
        size_t idx_p95 = (kSampleWindow - 1) * 0.95;
        std::nth_element(beta1_samples, beta1_samples + idx_p05,
                         beta1_samples + kSampleWindow);
        out_p05 = beta1_samples[idx_p05];
        std::nth_element(beta1_samples + idx_p05 + 1, beta1_samples + idx_p95,
                         beta1_samples + kSampleWindow);
        out_p95 = beta1_samples[idx_p95];
    }

    // Add sample to window
    void addSample(double b0, double b1) {
        beta0_samples[sample_idx] = b0;
        beta1_samples[sample_idx] = b1;
        sample_idx = (sample_idx + 1) % kSampleWindow;
        if (sample_idx == 0) window_full = true;
    }
};

thread_local std::unordered_map<int, TlsDeviceInfo> tl_device_info;

Status DeviceQuota::allocate(uint64_t length, const std::string& location,
                             int& chosen_dev_id) {
    auto entry = local_topology_->getMemEntry(location);
    if (!entry) return Status::InvalidArgument("Unknown location" LOC_MARK);

    if (!enable_quota_) {
        thread_local int id = 0;
        for (size_t rank = 0; rank < Topology::DevicePriorityRanks; ++rank) {
            auto& list = entry->device_list[rank];
            if (list.empty()) continue;
            chosen_dev_id = list[id % list.size()];
            id++;
            return Status::OK();
        }
        return Status::DeviceNotFound("no eligible devices for " + location);
    }

    static constexpr double penalty[] = {1.0, 3.0, 10.0};
    const double w = local_weight_;
    std::unordered_map<int, double> score_map;
    bool found_device = false;
    double best_score = std::numeric_limits<double>::infinity();
    for (size_t rank = 0; rank < Topology::DevicePriorityRanks; ++rank) {
        if (rank == Topology::DevicePriorityRanks - 1 && !allow_cross_numa_ &&
            found_device)
            continue;
        for (int dev_id : entry->device_list[rank]) {
            if (!devices_.count(dev_id)) continue;
            auto& dev = devices_[dev_id];
            auto& tl_dev = tl_device_info[dev_id];
            uint64_t overall_active_bytes =
                dev.diffusion_active_bytes.load(std::memory_order_relaxed) +
                dev.active_bytes.load(std::memory_order_relaxed);
            double weighted_active = w * tl_dev.active_bytes +
                                     (1.0 - w) * overall_active_bytes + length;
            double beta0_g = dev.beta0.load(std::memory_order_relaxed);
            double beta1_g = dev.beta1.load(std::memory_order_relaxed);
            double beta0 = w * tl_dev.beta0 + (1.0 - w) * beta0_g;
            double beta1 = w * tl_dev.beta1 + (1.0 - w) * beta1_g;
            double bw = dev.bw_gbps * 1e9 / 8;
            double predicted_time = (weighted_active / bw) * beta1 + beta0;
            double score = penalty[rank] * predicted_time;

            // QoS penalty: lower priority processes get penalized when higher
            // priority processes are using the device
            if (shared_quota_ && priority_ > PRIO_HIGH) {
                uint64_t high_load = shared_quota_->getHighPrioLoad(dev_id);
                uint64_t med_load = shared_quota_->getMediumPrioLoad(dev_id);

                // Threshold: consider device "contended" if > 1MB used by
                // higher prio
                constexpr uint64_t QOS_THRESHOLD = 1024 * 1024;

                if (priority_ == PRIO_MEDIUM) {
                    if (high_load > QOS_THRESHOLD) {
                        score *= 2.0;
                    }
                } else if (priority_ == PRIO_LOW) {
                    if (high_load > QOS_THRESHOLD || med_load > QOS_THRESHOLD) {
                        score *= 5.0;
                    }
                }
            }

            score_map[dev_id] = score;

            // Idle discount: give unused devices a chance to be re-evaluated
            uint64_t last_ns =
                dev.last_update_ns.load(std::memory_order_relaxed);
            if (last_ns > 0) {
                uint64_t idle_ns = getFastTimeNanos() - last_ns;
                if (idle_ns > 10e8) {  // idle > 1 seconds
                    double discount =
                        std::min(0.4, idle_ns / 60e8);  // max 40% off
                    score_map[dev_id] *= (1.0 - discount);
                }
            }
            best_score = std::min(best_score, score_map[dev_id]);
            found_device = true;
        }
    }

    if (!found_device) {
        return Status::DeviceNotFound("no eligible devices for " + location);
    }

    std::vector<int> filtered;
    for (const auto& [dev_id, score] : score_map) {
        if (score <= best_score * 1.05) filtered.push_back(dev_id);
    }

    std::sort(filtered.begin(), filtered.end(), [&](int a, int b) {
        if (std::abs(score_map[a] - score_map[b]) > 1e-9)
            return score_map[a] < score_map[b];
        return a < b;
    });

    thread_local size_t rr_index = 0;
    chosen_dev_id = filtered[rr_index % filtered.size()];
    rr_index++;

    tl_device_info[chosen_dev_id].active_bytes += length;
    if (local_weight_ < 1 - 1e-6)
        devices_[chosen_dev_id].active_bytes.fetch_add(
            length, std::memory_order_relaxed);
    return Status::OK();
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

    double bw = dev.bw_gbps * 1e9 / 8;
    double theory_time = static_cast<double>(length) / bw;
    double obs_time = latency;

    const double w = local_weight_;
    double beta0_g = dev.beta0.load(std::memory_order_relaxed);
    double beta1_g = dev.beta1.load(std::memory_order_relaxed);
    double beta0 = w * tl_dev.beta0 + (1.0 - w) * beta0_g;
    double beta1 = w * tl_dev.beta1 + (1.0 - w) * beta1_g;

    double pred_time = beta0 + beta1 * theory_time;
    double err = obs_time - pred_time;
    double rel_err = (pred_time > 1e-9) ? (err / pred_time) : 0.0;

    double adapt_alpha = alpha_;
    if (std::abs(err) > 0.05 * pred_time)
        adapt_alpha = std::min(1.0, alpha_ * 5.0);

    double delta0 = adapt_alpha * err;
    double delta1 = adapt_alpha * rel_err;

    double new_beta0_l = tl_dev.beta0 + w * delta0;
    double new_beta1_l = tl_dev.beta1 * (1.0 + w * delta1);
    double new_beta0_g = beta0_g + (1.0 - w) * delta0;
    double new_beta1_g = beta1_g * (1.0 + (1.0 - w) * delta1);

    // Compute percentile bounds once (shared by local and global)
    double beta0_min, beta0_max, beta1_min, beta1_max;
    if (use_robust_clamp_) {
        tl_dev.addSample(new_beta0_l, new_beta1_l);

        double p95_beta0, p05_beta0, p95_beta1, p05_beta1;
        tl_dev.getBeta0Percentiles(p05_beta0, p95_beta0);
        tl_dev.getBeta1Percentiles(p05_beta1, p95_beta1);

        const double headroom = 1.1;
        beta0_min = p05_beta0;
        beta0_max = std::min(p95_beta0 * headroom, 5e-4);
        beta1_min = p05_beta1;
        beta1_max = std::min(p95_beta1 * headroom, 20.0);
    } else {
        beta0_min = 0.0;
        beta0_max = 5e-4;
        beta1_min = 0.5;
        beta1_max = 20.0;
    }

    // Apply clamping to local beta
    tl_dev.beta0 = std::clamp(new_beta0_l, beta0_min, beta0_max);
    tl_dev.beta1 = std::clamp(new_beta1_l, beta1_min, beta1_max);

    // Apply clamping to global beta (if needed)
    if (local_weight_ < 1 - 1e-6) {
        dev.last_update_ns.store(getFastTimeNanos(), std::memory_order_relaxed);
        dev.beta0.store(std::clamp(new_beta0_g, beta0_min, beta0_max),
                        std::memory_order_relaxed);
        dev.beta1.store(std::clamp(new_beta1_g, beta1_min, beta1_max),
                        std::memory_order_relaxed);

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