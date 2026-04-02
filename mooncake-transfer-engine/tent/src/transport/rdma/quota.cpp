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
#include <iostream>
#include <iomanip>

namespace mooncake {
namespace tent {

// ========== NUMA node cache helper ==========
static int getNumaNodeForLocation(const std::string& location,
                                   std::shared_ptr<Topology>& topology,
                                   std::unordered_map<std::string, int>& cache,
                                   std::mutex& cache_lock) {
    // Check cache first
    {
        std::lock_guard<std::mutex> lock(cache_lock);
        auto it = cache.find(location);
        if (it != cache.end()) return it->second;
    }

    // Lookup from topology
    int numa_node = -1;
    auto entry = topology->getMemEntry(location);
    if (entry) numa_node = entry->numa_node;

    // Update cache
    {
        std::lock_guard<std::mutex> lock(cache_lock);
        cache[location] = numa_node;
    }

    return numa_node;
}

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
        // Initialize EWMA bandwidth with theoretical value
        info.ewma_bandwidth_bps.store(info.getTheoreticalBandwidth(),
                                      std::memory_order_relaxed);
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

// Thread-local round-robin counter for enable_quota=false mode
thread_local uint64_t tl_allocate_count = 0;

Status DeviceQuota::allocate(uint64_t length, const std::string &location,
                             int &chosen_dev_id, int priority,
                             uint64_t device_mask) {
    auto entry = local_topology_->getMemEntry(location);
    if (!entry) return Status::InvalidArgument("Unknown location" LOC_MARK);

    // When quota is disabled, use simple round-robin
    if (!enable_quota_) {
        std::vector<int> eligible_devices;
        for (size_t rank = 0; rank < Topology::DevicePriorityRanks; ++rank) {
            for (int dev_id : entry->device_list[rank]) {
                if (!devices_.count(dev_id)) continue;
                if ((device_mask & (1ULL << dev_id)) == 0) continue;
                eligible_devices.push_back(dev_id);
            }
        }
        if (eligible_devices.empty())
            return Status::DeviceNotFound("no eligible devices");

        chosen_dev_id = eligible_devices[tl_allocate_count % eligible_devices.size()];
        tl_allocate_count++;
        devices_[chosen_dev_id].total_bytes.fetch_add(length, std::memory_order_relaxed);
        return Status::OK();
    }

    // ========== EWMA-based NUMA-aware scheduling ==========

    struct Candidate {
        int dev_id;
        double pred_time_us;
        bool is_cross_numa;
    };

    std::vector<Candidate> local_candidates;   // rank 0 devices (local NUMA)

    // First pass: evaluate local NUMA devices (rank 0) ONLY
    for (int dev_id : entry->device_list[0]) {
        if (!devices_.count(dev_id)) continue;
        if ((device_mask & (1ULL << dev_id)) == 0) continue;

        auto &dev = devices_[dev_id];
        uint64_t inflight = dev.getInflightBytes();
        double bw_bps = dev.getEwmaBandwidth();

        // Predict completion time: (inflight + length) / bandwidth
        double pred_time_us = (inflight + length) / bw_bps * 1e6;

        local_candidates.push_back({dev_id, pred_time_us, false});
    }

    // If we have local candidates, prefer them
    if (!local_candidates.empty()) {
        // Find best with 10% tolerance
        static constexpr double kTieEpsilon = 0.10;
        double best_score = local_candidates[0].pred_time_us;
        for (const auto& cand : local_candidates) {
            if (cand.pred_time_us < best_score) {
                best_score = cand.pred_time_us;
            }
        }

        std::vector<int> best_candidates;
        for (const auto& cand : local_candidates) {
            if (cand.pred_time_us <= best_score * (1.0 + kTieEpsilon)) {
                best_candidates.push_back(cand.dev_id);
            }
        }

        uint32_t idx = SimpleRandom::Get().next(best_candidates.size());
        chosen_dev_id = best_candidates[idx];

        devices_[chosen_dev_id].addInflight(length);
        devices_[chosen_dev_id].total_bytes.fetch_add(length, std::memory_order_relaxed);
        total_local_bytes_.fetch_add(length, std::memory_order_relaxed);

        return Status::OK();
    }

    // ========== Fallback: consider cross-NUMA only if completely idle ==========

    // Check cross-NUMA usage ratio
    uint64_t local_bytes = total_local_bytes_.load(std::memory_order_relaxed);
    uint64_t cross_bytes = total_cross_numa_bytes_.load(std::memory_order_relaxed);
    uint64_t total_bytes_tracked = local_bytes + cross_bytes;

    bool can_use_cross_numa = false;
    if (sched_params_.cross_numa_max_ratio <= 0) {
        // Cross-NUMA disabled
        can_use_cross_numa = false;
    } else if (total_bytes_tracked == 0) {
        // No stats yet, allow small amount
        can_use_cross_numa = true;
    } else {
        double current_ratio = (double)cross_bytes / total_bytes_tracked;
        can_use_cross_numa = (current_ratio < sched_params_.cross_numa_max_ratio);
    }

    std::vector<Candidate> cross_numa_candidates;

    if (can_use_cross_numa) {
        // Check rank 1/2 devices, but only if they are IDLE
        for (size_t rank = 1; rank < Topology::DevicePriorityRanks; ++rank) {
            for (int dev_id : entry->device_list[rank]) {
                if (!devices_.count(dev_id)) continue;
                if ((device_mask & (1ULL << dev_id)) == 0) continue;

                auto &dev = devices_[dev_id];

                // ONLY use if completely idle
                if (dev.getInflightBytes() > 0) continue;

                double bw_bps = dev.getEwmaBandwidth();
                double effective_bw = bw_bps * sched_params_.cross_numa_penalty;
                double pred_time_us = length / effective_bw * 1e6;

                cross_numa_candidates.push_back({dev_id, pred_time_us, true});
            }
        }
    }

    // Combine: prefer local, then idle cross-NUMA
    std::vector<Candidate> all_candidates;
    all_candidates.insert(all_candidates.end(), local_candidates.begin(), local_candidates.end());
    all_candidates.insert(all_candidates.end(), cross_numa_candidates.begin(), cross_numa_candidates.end());

    if (all_candidates.empty())
        return Status::DeviceNotFound("no eligible devices");

    // Select best
    static constexpr double kTieEpsilon = 0.10;
    double best_score = all_candidates[0].pred_time_us;
    for (const auto& cand : all_candidates) {
        if (cand.pred_time_us < best_score) {
            best_score = cand.pred_time_us;
        }
    }

    std::vector<Candidate> best_candidates;
    for (const auto& cand : all_candidates) {
        if (cand.pred_time_us <= best_score * (1.0 + kTieEpsilon)) {
            best_candidates.push_back(cand);
        }
    }

    uint32_t idx = SimpleRandom::Get().next(best_candidates.size());
    chosen_dev_id = best_candidates[idx].dev_id;

    devices_[chosen_dev_id].addInflight(length);
    devices_[chosen_dev_id].total_bytes.fetch_add(length, std::memory_order_relaxed);

    // Track cross-NUMA usage
    bool is_cross_numa = best_candidates[idx].is_cross_numa;
    if (is_cross_numa) {
        total_cross_numa_bytes_.fetch_add(length, std::memory_order_relaxed);
    } else {
        total_local_bytes_.fetch_add(length, std::memory_order_relaxed);
    }

    return Status::OK();
}

Status DeviceQuota::allocateBatch(uint64_t total_length, uint32_t num_slices,
                                  const std::string &location,
                                  std::vector<int>& slice_dev_ids,
                                  int priority,
                                  uint64_t device_mask) {
    slice_dev_ids.clear();
    slice_dev_ids.reserve(num_slices);

    auto entry = local_topology_->getMemEntry(location);
    if (!entry) return Status::InvalidArgument("Unknown location" LOC_MARK);

    uint64_t slice_bytes = (total_length + num_slices - 1) / num_slices;

    // Collect all eligible devices from rank 0 (local NUMA only)
    struct DeviceScore {
        int dev_id;
        double bandwidth_bps;
    };
    std::vector<DeviceScore> devices_scores;

    for (int dev_id : entry->device_list[0]) {
        if (!devices_.count(dev_id)) continue;
        if ((device_mask & (1ULL << dev_id)) == 0) continue;

        auto &dev = devices_[dev_id];
        double bw_bps = dev.getEwmaBandwidth();
        devices_scores.push_back({dev_id, bw_bps});
    }

    if (devices_scores.empty())
        return Status::DeviceNotFound("no eligible devices");

    // When quota is disabled, use simple round-robin
    if (!enable_quota_) {
        for (uint32_t i = 0; i < num_slices; ++i) {
            int dev_id = devices_scores[tl_allocate_count % devices_scores.size()].dev_id;
            slice_dev_ids.push_back(dev_id);
            tl_allocate_count++;
            devices_[dev_id].total_bytes.fetch_add(slice_bytes, std::memory_order_relaxed);
        }
        return Status::OK();
    }

    // ========== Weighted round-robin based on bandwidth ==========
    // Calculate cumulative scores
    std::vector<double> cumulative_scores;
    double total_score = 0;
    for (const auto& ds : devices_scores) {
        total_score += ds.bandwidth_bps;
        cumulative_scores.push_back(total_score);
    }

    // Use a per-thread counter to ensure different threads distribute differently
    thread_local uint64_t tl_batch_count = 0;
    double offset = (tl_batch_count++ * slice_bytes) / total_score;

    for (uint32_t i = 0; i < num_slices; ++i) {
        double pos = fmod(offset + i * (total_score / num_slices), total_score);

        // Find device for this position
        int chosen_dev_id = devices_scores[0].dev_id;
        for (size_t j = 0; j < cumulative_scores.size(); ++j) {
            if (pos < cumulative_scores[j]) {
                chosen_dev_id = devices_scores[j].dev_id;
                break;
            }
        }

        slice_dev_ids.push_back(chosen_dev_id);

        // Update inflight for this slice
        devices_[chosen_dev_id].addInflight(slice_bytes);
        devices_[chosen_dev_id].total_bytes.fetch_add(slice_bytes, std::memory_order_relaxed);
        total_local_bytes_.fetch_add(slice_bytes, std::memory_order_relaxed);
    }

    return Status::OK();
}

// Print per-second traffic statistics
static void printTrafficStats(std::unordered_map<int, DeviceQuota::DeviceInfo>& devices) {
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
        if (last_print_time_ns.compare_exchange_strong(expected_last, now,
                std::memory_order_relaxed, std::memory_order_relaxed)) {
            std::ostringstream oss;
            oss << "[RDMA Traffic] ";
            bool first = true;
            for (auto& [dev_id, dev] : devices) {
                uint64_t total = dev.total_bytes.load(std::memory_order_relaxed);
                uint64_t last = dev.last_second_bytes.load(std::memory_order_relaxed);
                uint64_t delta = total - last;
                double mbps = delta * 8.0 / 1e6;  // Mb/s

                dev.last_second_bytes.store(total, std::memory_order_relaxed);

                // Skip devices with zero traffic
                if (mbps < 0.01) continue;

                if (!first) oss << " | ";
                oss << "dev" << dev_id << ": " << std::fixed << std::setprecision(2)
                    << mbps << " Mb/s";
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

    auto &dev = it->second;

    // Release inflight bytes
    dev.subInflight(length);

    // Early exit if EWMA learning disabled
    if (!sched_params_.enable_ewma_learning)
        return Status::OK();

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