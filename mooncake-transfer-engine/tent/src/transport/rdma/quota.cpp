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

Status DeviceQuota::loadTopology(std::shared_ptr<Topology>& local_topology) {
    local_topology_ = local_topology;
    for (size_t dev_id = 0; dev_id < local_topology->getNicCount(); ++dev_id) {
        auto entry = local_topology->getNicEntry(dev_id);
        if (!entry || entry->type != Topology::NIC_RDMA) continue;
        DeviceInfo& info = devices_[dev_id];
        info.dev_id = dev_id;
        info.bw_gbps = entry->bw_gbps;
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

// Thread-local cache for location entry lookup
struct QuotaLocationCache {
    std::string location;
    const Topology::MemEntry* entry;
    uint64_t last_update_ns;
};
thread_local QuotaLocationCache tl_quota_location_cache{"", nullptr, 0};

// Fixed-size buffers to avoid dynamic allocation
thread_local std::vector<int> tl_eligible;
thread_local std::vector<DeviceQuota::Candidate> tl_candidates;
thread_local DeviceQuota::LocationWeightsCache tl_weights_cache{"", nullptr, 0};
constexpr uint64_t kLocationCacheTtlNs = 10 * 1000000000ULL;  // 10 seconds

// Constants for adaptive weight adjustment
constexpr uint32_t kWeightAdjustInterval = 10000;  // Adjust every N samples
constexpr double kWeightLearningRate = 0.01;  // Very conservative (was 0.1)
constexpr double kMinWeight = 0.1;           // Minimum weight
constexpr double kMaxWeight = 15.0;          // Maximum weight
constexpr uint64_t kMinDataBytes = 200 * 1024 * 1024ULL;  // Min 200MB for stable stats
constexpr double kMaxAdjustRatio = 0.2;      // Max 20% change per adjustment

// NUMA-based minimum weight ratios to preserve affinity hierarchy
// rank0 should always be significantly preferred due to same-NUMA latency
constexpr double kMinRank0ToRank1Ratio = 2.0;  // rank0 >= 2x rank1
constexpr double kMinRank1ToRank2Ratio = 2.0;  // rank1 >= 2x rank2

Status DeviceQuota::allocate(uint64_t total_length, uint32_t num_slices,
                             const std::string& location,
                             std::vector<int>& slice_dev_ids, int priority,
                             uint64_t device_mask) {
    slice_dev_ids.clear();
    slice_dev_ids.reserve(num_slices);  // Pre-allocate to avoid reallocation

    // Fast path: cached location entry lookup
    const Topology::MemEntry* entry = nullptr;
    uint64_t now = getCurrentTimeInNano();
    if (tl_quota_location_cache.entry &&
        tl_quota_location_cache.location == location &&
        (now - tl_quota_location_cache.last_update_ns) < kLocationCacheTtlNs) {
        entry = tl_quota_location_cache.entry;
    } else {
        entry = local_topology_->getMemEntry(location);
        if (!entry) return Status::InvalidArgument("Unknown location");
        tl_quota_location_cache.location = location;
        tl_quota_location_cache.entry = entry;
        tl_quota_location_cache.last_update_ns = now;
    }

    uint64_t slice_bytes = (total_length + num_slices - 1) / num_slices;

    // ========== Baseline mode: simple round-robin ==========
    if (!enable_quota_) {
        // Reuse thread-local buffer
        for (size_t rank = 0; rank < Topology::DevicePriorityRanks; ++rank) {
            tl_eligible.clear();
            for (int dev_id : entry->device_list[rank]) {
                if (!devices_.count(dev_id)) continue;
                if ((device_mask & (1ULL << dev_id)) == 0) continue;
                tl_eligible.push_back(dev_id);
            }
            if (tl_eligible.empty()) break;
            for (uint32_t i = 0; i < num_slices; ++i) {
                int dev_id = tl_eligible[tl_rr_counter % tl_eligible.size()];
                tl_rr_counter++;
                slice_dev_ids.push_back(dev_id);
                devices_[dev_id].total_bytes.fetch_add(
                    slice_bytes, std::memory_order_relaxed);
            }
            return Status::OK();
        }
        return Status::DeviceNotFound("no eligible devices");
    }

    // ========== Smart mode: EWMA-based selection ==========
    // Reuse thread-local buffer
    tl_candidates.clear();
    Status status = buildCandidates(entry, slice_bytes, device_mask, location,
                                    tl_candidates);
    if (!status.ok()) return status;

    // Exploration: 1% of requests use multi-path to discover true device
    // performance
    thread_local uint64_t tl_call_count = 0;
    bool explore_mode =
        ((++tl_call_count % 100) == 0) && (tl_candidates.size() > 1);

    bool use_multi_path = explore_mode ||  // Force multi-path for exploration
                          (num_slices >= sched_params_.batch_threshold &&
                           tl_candidates.size() > 1);

    if (!use_multi_path) {
        selectSinglePath(tl_candidates, num_slices, total_length,
                         slice_dev_ids);
    } else {
        selectMultiPath(tl_candidates, num_slices, slice_bytes, slice_dev_ids,
                        explore_mode);
    }
    return Status::OK();
}

// ============================================================================
// Helper: get device rank for a location
// ============================================================================

int DeviceQuota::getDeviceRank(const std::string& location, int dev_id) const {
    auto entry = local_topology_->getMemEntry(location);
    if (!entry) return 0;  // Default to rank0

    for (size_t rank = 0; rank < Topology::DevicePriorityRanks; ++rank) {
        for (int id : entry->device_list[rank]) {
            if (id == dev_id) return static_cast<int>(rank);
        }
    }
    return 0;  // Default to rank0 if not found
}

// ============================================================================
// Helper functions
// ============================================================================

bool DeviceQuota::isCrossNumaNodeIdle(int dev_numa) const {
    for (const auto& pair : devices_) {
        if (pair.second.numa_id == dev_numa &&
            pair.second.getInflightBytes() > 0) {
            return false;
        }
    }
    return true;
}

Status DeviceQuota::buildCandidates(const Topology::MemEntry* entry,
                                    uint64_t slice_bytes, uint64_t device_mask,
                                    const std::string& location,
                                    std::vector<Candidate>& candidates) {
    // Select rank weights: static or adaptive
    const double* rank_weight;
    if (sched_params_.enable_adaptive_rank_weights) {
        // Use per-location adaptive weights (with thread-local caching)
        LocationRankWeights* loc_weights = getLocationWeights(location);
        rank_weight = loc_weights->weight;
    } else {
        // Use static default weights (9:3:0.3 for rank 0:1:2)
        rank_weight = kStaticRankWeight;
    }

    candidates.clear();
    for (size_t rank = 0; rank < Topology::DevicePriorityRanks; ++rank) {
        for (int dev_id : entry->device_list[rank]) {
            if (!devices_.count(dev_id) || !(device_mask & (1ULL << dev_id)))
                continue;
            const auto& dev = devices_[dev_id];
            bool is_cross_numa = (rank == 2);
            // Cross-NUMA devices: only use if their local node is idle
            if (is_cross_numa && !isCrossNumaNodeIdle(dev.numa_id)) continue;
            double effective_bw = dev.getEwmaBandwidth();
            uint64_t inflight = dev.getInflightBytes();
            // Score = (effective_bandwidth * rank_weight) / pending_bytes
            double score =
                (effective_bw * rank_weight[rank]) / (inflight + slice_bytes);
            candidates.push_back({dev_id, score, is_cross_numa});
        }
    }
    if (candidates.empty())
        return Status::DeviceNotFound("no eligible devices");
    return Status::OK();
}

void DeviceQuota::selectSinglePath(const std::vector<Candidate>& candidates,
                                   uint32_t num_slices, uint64_t total_length,
                                   std::vector<int>& slice_dev_ids) {
    // Select best device with 10% tolerance for load balancing
    double max_score = 0;
    for (const auto& c : candidates) max_score = std::max(max_score, c.score);
    std::vector<size_t> best_indices;
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (candidates[i].score >= max_score * 0.9) best_indices.push_back(i);
    }
    size_t idx = best_indices[SimpleRandom::Get().next(best_indices.size())];
    const auto& target = candidates[idx];
    for (uint32_t i = 0; i < num_slices; ++i)
        slice_dev_ids.push_back(target.dev_id);
    updateStats(target.dev_id, total_length);
}

void DeviceQuota::selectMultiPath(const std::vector<Candidate>& candidates,
                                  uint32_t num_slices, uint64_t slice_bytes,
                                  std::vector<int>& slice_dev_ids,
                                  bool explore_mode) {
    if (explore_mode) {
        // Exploration mode: randomly distribute slices across all devices
        // Shuffle candidate indices for random exploration
        std::vector<size_t> shuffled_idx(candidates.size());
        for (size_t i = 0; i < candidates.size(); ++i) shuffled_idx[i] = i;
        for (size_t i = shuffled_idx.size() - 1; i > 0; --i) {
            size_t j = SimpleRandom::Get().next(i + 1);
            std::swap(shuffled_idx[i], shuffled_idx[j]);
        }

        // Distribute slices across shuffled devices (round-robin)
        for (uint32_t i = 0; i < num_slices; ++i) {
            size_t cand_idx = shuffled_idx[i % candidates.size()];
            slice_dev_ids.push_back(candidates[cand_idx].dev_id);
            updateStats(candidates[cand_idx].dev_id, slice_bytes);
        }
    } else {
        // Normal mode: bandwidth-weighted distribution
        std::vector<double> cumulative_scores;
        double total_score = 0;
        for (const auto& c : candidates) {
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

            const auto& selected = candidates[idx];
            slice_dev_ids.push_back(selected.dev_id);
            updateStats(selected.dev_id, slice_bytes);
        }
    }
}

void DeviceQuota::updateStats(int dev_id, uint64_t bytes) {
    auto& dev = devices_[dev_id];
    dev.addInflight(bytes);
    dev.total_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

void DeviceQuota::printTrafficStats() {
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
            for (auto& [dev_id, dev] : devices_) {
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

            // Print adaptive rank weights if enabled and non-empty
            if (sched_params_.enable_adaptive_rank_weights &&
                !location_weights_.empty()) {
                std::ostringstream woss;
                woss << "[Rank Weights] ";
                bool wfirst = true;
                for (const auto& [loc, weights] : location_weights_) {
                    if (!wfirst) woss << " | ";
                    woss << loc << ": [" << std::fixed << std::setprecision(2)
                         << weights.weight[0] << ", " << weights.weight[1]
                         << ", " << weights.weight[2] << "]";
                    wfirst = false;
                }
                LOG(INFO) << woss.str();
            }
        }
    }
}

Status DeviceQuota::release(int dev_id, uint64_t length, double latency,
                            int priority, const std::string& location,
                            int rank) {
    // Print traffic stats periodically (including weights)
    printTrafficStats();

    if (!enable_quota_) return Status::OK();

    auto it = devices_.find(dev_id);
    if (it == devices_.end())
        return Status::InvalidArgument("device not found");

    auto& dev = it->second;

    // ========== MUST EXECUTE: Release inflight bytes ==========
    dev.subInflight(length);

    bool learning_enabled = sched_params_.enable_ewma_learning ||
                            sched_params_.enable_adaptive_rank_weights;
    if (learning_enabled) {
        thread_local uint64_t tl_sample_counter = 0;
        double sample_rate = sched_params_.param_update_sample_rate;
        bool should_update =
            (sample_rate >= 1.0) ||
            ((++tl_sample_counter % static_cast<uint64_t>(1.0 / sample_rate)) == 0);

        if (should_update) {
            if (length >= 4096 && latency >= 1e-6) {
                if (sched_params_.enable_ewma_learning) {
                    double observed_bps = length / latency;
                    double alpha = sched_params_.ewma_alpha;
                    double current_ewma = dev.getEwmaBandwidth();
                    double new_ewma =
                        alpha * current_ewma + (1.0 - alpha) * observed_bps;

                    // Clamp to reasonable bounds (10% to 300% of theoretical)
                    double min_bw = dev.getTheoreticalBandwidth() * 0.1;
                    double max_bw = dev.getTheoreticalBandwidth() * 3.0;
                    new_ewma = std::clamp(new_ewma, min_bw, max_bw);

                    dev.ewma_bandwidth_bps.store(new_ewma,
                                                 std::memory_order_relaxed);
                }

                // --- Adaptive rank weight update ---
                if (sched_params_.enable_adaptive_rank_weights &&
                    !location.empty() && rank >= 0 &&
                    rank < (int)Topology::DevicePriorityRanks) {
                    LocationRankWeights* loc_weights =
                        getLocationWeights(location);

                    // Update statistics for this rank
                    auto& stats = loc_weights->stats[rank];
                    __sync_fetch_and_add(&stats.total_bytes, length);
                    __sync_fetch_and_add(&stats.total_latency_ns,
                                         static_cast<uint64_t>(latency * 1e9));
                    uint32_t count =
                        __sync_fetch_and_add(&stats.sample_count, 1) + 1;

                    // Periodically adjust weights (every N samples per rank)
                    if (count % kWeightAdjustInterval == 0) {
                        adjustLocationWeights(loc_weights);
                    }
                }
            }
        }
    }

    // ========== MUST EXECUTE: Periodic diffusion for shared quota ==========
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

// ============================================================================
// Adaptive weight management
// ============================================================================

DeviceQuota::LocationRankWeights* DeviceQuota::getLocationWeights(
    const std::string& location) {
    // Fast path: thread-local cache hit
    uint64_t now = getCurrentTimeInNano();
    if (tl_weights_cache.weights && tl_weights_cache.location == location &&
        (now - tl_weights_cache.last_update_ns) < kLocationCacheTtlNs) {
        return static_cast<LocationRankWeights*>(tl_weights_cache.weights);
    }

    // Slow path: lookup or create
    {
        std::shared_lock<std::shared_mutex> read_lock(location_weights_lock_);
        auto it = location_weights_.find(location);
        if (it != location_weights_.end()) {
            tl_weights_cache.location = location;
            tl_weights_cache.weights = &it->second;
            tl_weights_cache.last_update_ns = now;
            return &it->second;
        }
    }

    // Create new entry (write lock)
    {
        std::unique_lock<std::shared_mutex> write_lock(location_weights_lock_);
        // Double-check after acquiring write lock
        auto it = location_weights_.find(location);
        if (it == location_weights_.end()) {
            auto result =
                location_weights_.emplace(location, LocationRankWeights());
            tl_weights_cache.location = location;
            tl_weights_cache.weights = &result.first->second;
            tl_weights_cache.last_update_ns = now;
            return &result.first->second;
        }
        tl_weights_cache.location = location;
        tl_weights_cache.weights = &it->second;
        tl_weights_cache.last_update_ns = now;
        return &it->second;
    }
}

void DeviceQuota::adjustLocationWeights(LocationRankWeights* loc_weights) {
    // Calculate normalized latency for each rank (ns per GB)
    double norm_latency[Topology::DevicePriorityRanks] = {0};
    uint64_t total_bytes_arr[Topology::DevicePriorityRanks] = {0};
    bool has_valid_data[Topology::DevicePriorityRanks] = {false};

    for (size_t r = 0; r < Topology::DevicePriorityRanks; ++r) {
        // Read stats atomically
        total_bytes_arr[r] =
            __sync_fetch_and_add(&loc_weights->stats[r].total_bytes, 0);
        uint64_t total_ns =
            __sync_fetch_and_add(&loc_weights->stats[r].total_latency_ns, 0);

        // Require sufficient data for stable statistics
        if (total_bytes_arr[r] >= kMinDataBytes) {
            // Normalize: latency per GB (to make values comparable)
            double gb =
                static_cast<double>(total_bytes_arr[r]) / (1024.0 * 1024.0 * 1024.0);
            norm_latency[r] = static_cast<double>(total_ns) / gb;
            has_valid_data[r] = true;
        }
    }

    // Require rank0 to have valid data (it's the baseline)
    // Allow partial adjustment if rank1/2 have insufficient data
    if (!has_valid_data[0]) {
        return;  // No baseline, skip adjustment
    }

    // Key insight from NUMA perspective:
    // rank0 (same-NUMA) should ALWAYS be preferred significantly
    // Even if rank1/2 appear faster due to being temporarily idle,
    // their cross-NUMA penalty will manifest under load.
    // Therefore, we only adjust weights to reflect LONG-TERM trends,
    // not transient idle-state performance.

    double w0_current = loc_weights->weight[0];
    double w1_current = loc_weights->weight[1];
    double w2_current = loc_weights->weight[2];

    // Calculate latency ratios relative to rank0
    // ratio > 1.0 means this rank is slower than rank0 (expected for NUMA)
    // ratio < 1.0 would be suspicious (rank1/2 faster than rank0?)
    double latency_ratio_1 = has_valid_data[1] ?
        (norm_latency[1] / norm_latency[0]) : 3.0;  // Default: 3x slower
    double latency_ratio_2 = has_valid_data[2] ?
        (norm_latency[2] / norm_latency[0]) : 10.0;  // Default: 10x slower

    // Sanity check: if rank1/2 appear faster than rank0, it's likely due to
    // them being idle. Don't reward this - maintain NUMA hierarchy.
    if (latency_ratio_1 < 1.2) latency_ratio_1 = 1.2;  // At least 1.2x slower
    if (latency_ratio_2 < 2.0) latency_ratio_2 = 2.0;  // At least 2x slower

    // Calculate target weights based on observed latency penalties
    // Higher latency penalty -> lower target weight
    double w0_target = w0_current;  // Keep rank0 stable as anchor
    double w1_target = w0_target / latency_ratio_1;
    double w2_target = w0_target / latency_ratio_2;

    // Apply EWMA update with conservative rate
    double w0_new = w0_current;  // Keep rank0 unchanged (it's the anchor)
    double w1_new = kWeightLearningRate * w1_target +
                    (1.0 - kWeightLearningRate) * w1_current;
    double w2_new = kWeightLearningRate * w2_target +
                    (1.0 - kWeightLearningRate) * w2_current;

    // Limit adjustment magnitude (prevent large swings)
    w1_new = std::clamp(w1_new, w1_current * (1.0 - kMaxAdjustRatio),
                        w1_current * (1.0 + kMaxAdjustRatio));
    w2_new = std::clamp(w2_new, w2_current * (1.0 - kMaxAdjustRatio),
                        w2_current * (1.0 + kMaxAdjustRatio));

    // Enforce NUMA hierarchy: rank0 >> rank1 >> rank2
    // This is critical - cross-NUMA should never be preferred
    if (w1_new > w0_new / kMinRank0ToRank1Ratio) {
        w1_new = w0_new / kMinRank0ToRank1Ratio;
    }
    if (w2_new > w1_new / kMinRank1ToRank2Ratio) {
        w2_new = w1_new / kMinRank1ToRank2Ratio;
    }

    // Final bounds check
    loc_weights->weight[0] = std::clamp(w0_new, kMinWeight, kMaxWeight);
    loc_weights->weight[1] = std::clamp(w1_new, kMinWeight, kMaxWeight);
    loc_weights->weight[2] = std::clamp(w2_new, kMinWeight, kMaxWeight);

    // Reset statistics after successful adjustment
    for (size_t r = 0; r < Topology::DevicePriorityRanks; ++r) {
        loc_weights->stats[r].total_bytes = 0;
        loc_weights->stats[r].total_latency_ns = 0;
    }
}

}  // namespace tent
}  // namespace mooncake