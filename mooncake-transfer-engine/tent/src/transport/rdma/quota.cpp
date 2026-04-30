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

// Enable bandwidth learning debug logging
// #define BANDWIDTH_LEARNING_DEBUG 1

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
// Fixed-size buffers to avoid dynamic allocation
thread_local std::vector<int> tl_eligible;
thread_local std::vector<DeviceQuota::Candidate> tl_candidates;

Status DeviceQuota::allocate(uint64_t total_length, uint32_t num_slices,
                             const std::string& location,
                             std::vector<int>& slice_dev_ids, int priority,
                             uint64_t device_mask) {
    slice_dev_ids.clear();
    slice_dev_ids.reserve(num_slices);  // Pre-allocate to avoid reallocation

    // Get location entry (getMemEntry is already cached internally)
    const Topology::MemEntry* entry = local_topology_->getMemEntry(location);
    if (!entry) return Status::InvalidArgument("Unknown location");

    // Calculate block_size using same logic as rdma_transport.cpp
    const double merge_ratio = 0.25;
    uint64_t base_block = default_block_size_;
    uint64_t calc_num_slices = (total_length + base_block - 1) / base_block;
    calc_num_slices = std::max<uint64_t>(
        1, std::min<uint64_t>(calc_num_slices, max_slice_count_));

    if (calc_num_slices > 1) {
        uint64_t tail = total_length % base_block;
        if (tail > 0 &&
            tail < static_cast<uint64_t>(base_block * merge_ratio)) {
            calc_num_slices = std::max<uint64_t>(1, calc_num_slices - 1);
        }
    }

    // Align to default_block_size
    auto roundup = [](uint64_t a, uint64_t b) -> uint64_t {
        return (a % b == 0) ? a : (a / b + 1) * b;
    };
    uint64_t block_size =
        roundup((total_length + calc_num_slices - 1) / calc_num_slices,
                default_block_size_);

    calc_num_slices = std::max<uint64_t>(
        1, std::min<uint64_t>(calc_num_slices, max_slice_count_));

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
    Status status = buildCandidates(entry, slice_bytes, device_mask,
                                    tl_candidates, priority);
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
        selectMultiPath(tl_candidates, num_slices, total_length, slice_dev_ids,
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

int DeviceQuota::getDevicePriority(int dev_id) const {
    // Rotation enabled: compute priority based on current epoch
    uint64_t now = getCurrentTimeInNano();
    uint64_t epoch = now / sched_params_.epoch_duration_ns;

    // Get base priority for this device
    int base_priority = 0;  // Default HIGH
    if (dev_id < (int)sched_params_.device_base_priorities.size()) {
        base_priority = sched_params_.device_base_priorities[dev_id];
    }

    // Rotate: shift priority by epoch number
    // epoch 0: [0,1,2,0] -> [0,1,2,0]
    // epoch 1: [0,1,2,0] -> [1,2,0,1]
    // epoch 2: [0,1,2,0] -> [2,0,1,2]
    int num_priority_levels = 3;  // HIGH, MEDIUM, LOW
    int dev_priority = (base_priority + epoch) % num_priority_levels;

    return dev_priority;
}

void DeviceQuota::fillDevicePriorities() {
    if (!sched_params_.enable_device_priority) return;

    size_t num_devices = devices_.size();
    sched_params_.device_base_priorities.clear();
    sched_params_.device_base_priorities.reserve(num_devices);

    // Distribute priorities evenly: 0,1,2,0,1,2,...
    int num_priority_levels = 3;  // HIGH, MEDIUM, LOW
    for (size_t i = 0; i < num_devices; ++i) {
        sched_params_.device_base_priorities.push_back(i % num_priority_levels);
    }

    // Log the auto-filled priorities
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < sched_params_.device_base_priorities.size(); ++i) {
        if (i > 0) oss << ",";
        oss << sched_params_.device_base_priorities[i];
    }
    oss << "]";
    LOG(INFO) << "Auto-filled device base priorities: " << oss.str();
}

Status DeviceQuota::buildCandidates(const Topology::MemEntry* entry,
                                    uint64_t slice_bytes, uint64_t device_mask,
                                    std::vector<Candidate>& candidates,
                                    int request_priority) {
    candidates.clear();
    for (size_t rank = 0; rank < Topology::DevicePriorityRanks; ++rank) {
        for (int dev_id : entry->device_list[rank]) {
            if (!devices_.count(dev_id) || !(device_mask & (1ULL << dev_id)))
                continue;
            const auto& dev = devices_[dev_id];
            bool is_cross_numa = (rank == 2);
            // Cross-NUMA devices: only use if their local node is idle
            if (is_cross_numa && !isCrossNumaNodeIdle(dev.numa_id)) continue;

            // Get device's current priority slot (for priority penetration)
            // Priority values: 0=HIGH, 1=MEDIUM, 2=LOW
            // Device accepts request if dev_priority >= request_priority
            int dev_priority;
            if (shared_quota_) {
                // Multi-process: use shared quota per-device slot
                // All processes see the same device slot (coordinated)
                dev_priority = shared_quota_->getDevicePriority(dev_id);
            } else {
                // Single-process: use local device priority rotation
                dev_priority = getDevicePriority(dev_id);
                // If rotation disabled, default to LOW (accept all)
                if (!sched_params_.enable_device_priority) {
                    dev_priority = PRIO_LOW;
                }
            }

            // Filter: device must accept this request priority (priority penetration)
            // dev_priority=0(HIGH): only request_priority=0(HIGH)
            // dev_priority=1(MEDIUM): request_priority=0,1
            // dev_priority=2(LOW): request_priority=0,1,2
            if (dev_priority < request_priority) continue;

            double effective_bw = dev.getEwmaBandwidth();
            uint64_t inflight = dev.getInflightBytes();
            // Score = (effective_bandwidth * rank_weight) / pending_bytes
            double raw_score =
                (effective_bw * sched_params_.rank_weights[rank]) /
                (inflight + slice_bytes);
            // Set minimum score to 0.01 to avoid zero scores
            double score = std::max(raw_score, 0.01);

            candidates.push_back({dev_id, score, is_cross_numa, dev_priority});
        }
    }

    if (candidates.empty())
        return Status::DeviceNotFound("no eligible devices");
    return Status::OK();
}

void DeviceQuota::selectSinglePath(const std::vector<Candidate>& candidates,
                                   uint32_t num_slices, uint64_t total_length,
                                   std::vector<int>& slice_dev_ids) {
    // Select best devices with 10% tolerance for load balancing
    double max_score = 0;
    for (const auto& c : candidates) max_score = std::max(max_score, c.score);
    std::vector<size_t> best_indices;
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (candidates[i].score >= max_score * 0.9) best_indices.push_back(i);
    }

    // Calculate block_size using shared helper (must match rdma_transport.cpp)
    uint64_t block_size =
        calculateBlockSize(total_length, num_slices, default_block_size_);

    // Distribute slices with actual lengths
    uint64_t offset = 0;
    for (uint32_t i = 0; i < num_slices; ++i) {
        size_t idx = best_indices[tl_rr_counter++ % best_indices.size()];
        slice_dev_ids.push_back(candidates[idx].dev_id);
        uint64_t actual_slice_bytes =
            getSliceLength(total_length, offset, block_size);
        offset += actual_slice_bytes;
        updateStats(candidates[idx].dev_id, actual_slice_bytes);
    }
}

void DeviceQuota::selectMultiPath(const std::vector<Candidate>& candidates,
                                  uint32_t num_slices, uint64_t total_length,
                                  std::vector<int>& slice_dev_ids,
                                  bool explore_mode) {
    // Calculate block_size using shared helper (must match rdma_transport.cpp)
    uint64_t block_size =
        calculateBlockSize(total_length, num_slices, default_block_size_);

    // Distribute slices with actual lengths
    uint64_t offset = 0;

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
            uint64_t actual_slice_bytes =
                getSliceLength(total_length, offset, block_size);
            offset += actual_slice_bytes;
            updateStats(candidates[cand_idx].dev_id, actual_slice_bytes);
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
        uint64_t slice_bytes = (total_length + num_slices - 1) / num_slices;
        double weight_offset = (tl_batch_offset++ * slice_bytes) / total_score;
        for (uint32_t i = 0; i < num_slices; ++i) {
            double pos = fmod(weight_offset + i * (total_score / num_slices),
                              total_score);
            auto it = std::lower_bound(cumulative_scores.begin(),
                                       cumulative_scores.end(), pos);
            size_t idx = std::distance(cumulative_scores.begin(), it);
            if (idx >= candidates.size()) idx = candidates.size() - 1;

            const auto& selected = candidates[idx];
            slice_dev_ids.push_back(selected.dev_id);
            uint64_t actual_slice_bytes =
                getSliceLength(total_length, offset, block_size);
            offset += actual_slice_bytes;
            updateStats(selected.dev_id, actual_slice_bytes);
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
            // First, update EWMA bandwidth for all devices
            if (sched_params_.enable_ewma_learning) {
                for (auto& [dev_id, dev] : devices_) {
                    uint64_t total =
                        dev.total_bytes.load(std::memory_order_relaxed);
                    uint64_t last =
                        dev.last_second_bytes.load(std::memory_order_relaxed);
                    uint64_t delta_bytes = total - last;
                    double actual_bw_bps = static_cast<double>(delta_bytes);

                    double max_bw_ratio = dev.getMaxBwRatio();
                    double l_min = dev.getMinLatency();
                    uint64_t sample_count =
                        dev.sample_count_.load(std::memory_order_relaxed);

                    if (sample_count > 0 && max_bw_ratio > 0) {
                        // Skip update if insufficient samples
                        constexpr uint64_t kMinSampleThreshold = 20000;
                        if (sample_count < kMinSampleThreshold) {
                            // Use initial theoretical value
                            double theoretical_bw =
                                dev.getTheoreticalBandwidth();
                            dev.ewma_bandwidth_bps.store(
                                theoretical_bw, std::memory_order_relaxed);
                        } else {
                            // bw_probe_bps is already max of L_i / (tau_i -
                            // beta)
                            double bw_probe_bps = max_bw_ratio;

                            // Give bw_probe parallel correction factor
                            double raw_norm_bps = std::max(
                                actual_bw_bps,
                                bw_probe_bps * sched_params_.bw_probe_factor);

                            double theoretical_bw =
                                dev.getTheoreticalBandwidth();
                            double min_bw = theoretical_bw * 0.75;
                            double max_bw = theoretical_bw * 1.0;
                            raw_norm_bps =
                                std::clamp(raw_norm_bps, min_bw, max_bw);

                            double alpha = sched_params_.ewma_alpha;
                            double current_ewma = dev.getEwmaBandwidth();
                            double new_ewma = alpha * current_ewma +
                                              (1.0 - alpha) * raw_norm_bps;
                            dev.ewma_bandwidth_bps.store(
                                new_ewma, std::memory_order_relaxed);

#ifdef BANDWIDTH_LEARNING_DEBUG
                            LOG(INFO)
                                << "[Bandwidth Normalization] dev" << dev_id
                                << " actual_bw: " << (actual_bw_bps / 1e9)
                                << " GB/s"
                                << ", l_min: " << (l_min * 1e6) << " us"
                                << ", max_bw_ratio: " << (max_bw_ratio / 1e9)
                                << " GB/s"
                                << ", bw_probe: " << (bw_probe_bps / 1e9)
                                << " GB/s"
                                << ", raw_norm: " << (raw_norm_bps / 1e9)
                                << " GB/s"
                                << ", ewma: " << (current_ewma / 1e9) << " -> "
                                << (new_ewma / 1e9) << " GB/s"
                                << ", samples: " << sample_count;
#endif
                        }
                    }

                    // Reset tracker for next period
                    dev.resetPeriodTracker();
                }
            }

            // Then, print traffic stats
            std::ostringstream oss;
            oss << "[RDMA Traffic] ";
            bool first = true;
            for (auto& [dev_id, dev] : devices_) {
                uint64_t total =
                    dev.total_bytes.load(std::memory_order_relaxed);
                uint64_t last =
                    dev.last_second_bytes.load(std::memory_order_relaxed);
                uint64_t delta = total - last;
                double throughput_gb_s = delta / 1e9;
                double ewma_gb_s = dev.getEwmaBandwidth() / 1e9;

                dev.last_second_bytes.store(total, std::memory_order_relaxed);

                if (throughput_gb_s < 0.01) continue;

                if (!first) oss << " | ";
                oss << "dev" << dev_id << ": " << std::fixed
                    << std::setprecision(2) << throughput_gb_s
                    << " GB/s (ewma: " << std::setprecision(2) << ewma_gb_s
                    << " GB/s)";
                first = false;
            }

            if (!first) {
                LOG(INFO) << oss.str();
            }
        }
    }
}

Status DeviceQuota::release(int dev_id, uint64_t length, double latency) {
    printTrafficStats();
    if (!enable_quota_) return Status::OK();
    auto it = devices_.find(dev_id);
    if (it == devices_.end())
        return Status::InvalidArgument("device not found");

    auto& dev = it->second;
    dev.subInflight(length);

    // Add sample for bandwidth estimation (updated in printTrafficStats)
    if (sched_params_.enable_ewma_learning) {
        dev.addSample(length, latency);
    }

    return Status::OK();
}

}  // namespace tent
}  // namespace mooncake