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
constexpr uint64_t kLocationCacheTtlNs = 10 * 1000000000ULL;  // 10 seconds

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
    Status status =
        buildCandidates(entry, slice_bytes, device_mask, tl_candidates);
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
                                    std::vector<Candidate>& candidates) {
    // Rank priority weights: 9:3:1 for rank 0:1:2
    static constexpr double kRankWeight[Topology::DevicePriorityRanks] = {
        9.0, 3.0, 1.0};
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
            double score =
                (effective_bw * kRankWeight[rank]) / (inflight + slice_bytes);
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