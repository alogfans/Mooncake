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

#ifndef TENT_QUOTA_H
#define TENT_QUOTA_H

#include <atomic>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <shared_mutex>
#include <mutex>

#include "tent/common/status.h"
#include "tent/runtime/topology.h"
#include "tent/common/types.h"  // for PRIO_* constants

namespace mooncake {
namespace tent {

// Bandwidth constants (Gbps)
static constexpr double kDefaultBwGbps = 400.0;
static constexpr double kMinBwGbps = 10.0;
static constexpr double kMaxBwGbps = 800.0;

class SharedQuotaManager;

/**
 * @brief DeviceQuota implements NIC selection with two modes:
 *
 * 1. Baseline mode (enable_quota=false): Simple round-robin
 *    - Deterministic, no load tracking
 *    - All devices used equally
 *
 * 2. Smart mode (enable_quota=true): EWMA-based selection
 *    - Tracks global inflight bytes per device
 *    - Learns effective bandwidth via EWMA
 *    - Selects device with minimal predicted completion time
 *    - Supports multi-path for large requests
 *
 * Selection formula:
 *     predicted_time = (inflight + length) / ewma_bandwidth
 *
 * EWMA update:
 *     ewma_bandwidth <- alpha * ewma_bandwidth + (1 - alpha) *
 * observed_bandwidth
 */
class DeviceQuota {
   public:
    // Candidate device for allocation (used internally and for thread-local
    // cache)
    struct Candidate {
        int dev_id;
        double score;
        bool is_cross_numa;
        int dev_priority;
    };

    struct DeviceInfo {
        int dev_id;
        double bw_gbps;  // Theoretical bandwidth from topology
        int numa_id;
        uint64_t padding0[5];
        std::atomic<uint64_t> inflight_bytes{0};  // Global inflight bytes
        uint64_t padding1[7];
        std::atomic<double> ewma_bandwidth_bps{
            25e9};  // Learned effective bandwidth
        uint64_t padding2[7];
        std::atomic<uint64_t> total_bytes{0};        // Total bytes transferred
        std::atomic<uint64_t> last_second_bytes{0};  // Bytes in last second
        uint64_t padding3[4];

        // For bandwidth normalization using peak bandwidth probe
        // Track maximum L_i / (tau_i - beta) during the period
        // This represents the peak bandwidth capability observed
        std::atomic<double> max_bw_ratio_{0.0};  // max of L_i / (tau_i - beta)
        std::atomic<double> min_latency_{std::numeric_limits<double>::max()};
        std::atomic<uint64_t> sample_count_{0};
        static constexpr double kBwNormalizationBeta = 5e-6;  // 5 microseconds

        void addSample(uint64_t length, double latency) {
            // Compute bandwidth ratio for this sample
            double pure_latency =
                std::max(1e-9, latency - kBwNormalizationBeta);
            double bw_ratio = static_cast<double>(length) / pure_latency;

            // Update maximum if this sample is better
            double current_max = max_bw_ratio_.load(std::memory_order_relaxed);
            while (bw_ratio > current_max) {
                if (max_bw_ratio_.compare_exchange_weak(
                        current_max, bw_ratio, std::memory_order_relaxed,
                        std::memory_order_relaxed)) {
                    break;
                }
            }

            // Also track minimum latency for monitoring
            double current_min = min_latency_.load(std::memory_order_relaxed);
            while (latency < current_min) {
                if (min_latency_.compare_exchange_weak(
                        current_min, latency, std::memory_order_relaxed,
                        std::memory_order_relaxed)) {
                    break;
                }
            }

            sample_count_.fetch_add(1, std::memory_order_relaxed);
        }

        // Get peak bandwidth ratio for this period
        double getMaxBwRatio() const {
            return max_bw_ratio_.load(std::memory_order_relaxed);
        }

        // Get minimum latency for monitoring/debugging
        double getMinLatency() const {
            return min_latency_.load(std::memory_order_relaxed);
        }

        // Reset tracker for next period
        void resetPeriodTracker() {
            max_bw_ratio_.store(0.0, std::memory_order_relaxed);
            min_latency_.store(std::numeric_limits<double>::max(),
                               std::memory_order_relaxed);
            sample_count_.store(0, std::memory_order_relaxed);
        }

        // Get current inflight bytes
        uint64_t getInflightBytes() const {
            return inflight_bytes.load(std::memory_order_relaxed);
        }

        // Add to inflight
        void addInflight(uint64_t bytes) {
            inflight_bytes.fetch_add(bytes, std::memory_order_relaxed);
        }

        // Subtract from inflight
        void subInflight(uint64_t bytes) {
            inflight_bytes.fetch_sub(bytes, std::memory_order_relaxed);
        }

        // Get current EWMA bandwidth
        double getEwmaBandwidth() const {
            return ewma_bandwidth_bps.load(std::memory_order_relaxed);
        }

        // Calculate theoretical bandwidth in bytes/sec
        double getTheoreticalBandwidth() const {
            if (bw_gbps >= kMinBwGbps && bw_gbps <= kMaxBwGbps)
                return bw_gbps * 1e9 / 8.0;
            return kDefaultBwGbps * 1e9 / 8.0;  // 25e9
        }
    };

   public:
    DeviceQuota() = default;
    ~DeviceQuota() = default;

    DeviceQuota(const DeviceQuota&) = delete;
    DeviceQuota& operator=(const DeviceQuota&) = delete;

    Status loadTopology(std::shared_ptr<Topology>& local_topology);

    std::shared_ptr<Topology> getTopology() const { return local_topology_; }

    Status enableSharedQuota(const std::string& shm_name);

    std::shared_ptr<SharedQuotaManager> getSharedQuota() const {
        return shared_quota_;
    }

    // Allocate devices for a request
    // Returns device IDs for each slice (single device for small requests,
    // multiple devices for large requests)
    Status allocate(uint64_t total_length, uint32_t num_slices,
                    const std::string& location,
                    std::vector<int>& slice_dev_ids, int priority = 0,
                    uint64_t device_mask = ~0ULL);

    // Release completed transfer (update EWMA bandwidth)
    Status release(int dev_id, uint64_t length, double latency);

    // Update traffic statistics for pre-assigned devices (bypasses quota
    // allocation)
    void updateTrafficStats(int dev_id, uint64_t length) {
        auto it = devices_.find(dev_id);
        if (it != devices_.end()) {
            it->second.total_bytes.fetch_add(length, std::memory_order_relaxed);
        }
    }

    void updateStats(int dev_id, uint64_t bytes);

    // Get inflight bytes for a device (used by shared quota)
    uint64_t getInflightBytes(int dev_id) const {
        auto it = devices_.find(dev_id);
        if (it == devices_.end()) return 0;
        return it->second.getInflightBytes();
    }

    void setEnableQuota(bool enable) { enable_quota_ = enable; }
    bool getEnableQuota() const { return enable_quota_; }

    void setPriority(int priority) { priority_ = priority; }
    int getPriority() const { return priority_; }

    // Set slice calculation parameters (must match rdma_transport.cpp)
    void setSliceParams(uint64_t default_block_size,
                        size_t max_slice_count = 64) {
        default_block_size_ = default_block_size;
        max_slice_count_ = max_slice_count;
    }

    // Helper: calculate block_size (must match rdma_transport.cpp logic
    // exactly) Used by both quota allocation and rdma_transport slice creation
    static uint64_t calculateBlockSize(uint64_t total_length,
                                       uint32_t num_slices,
                                       uint64_t default_block_size) {
        auto roundup = [](uint64_t a, uint64_t b) -> uint64_t {
            return (a % b == 0) ? a : (a / b + 1) * b;
        };
        return roundup((total_length + num_slices - 1) / num_slices,
                       default_block_size);
    }

    // Helper: get actual slice length at given offset (must match
    // rdma_transport.cpp)
    static uint64_t getSliceLength(uint64_t total_length, uint64_t offset,
                                   uint64_t block_size) {
        return std::min<uint64_t>(total_length - offset, block_size);
    }

    // Get the rank (0/1/2) of a device for a given location
    int getDeviceRank(const std::string& location, int dev_id) const;

    // Print traffic and weight statistics (called periodically from release)
    void printTrafficStats();

    void fillDevicePriorities();
    int getDevicePriority(int dev_id) const;

    // Scheduling parameters (only used when enable_quota=true)
    struct SchedulingParams {
        // Rank weights for device selection (rank 0:1:2)
        // Reflects PCIe hierarchy: same-NUMA >> cross-socket >> cross-NUMA
        double rank_weights[Topology::DevicePriorityRanks] = {10.0, 2.0, 0.2};

        // EWMA learning rate (0.01 = slow, 0.1 = fast)
        double ewma_alpha = 0.01;

        // Batch threshold: use multi-path when num_slices >= this value
        uint32_t batch_threshold = 4;

        // Enable/disable EWMA learning (bandwidth adaptation)
        bool enable_ewma_learning = true;

        // Fixed overhead for bandwidth normalization (seconds)
        // Represents protocol overhead independent of transfer size
        double bw_normalization_beta = 5e-6;  // 5 microseconds default

        // Parallel correction factor for bw_probe
        // NICs can internally parallelize send/receive operations
        double bw_probe_factor = 2.0;

        // Device priority rotation parameters
        bool enable_device_priority = true;          // Enable/disable rotation
        uint64_t epoch_duration_ns = 1000000000ull;  // 1 second per epoch
        std::vector<int>
            device_base_priorities;  // Base priority for each dev_id
    };

    void setSchedulingParams(const SchedulingParams& params) {
        sched_params_ = params;
    }

    const SchedulingParams& getSchedulingParams() const {
        return sched_params_;
    }

   private:
    std::shared_ptr<Topology> local_topology_;
    std::unordered_map<int, DeviceInfo> devices_;
    mutable std::shared_mutex rwlock_;
    std::shared_ptr<SharedQuotaManager> shared_quota_;
    bool enable_quota_ = true;
    SchedulingParams sched_params_;
    int priority_ = 0;  // PRIO_HIGH (0), PRIO_MEDIUM (1), or PRIO_LOW (2)

    // Slice calculation parameters (must match rdma_transport.cpp)
    uint64_t default_block_size_ = 1;  // Default to 1 (no alignment)
    size_t max_slice_count_ = 64;

    // Check if cross-NUMA device's local node is idle
    bool isCrossNumaNodeIdle(int dev_numa) const;

    // Build candidate devices list for smart scheduling
    Status buildCandidates(const Topology::MemEntry* entry,
                           uint64_t slice_bytes, uint64_t device_mask,
                           std::vector<Candidate>& candidates,
                           int request_priority = 0);  // Request's priority

    // Select device for single-path (small requests)
    void selectSinglePath(const std::vector<Candidate>& candidates,
                          uint32_t num_slices, uint64_t total_length,
                          std::vector<int>& slice_dev_ids);

    // Select devices for multi-path (large requests)
    void selectMultiPath(const std::vector<Candidate>& candidates,
                         uint32_t num_slices, uint64_t total_length,
                         std::vector<int>& slice_dev_ids,
                         bool explore_mode = false);
};

}  // namespace tent
}  // namespace mooncake

#endif  // TENT_QUOTA_H
