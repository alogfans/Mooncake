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
static constexpr double kDefaultBwGbps = 200.0;
static constexpr double kMinBwGbps = 10.0;
static constexpr double kMaxBwGbps = 400.0;

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

        // Update EWMA bandwidth
        void updateEwmaBandwidth(double observed_bps, double alpha) {
            double current = ewma_bandwidth_bps.load(std::memory_order_relaxed);
            double new_bw = alpha * current + (1.0 - alpha) * observed_bps;
            ewma_bandwidth_bps.store(new_bw, std::memory_order_relaxed);
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

    DeviceQuota(const DeviceQuota &) = delete;
    DeviceQuota &operator=(const DeviceQuota &) = delete;

    Status loadTopology(std::shared_ptr<Topology> &local_topology);

    std::shared_ptr<Topology> getTopology() const { return local_topology_; }

    Status enableSharedQuota(const std::string &shm_name);

    std::shared_ptr<SharedQuotaManager> getSharedQuota() const {
        return shared_quota_;
    }

    // Allocate devices for a request
    // Returns device IDs for each slice (single device for small requests,
    // multiple devices for large requests)
    Status allocate(uint64_t total_length, uint32_t num_slices,
                    const std::string &location,
                    std::vector<int> &slice_dev_ids, int priority = 0,
                    uint64_t device_mask = ~0ULL);

    // Release completed transfer (update EWMA)
    Status release(int dev_id, uint64_t length, double latency,
                   int priority = 0);

    // Update traffic statistics for pre-assigned devices (bypasses quota
    // allocation)
    void updateTrafficStats(int dev_id, uint64_t length) {
        auto it = devices_.find(dev_id);
        if (it != devices_.end()) {
            it->second.total_bytes.fetch_add(length, std::memory_order_relaxed);
        }
    }

    void updateStats(int dev_id, uint64_t bytes, bool is_cross);

    // Get inflight bytes for a device (used by shared quota)
    uint64_t getInflightBytes(int dev_id) const {
        auto it = devices_.find(dev_id);
        if (it == devices_.end()) return 0;
        return it->second.getInflightBytes();
    }

    // Get active bytes by priority (for shared quota compatibility - always 0
    // now)
    uint64_t getActiveBytesByPriority(int dev_id, int priority) const {
        // Priority tracking removed, just return 0
        (void)dev_id;
        (void)priority;
        return 0;
    }

    // Set diffusion active bytes (for shared quota compatibility)
    void setDiffusionActiveBytes(int dev_id, uint64_t value) {
        // No longer used, kept for API compatibility
        (void)dev_id;
        (void)value;
    }

    void setCrossNumaAccess(bool enable = true) { allow_cross_numa_ = enable; }

    void setEnableQuota(bool enable) { enable_quota_ = enable; }
    bool getEnableQuota() const { return enable_quota_; }

    // Scheduling parameters (only used when enable_quota=true)
    struct SchedulingParams {
        // EWMA learning rate (0.01 = slow, 0.1 = fast)
        double ewma_alpha = 0.01;

        // Batch threshold: use multi-path when num_slices >= this value
        uint32_t batch_threshold = 4;

        // Cross-NUMA max ratio (0.0 = never use, 1.0 = use freely)
        double cross_numa_max_ratio = 0.0;

        // Cross-NUMA bandwidth penalty (effective_bw = bw * penalty)
        double cross_numa_penalty = 0.7;

        // Enable/disable EWMA learning
        bool enable_ewma_learning = true;
    };

    void setSchedulingParams(const SchedulingParams &params) {
        sched_params_ = params;
    }

    const SchedulingParams &getSchedulingParams() const {
        return sched_params_;
    }

   private:
    std::shared_ptr<Topology> local_topology_;
    std::unordered_map<int, DeviceInfo> devices_;
    mutable std::shared_mutex rwlock_;
    bool allow_cross_numa_ = false;
    std::shared_ptr<SharedQuotaManager> shared_quota_;
    bool enable_quota_ = true;
    SchedulingParams sched_params_;

    // Track source NUMA node for each location
    std::unordered_map<std::string, int> location_numa_cache_;
    mutable std::mutex numa_cache_lock_;

    // Cross-NUMA usage tracking (for ratio control)
    std::atomic<uint64_t> total_local_bytes_{0};
    std::atomic<uint64_t> total_cross_numa_bytes_{0};
};

}  // namespace tent
}  // namespace mooncake

#endif  // TENT_QUOTA_H
