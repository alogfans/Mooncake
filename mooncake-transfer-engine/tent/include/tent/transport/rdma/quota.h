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
 * @brief DeviceQuota implements NIC selection based on adaptive feedback.
 *
 * Each NIC maintains a smoothed estimate of its average service time,
 * updated after each request completes. The allocator predicts the total
 * completion time of each NIC as:
 *
 *     predicted_time = (active_bytes / bandwidth) + avg_service_time
 *
 * and selects the NIC with the smallest predicted_time.
 *
 * The estimator is updated using exponential smoothing:
 *
 *     avg_service_time <- (1 - alpha) * avg_service_time + alpha *
 * observed_time
 */
class DeviceQuota {
   public:
    struct DeviceInfo {
        int dev_id;
        double bw_gbps;
        int numa_id;
        uint64_t padding0[5];
        std::atomic<uint64_t> active_bytes_by_prio[3];  // [high, medium, low]
        uint64_t padding1[5];
        std::atomic<uint64_t> diffusion_active_bytes{0};
        uint64_t padding2[7];
        std::atomic<double> beta0{
            0.0};  // Fixed overhead (microseconds), origin/main: 0.0
        uint64_t padding3[7];
        std::atomic<double> beta1{
            1.0};  // Bandwidth correction factor, origin/main: 1.0
        uint64_t padding4[7];
        std::atomic<uint64_t> last_update_ns{
            0};  // Last update time (RDTSCP-based)
        uint64_t padding5[7];

        // Get total active bytes across all priorities
        uint64_t getTotalActiveBytes() const {
            uint64_t sum = 0;
            for (int i = 0; i < NUM_PRIORITIES; ++i)
                sum += active_bytes_by_prio[i].load(std::memory_order_relaxed);
            return sum;
        }

        // Get active bytes for specific priority
        uint64_t getActiveBytes(int priority) const {
            if (priority < 0 || priority >= NUM_PRIORITIES) return 0;
            return active_bytes_by_prio[priority].load(
                std::memory_order_relaxed);
        }

        // Add bytes to specific priority bucket
        void addBytes(int priority, uint64_t bytes) {
            if (priority >= 0 && priority < NUM_PRIORITIES)
                active_bytes_by_prio[priority].fetch_add(
                    bytes, std::memory_order_relaxed);
        }

        // Subtract bytes from specific priority bucket
        void subBytes(int priority, uint64_t bytes) {
            if (priority >= 0 && priority < NUM_PRIORITIES)
                active_bytes_by_prio[priority].fetch_sub(
                    bytes, std::memory_order_relaxed);
        }

        // Calculate bandwidth in bytes/sec (with fallback to default)
        double getBandwidthBytesPerSec() const {
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

    Status allocate(uint64_t length, const std::string &location,
                    int &chosen_dev_id, int priority = 0,
                    uint64_t device_mask = ~0ULL);

    Status release(int dev_id, uint64_t length, double latency,
                   int priority = 0);

    void setDiffusionActiveBytes(int dev_id, uint64_t value) {
        devices_[dev_id].diffusion_active_bytes.store(
            value, std::memory_order_relaxed);
    }

    // Get total active bytes across all priorities
    uint64_t getActiveBytes(int dev_id) {
        return devices_[dev_id].getTotalActiveBytes();
    }

    // Get active bytes for specific priority
    uint64_t getActiveBytesByPriority(int dev_id, int priority) {
        return devices_[dev_id].getActiveBytes(priority);
    }

    void setLearningRate(double alpha) { alpha_ = std::clamp(alpha, 0.0, 1.0); }

    void setLocalWeight(double local_weight) {
        local_weight_ = std::clamp(local_weight, 0.0, 1.0);
    }

    void setDiffusionInterval(uint64_t msec) {
        diffusion_interval_ = msec * 1000000ull;
    }

    void setCrossNumaAccess(bool enable = true) { allow_cross_numa_ = enable; }

    void setRobustClamping(bool enable) { use_robust_clamp_ = enable; }

    bool getRobustClamping() const { return use_robust_clamp_; }

    void setEnableQuota(bool enable) { enable_quota_ = enable; }
    bool getEnableQuota() const { return enable_quota_; }

    // Update device bandwidth from IBV port speed (bandwidth passthrough)
    void updateDeviceBandwidth(int dev_id, int ibv_speed) {
        auto it = devices_.find(dev_id);
        if (it != devices_.end()) {
            it->second.bw_gbps = speedToGbps(ibv_speed);
        }
    }

    // Convert IBV_SPEED enum to Gbps
    static double speedToGbps(int speed) {
        switch (speed) {
            case 1:   return 10.0;    // SDR: 2.5 Gbps/lane * 4
            case 2:   return 30.0;    // DDR: 5 Gbps/lane * 4
            case 4:   return 40.0;    // QDR: 10 Gbps/lane * 4
            case 8:   return 41.25;   // FDR10: 10.3125 Gbps/lane * 4
            case 16:  return 56.0;    // FDR: 14.0625 Gbps/lane * 4
            case 32:  return 100.0;   // EDR: 25 Gbps/lane * 4
            case 64:  return 200.0;   // HDR: 50 Gbps/lane * 4
            case 128: return 400.0;   // NDR: 100 Gbps/lane * 4
            case 256: return 800.0;   // XDR: 200 Gbps/lane * 4
            default:  return 200.0;   // Default to HDR
        }
    }

   private:
    std::shared_ptr<Topology> local_topology_;
    std::unordered_map<int, DeviceInfo> devices_;
    mutable std::shared_mutex rwlock_;
    bool allow_cross_numa_ = false;
    double alpha_ = 0.01;
    double local_weight_ = 0.9;
    uint64_t diffusion_interval_ = 10 * 1000000ull;
    bool use_robust_clamp_ = true;  // Use adaptive bounds instead of static
    uint32_t sample_window_size_ =
        100;  // Number of samples for percentile calculation
    std::shared_ptr<SharedQuotaManager> shared_quota_;
    bool enable_quota_ = true;
    bool update_quota_params_ = true;
};

}  // namespace tent
}  // namespace mooncake

#endif  // TENT_QUOTA_H
