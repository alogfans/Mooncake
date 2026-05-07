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
    // Candidate device for allocation
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
        std::atomic<double> ewma_bandwidth_bps{50e9};
        uint64_t padding2[7];
        std::atomic<uint64_t> total_bytes{0};        // Total bytes transferred
        std::atomic<uint64_t> last_second_bytes{0};  // Bytes in last second
        uint64_t padding3[4];

        // For bandwidth normalization using peak bandwidth probe
        std::atomic<double> max_bw_ratio_{0.0};  // max of L_i / (tau_i - beta)
        std::atomic<double> min_latency_{std::numeric_limits<double>::max()};
        std::atomic<uint64_t> sample_count_{0};
        static constexpr double kBwNormalizationBeta = 5e-6;  // 5 microseconds

        void addSample(uint64_t length, double latency) {
            double pure_latency =
                std::max(1e-9, latency - kBwNormalizationBeta);
            double bw_ratio = static_cast<double>(length) / pure_latency;

            double current_max = max_bw_ratio_.load(std::memory_order_relaxed);
            while (bw_ratio > current_max) {
                if (max_bw_ratio_.compare_exchange_weak(
                        current_max, bw_ratio, std::memory_order_relaxed,
                        std::memory_order_relaxed)) {
                    break;
                }
            }

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

        double getMaxBwRatio() const {
            return max_bw_ratio_.load(std::memory_order_relaxed);
        }

        double getMinLatency() const {
            return min_latency_.load(std::memory_order_relaxed);
        }

        void resetPeriodTracker() {
            max_bw_ratio_.store(0.0, std::memory_order_relaxed);
            min_latency_.store(std::numeric_limits<double>::max(),
                               std::memory_order_relaxed);
            sample_count_.store(0, std::memory_order_relaxed);
        }

        uint64_t getInflightBytes() const {
            return inflight_bytes.load(std::memory_order_relaxed);
        }

        void addInflight(uint64_t bytes) {
            inflight_bytes.fetch_add(bytes, std::memory_order_relaxed);
        }

        void releaseInflight(uint64_t bytes) {
            inflight_bytes.fetch_sub(bytes, std::memory_order_relaxed);
        }

        double getEwmaBandwidth() const {
            return ewma_bandwidth_bps.load(std::memory_order_relaxed);
        }

        double getTheoreticalBandwidth() const {
            if (bw_gbps >= kMinBwGbps && bw_gbps <= kMaxBwGbps)
                return bw_gbps * 1e9 / 8.0;
            return kDefaultBwGbps * 1e9 / 8.0;
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

    // Allocate devices for a request (new API)
    // slice_bytes: pre-calculated slice size from rdma_transport to ensure consistency
    Status allocate(uint64_t total_length, uint32_t num_slices,
                    uint64_t slice_bytes,
                    const std::string &location,
                    std::vector<int> &slice_dev_ids,
                    uint64_t device_mask = ~0ULL);

    // Backward-compatible allocate for single device selection
    Status allocate(uint64_t length, const std::string &location,
                    int &chosen_dev_id);

    // Release completed transfer (update EWMA bandwidth)
    Status release(int dev_id, uint64_t length, double latency);

    void updateTrafficStats(int dev_id, uint64_t length) {
        auto it = devices_.find(dev_id);
        if (it != devices_.end()) {
            it->second.total_bytes.fetch_add(length, std::memory_order_relaxed);
        }
    }

    void updateStats(int dev_id, uint64_t bytes);

    uint64_t getInflightBytes(int dev_id) const {
        auto it = devices_.find(dev_id);
        if (it == devices_.end()) return 0;
        return it->second.getInflightBytes();
    }

    uint64_t getActiveBytesByPriority(int dev_id, int priority) const {
        (void)dev_id;
        (void)priority;
        return 0;
    }

    void setDiffusionActiveBytes(int dev_id, uint64_t value) {
        (void)dev_id;
        (void)value;
    }

    void setEnableQuota(bool enable) { enable_quota_ = enable; }
    bool getEnableQuota() const { return enable_quota_; }

    // Legacy API compatibility methods
    void setCrossNumaAccess(bool enable = true) { (void)enable; }
    void setLocalWeight(double local_weight) { (void)local_weight; }
    void setLearningRate(double alpha) {
        sched_params_.ewma_alpha = std::clamp(alpha, 0.0, 1.0);
    }
    void setDiffusionInterval(uint64_t msec) { (void)msec; }

    void setSliceParams(uint64_t default_block_size,
                        size_t max_slice_count = 64) {
        default_block_size_ = default_block_size;
        max_slice_count_ = max_slice_count;
    }

    int getDeviceRank(const std::string &location, int dev_id) const;

    void printTrafficStats();

    void fillDevicePriorities();
    int getDevicePriority(int dev_id) const;

    struct SchedulingParams {
        double rank_weights[Topology::DevicePriorityRanks] = {10.0, 2.0, 0.2};
        double ewma_alpha = 0.01;
        uint32_t batch_threshold = 4;
        bool enable_ewma_learning = true;
        double bw_normalization_beta = 5e-6;
        double bw_probe_factor = 2.0;
        bool enable_device_priority = true;
        uint64_t epoch_duration_ns = 1000000000ull;
        std::vector<int> device_base_priorities;
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
    std::shared_ptr<SharedQuotaManager> shared_quota_;
    bool enable_quota_ = true;
    SchedulingParams sched_params_;
    uint64_t default_block_size_ = 1;
    size_t max_slice_count_ = 64;

    bool isCrossNumaNodeIdle(int dev_numa) const;

    Status buildCandidates(const Topology::MemEntry *entry,
                           uint64_t slice_bytes, uint64_t device_mask,
                           std::vector<Candidate> &candidates);

    void selectSinglePath(const std::vector<Candidate> &candidates,
                          uint32_t num_slices, uint64_t total_length,
                          std::vector<int> &slice_dev_ids);

    void selectMultiPath(const std::vector<Candidate> &candidates,
                         uint32_t num_slices, uint64_t total_length,
                         std::vector<int> &slice_dev_ids,
                         bool explore_mode = false);
};

}  // namespace tent
}  // namespace mooncake

#endif  // TENT_QUOTA_H