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

/**
 * @file transport_selector.h
 * @brief Configuration-driven transport selection policy
 *
 * Transport selection is driven by configuration with pattern-based rules.
 *
 * Example configuration:
 * {
 *   "policy": [
 *     {
 *       "name": "high_prio_fast",
 *       "segment_type": "memory",
 *       "min_priority": 5,
 *       "rdma_device_names": ["mlx5_0", "mlx5_1", "mlx5_2", "mlx5_3", "mlx5_4",
 * "mlx5_5"], "priority": ["nvlink", "rdma", "shm"]
 *     },
 *     {
 *       "name": "low_prio_slow",
 *       "segment_type": "memory",
 *       "rdma_device_names": ["mlx5_0", "mlx5_1"],
 *       "priority": ["rdma", "tcp"]
 *     },
 *     {
 *       "name": "file_storage",
 *       "segment_type": "file",
 *       "priority": ["gds", "io_uring", "rdma"]
 *     },
 *     {
 *       "name": "default",
 *       "segment_type": "memory",
 *       "priority": []
 *     }
 *   ]
 * }
 *
 * If no "policy" is configured, defaults to original behavior:
 * - File: GDS → IOURING → RDMA
 * - Memory: uses buffer_transports order
 */

#ifndef TENT_TRANSPORT_SELECTOR_H
#define TENT_TRANSPORT_SELECTOR_H

#include "tent/common/config.h"
#include "tent/common/types.h"
#include "tent/runtime/segment.h"
#include "tent/runtime/platform.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mooncake {
namespace tent {

class Transport;

/**
 * @brief Selection context for a single request
 */
struct SelectionContext {
    SegmentType segment_type;       // File or Memory
    bool same_machine;              // Local or remote
    MemoryType local_memory_type;   // CPU, CUDA, etc.
    MemoryType remote_memory_type;  // CPU, CUDA, etc. (for remote)
    const std::vector<TransportType>*
        buffer_transports;  // Pointer to transports in buffer
    size_t transfer_size;   // Transfer size in bytes
    int priority_level;     // Request priority level (higher = more urgent)
};

/**
 * @brief Transport selection policy rule
 */
struct SelectionPolicy {
    // Basic identification
    std::string name;

    // Segment type filter
    SegmentType segment_type;

    // Location filter
    std::optional<bool> same_machine;  // nullopt = don't care

    // Memory type filters (supports patterns: "cuda", "cpu", "npu", "*" for
    // any)
    std::optional<std::string> local_memory_pattern;
    std::optional<std::string> remote_memory_pattern;

    // Size filter (nullopt = no limit)
    std::optional<uint64_t> min_size;  // Minimum transfer size
    std::optional<uint64_t> max_size;  // Maximum transfer size

    // Priority filter: match requests with priority >= min_priority
    // nullopt = match any priority level
    std::optional<int> min_priority;

    // RDMA device allocation: list of device names this policy can use
    // e.g., ["mlx5_0", "mlx5_1", "rocep5s0f0"]
    // Empty = use all available devices
    std::vector<std::string> rdma_device_names;

    // Transport priority list (evaluated in order)
    std::vector<TransportType> priority;
};

/**
 * @brief Configuration-driven transport selector
 */
class TransportSelector {
   public:
    TransportSelector(std::shared_ptr<Config> config);

    /**
     * @brief Select the best transport for a given context
     * @param context Selection context
     * @param available_transports Array of available transports (indexed by
     * TransportType)
     * @param priority_offset Priority offset for fallback (0 = first choice)
     * @return Selected transport type, or UNSPEC if none available
     */
    TransportType select(
        const SelectionContext& context,
        const std::array<std::shared_ptr<Transport>, kSupportedTransportTypes>&
            available_transports,
        int priority_offset = 0);

    /**
     * @brief Get the RDMA device names for the last selected policy
     * @return Vector of device names, empty if no restriction
     */
    const std::vector<std::string>& getRdmaDeviceNames() const {
        return last_rdma_device_names_;
    }

    /**
     * @brief Parse transport type from string
     */
    static TransportType parseTransportType(const std::string& str);

    /**
     * @brief Get transport type name as string
     */
    static std::string transportTypeName(TransportType type);

   private:
    std::vector<SelectionPolicy> policies_;
    std::shared_ptr<Config> config_;
    std::vector<std::string>
        last_rdma_device_names_;  // Cached from last selection

    /**
     * @brief Load policies from configuration
     */
    void loadPolicies();

    /**
     * @brief Check if a policy matches the context
     */
    bool matchesPolicy(const SelectionPolicy& policy,
                       const SelectionContext& context) const;

    /**
     * @brief Check if memory type matches pattern
     */
    bool matchesMemoryPattern(const std::string& pattern,
                              MemoryType type) const;

    /**
     * @brief Check if transport is available for the context
     */
    bool isTransportAvailable(
        TransportType type, const SelectionContext& context,
        const std::array<std::shared_ptr<Transport>, kSupportedTransportTypes>&
            available_transports) const;

    /**
     * @brief Get default policies (used when no config provided)
     */
    static std::vector<SelectionPolicy> getDefaultPolicies();
};

}  // namespace tent
}  // namespace mooncake

#endif  // TENT_TRANSPORT_SELECTOR_H
