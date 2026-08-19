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

#ifndef XFER_UTILS_H
#define XFER_UTILS_H

#include <string>
#include <unordered_map>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <glog/logging.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <chrono>
#include <limits>
#include <cstring>

#include "tent/common/utils/os.h"
#include "tent/common/utils/random.h"

#if defined(USE_CUDA)
#include <cuda_runtime.h>
#elif defined(USE_SUNRISE)
#include "cuda_alike.h"
#endif

#ifdef USE_HIP
#include <hip/hip_runtime.h>
#endif

#define CHECK_FAIL(call)                                        \
    do {                                                        \
        auto status_ = call;                                    \
        if (!status_.ok()) {                                    \
            LOG(INFO) << "Found error: " << status_.ToString(); \
            exit(EXIT_FAILURE);                                 \
        }                                                       \
    } while (0)

namespace mooncake {
namespace tent {
struct XferBenchConfig {
    static void loadFromFlags();

    static std::string seg_name;
    static std::string seg_type;
    // Comma-separated segment types for mixed DRAM+VRAM runs, e.g.
    // "dram,vram". Empty falls back to --seg_type (single type).
    static std::string seg_type_mix;
    static std::string target_seg_name;
    static std::string op_type;
    static bool check_consistency;

    static size_t total_buffer_size;
    static size_t start_block_size;
    static size_t max_block_size;
    static size_t start_batch_size;
    static size_t max_batch_size;
    static int duration;
    static int max_num_threads;
    static int start_num_threads;
    static size_t target_offset;
    static size_t target_range_size;
    static std::string qos_classes;
    static std::string qos_classes_json;
    static std::string workload_classes_json;
    static double qos_link_capacity_gbps;
    static std::string qos_output_jsonl;
    static uint64_t request_interval_us;
    static uint64_t deadline_us;
    static int deadline_tight_threads;
    static bool deadline_bw_arbitration;

    static std::string metadata_type;
    static std::string metadata_url_list;
    static int rpc_server_port;
    static std::string xport_type;
    static std::string backend;
    static bool notifi;
    static std::string tent_transport_hint;
    static std::string tent_intent_type;

    static int local_gpu_id;
    static int target_gpu_id;
};

struct XferMetricStats {
   public:
    double min() const {
        if (samples.empty()) return 0.0;
        return *std::min_element(samples.begin(), samples.end());
    }

    double max() const {
        if (samples.empty()) return 0.0;
        return *std::max_element(samples.begin(), samples.end());
    }

    double avg() const {
        if (samples.empty()) return 0.0;
        double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
        return sum / samples.size();
    }

    double p90() { return percentile(90.0); }

    double p95() { return percentile(95.0); }

    double p99() { return percentile(99.0); }

    double p999() { return percentile(99.9); }

    double fractionAtOrBelow(double threshold) const {
        if (samples.empty()) return 0.0;
        const auto count = std::count_if(
            samples.begin(), samples.end(),
            [threshold](double value) { return value <= threshold; });
        return static_cast<double>(count) / samples.size();
    }

    void add(double value) { samples.push_back(value); }

    void add(const std::vector<double>& values) {
        samples.insert(samples.end(), values.begin(), values.end());
    }

    void clear() { samples.clear(); }

    size_t count() { return samples.size(); }

   private:
    double percentile(double p);

   private:
    std::vector<double> samples;
};

struct XferBenchStats {
    XferMetricStats total_duration;
    XferMetricStats transfer_duration;
    XferMetricStats instant_bandwidth;
};

class XferBenchTimer {
   public:
    XferBenchTimer() : start_ts_(getCurrentTimeNs()) {}

    void reset() { start_ts_ = getCurrentTimeNs(); }

    uint64_t lap_us(bool reset = true) {
        auto now_ts = getCurrentTimeNs();
        auto duration = now_ts - start_ts_;
        if (reset) start_ts_ = now_ts;
        return duration / 1000;
    }

   private:
    inline uint64_t getCurrentTimeNs() {
        auto ret = std::chrono::steady_clock::now().time_since_epoch();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(ret)
            .count();
    }

    uint64_t start_ts_;
};

void printStatsHeader();

void printStats(size_t block_size, size_t batch_size, XferBenchStats& stats,
                int num_threads);

void printDeadlineGroupStats(const char* group, size_t block_size,
                             size_t batch_size, XferBenchStats& stats,
                             int num_threads, uint64_t deadline_us);

std::vector<std::string> splitCommaSeparated(const std::string& value);

uint64_t stableDataSeed(uint64_t target_addr);

static inline uint64_t checkedMul(uint64_t lhs, uint64_t rhs,
                                  const char* label) {
    if (rhs != 0 && lhs > std::numeric_limits<uint64_t>::max() / rhs) {
        LOG(FATAL) << label << " overflows uint64_t: " << lhs << " * " << rhs;
    }
    return lhs * rhs;
}

static inline uint64_t checkedAdd(uint64_t lhs, uint64_t rhs,
                                  const char* label) {
    if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
        LOG(FATAL) << label << " overflows uint64_t: " << lhs << " + " << rhs;
    }
    return lhs + rhs;
}

static inline bool rangeContains(uint64_t offset, uint64_t bytes,
                                 uint64_t limit) {
    return offset <= limit && bytes <= limit - offset;
}

#if defined(USE_CUDA) || defined(USE_SUNRISE)
static inline bool isCudaMemory(void* ptr) {
    cudaPointerAttributes attr;
    auto ret = cudaPointerGetAttributes(&attr, ptr);
    return ret == cudaSuccess && attr.type == cudaMemoryTypeDevice;
}
#endif

#ifdef USE_HIP
static inline bool isHipMemory(void* ptr) {
    hipPointerAttribute_t attr;
    auto ret = hipPointerGetAttributes(&attr, ptr);
    return ret == hipSuccess && attr.type == hipMemoryTypeDevice;
}
#endif

static inline bool isGpuMemory(void* ptr) {
#if defined(USE_CUDA) || defined(USE_SUNRISE)
    if (isCudaMemory(ptr)) return true;
#endif
#ifdef USE_HIP
    if (isHipMemory(ptr)) return true;
#endif
    return false;
}

static inline uint64_t mixConsistencyWord(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

static inline uint64_t consistencyMarker(uint64_t seed, uint64_t chunk_index,
                                         uint64_t marker_index) {
    return mixConsistencyWord(seed ^ (chunk_index * 0xd1342543de82ef95ULL) ^
                              (marker_index * 0xa0761d6478bd642fULL));
}

static inline bool checkRepeatedByte(const uint8_t* data, size_t length,
                                     uint8_t expected, size_t base_offset) {
    for (size_t i = 0; i < length; ++i) {
        if (data[i] != expected) {
            LOG(FATAL) << "Inconsistent data detected at offset "
                       << (base_offset + i) << ": expected "
                       << static_cast<int>(expected) << ", got "
                       << static_cast<int>(data[i]);
            return false;
        }
    }
    return true;
}

static inline void fillConsistencyPatternHost(uint8_t* data, size_t length,
                                              uint64_t seed) {
    constexpr size_t kChunkSize = 4096;
    constexpr size_t kMarkerSize = sizeof(uint64_t);
    for (size_t offset = 0, chunk = 0; offset < length;
         offset += kChunkSize, ++chunk) {
        const size_t chunk_len = std::min(kChunkSize, length - offset);
        const uint8_t fill =
            static_cast<uint8_t>(mixConsistencyWord(seed + chunk) & 0xff);
        std::memset(data + offset, fill, chunk_len);
        if (chunk_len >= kMarkerSize) {
            const uint64_t front = consistencyMarker(seed, chunk, 0);
            std::memcpy(data + offset, &front, kMarkerSize);
        }
        if (chunk_len >= 2 * kMarkerSize) {
            const uint64_t back = consistencyMarker(seed, chunk, 1);
            std::memcpy(data + offset + chunk_len - kMarkerSize, &back,
                        kMarkerSize);
        }
    }
}

static inline void verifyConsistencyPatternHost(const uint8_t* data,
                                                size_t length,
                                                uint64_t seed) {
    constexpr size_t kChunkSize = 4096;
    constexpr size_t kMarkerSize = sizeof(uint64_t);
    for (size_t offset = 0, chunk = 0; offset < length;
         offset += kChunkSize, ++chunk) {
        const size_t chunk_len = std::min(kChunkSize, length - offset);
        const uint8_t fill =
            static_cast<uint8_t>(mixConsistencyWord(seed + chunk) & 0xff);
        const uint8_t* chunk_data = data + offset;
        size_t constant_begin = 0;
        size_t constant_end = chunk_len;
        if (chunk_len >= kMarkerSize) {
            uint64_t actual = 0;
            const uint64_t expected = consistencyMarker(seed, chunk, 0);
            std::memcpy(&actual, chunk_data, kMarkerSize);
            if (actual != expected) {
                LOG(FATAL) << "Inconsistent data detected at front marker for "
                           << "chunk " << chunk << " offset " << offset;
            }
            constant_begin = kMarkerSize;
        }
        if (chunk_len >= 2 * kMarkerSize) {
            uint64_t actual = 0;
            const uint64_t expected = consistencyMarker(seed, chunk, 1);
            const size_t marker_offset = chunk_len - kMarkerSize;
            std::memcpy(&actual, chunk_data + marker_offset, kMarkerSize);
            if (actual != expected) {
                LOG(FATAL) << "Inconsistent data detected at back marker for "
                           << "chunk " << chunk << " offset "
                           << (offset + marker_offset);
            }
            constant_end = marker_offset;
        }
        if (constant_begin < constant_end) {
            checkRepeatedByte(chunk_data + constant_begin,
                              constant_end - constant_begin, fill,
                              offset + constant_begin);
        }
    }
}

static inline std::vector<uint8_t>& consistencyScratch() {
    thread_local std::vector<uint8_t> scratch;
    return scratch;
}

static inline void fillData(void* addr, size_t length, uint64_t seed) {
#if defined(USE_CUDA)
    if (isCudaMemory(addr)) {
        auto& scratch = consistencyScratch();
        scratch.resize(length);
        fillConsistencyPatternHost(scratch.data(), length, seed);
        auto err = cudaMemcpy(addr, scratch.data(), length, cudaMemcpyDefault);
        LOG_ASSERT(err == cudaSuccess)
            << "cudaMemcpy failed: " << cudaGetErrorString(err);
        return;
    }
#elif defined(USE_SUNRISE)
    if (isCudaMemory(addr)) {
        auto& scratch = consistencyScratch();
        scratch.resize(length);
        fillConsistencyPatternHost(scratch.data(), length, seed);
        auto err = cudaMemcpy(addr, scratch.data(), length, cudaMemcpyDefault);
        LOG_ASSERT(err == cudaSuccess)
            << "cudaMemcpy failed: " << cudaGetErrorString(err);
        return;
    }
#endif
#ifdef USE_HIP
    if (isHipMemory(addr)) {
        auto& scratch = consistencyScratch();
        scratch.resize(length);
        fillConsistencyPatternHost(scratch.data(), length, seed);
        auto err = hipMemcpy(addr, scratch.data(), length, hipMemcpyDefault);
        LOG_ASSERT(err == hipSuccess)
            << "hipMemcpy failed: " << hipGetErrorString(err);
        return;
    }
#endif
    fillConsistencyPatternHost(static_cast<uint8_t*>(addr), length, seed);
}

static inline uint64_t fillData(void* addr, size_t length) {
    uint64_t seed = SimpleRandom::Get().next();
    seed = (seed << 32) | SimpleRandom::Get().next();
    fillData(addr, length, seed);
    return seed;
}

static inline void verifyData(void* addr, size_t length, uint64_t seed) {
#if defined(USE_CUDA)
    if (isCudaMemory(addr)) {
        auto& scratch = consistencyScratch();
        scratch.resize(length);
        cudaMemcpy(scratch.data(), addr, length, cudaMemcpyDefault);
        verifyConsistencyPatternHost(scratch.data(), length, seed);
        return;
    }
#elif defined(USE_SUNRISE)
    if (isCudaMemory(addr)) {
        auto& scratch = consistencyScratch();
        scratch.resize(length);
        auto err =
            cudaMemcpy(scratch.data(), addr, length, cudaMemcpyDeviceToHost);
        LOG_ASSERT(err == cudaSuccess)
            << "cudaMemcpy failed: " << cudaGetErrorString(err);
        verifyConsistencyPatternHost(scratch.data(), length, seed);
        return;
    }
#endif
#ifdef USE_HIP
    if (isHipMemory(addr)) {
        auto& scratch = consistencyScratch();
        scratch.resize(length);
        hipMemcpy(scratch.data(), addr, length, hipMemcpyDefault);
        verifyConsistencyPatternHost(scratch.data(), length, seed);
        return;
    }
#endif
    verifyConsistencyPatternHost(static_cast<const uint8_t*>(addr), length,
                                 seed);
}

enum OpCode { READ, WRITE };

}  // namespace tent
}  // namespace mooncake

#endif  // XFER_UTILS_H
