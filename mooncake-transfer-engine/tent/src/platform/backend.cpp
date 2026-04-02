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
 * @file backend.cpp
 * @brief Platform backend implementation
 *
 * This single file is compiled multiple times with different platform flags:
 *   -DUSE_CUDA   → libtent_platform_cuda.so
 *   -DUSE_MUSA   → libtent_platform_musa.so
 *   -DUSE_HIP    → libtent_platform_hip.so
 *   -DUSE_ASCEND → libtent_platform_ascend.so
 *
 * The gpu_vendor.h macros handle the API mapping to vendor-specific calls.
 */

#include "tent/runtime/platform.h"
#include "tent/platform/gpu_vendor.h"
#include "tent/common/status.h"
#include "tent/common/utils/prefault.h"

#include <numa.h>
#include <glog/logging.h>
#include <cstring>
#include <cstdlib>

namespace mooncake {
namespace tent {

class PlatformBackend : public IPlatformBackend {
   public:
    std::string name() const override {
#if defined(USE_CUDA)
        return "CUDA";
#elif defined(USE_MUSA)
        return "MUSA";
#elif defined(USE_HIP)
        return "HIP";
#elif defined(USE_MACA)
        return "MACA";
#elif defined(USE_ASCEND) || defined(USE_ASCEND_DIRECT)
        return "Ascend";
#else
        return "CPU";
#endif
    }

    Status initialize(std::shared_ptr<Config> config) override {
        config_ = config;

#if defined(HAS_DEVICE_SUPPORT)
        int count = 0;
        auto ret = cudaGetDeviceCount(&count);
        if (ret != CUDA_SUCCESS || count == 0) {
            LOG(WARNING) << name() << " not available";
            return Status::InternalError(name() + " not available");
        }
        LOG(INFO) << name() << " backend initialized with " << count
                  << " device(s)";
#else
        LOG(INFO) << "CPU backend initialized";
#endif
        return Status::OK();
    }

    Status probe(std::vector<Topology::NicEntry>& nic_list,
                 std::vector<Topology::MemEntry>& mem_list) override {
#if defined(HAS_DEVICE_SUPPORT)
        int device_count = 0;
        auto ret = cudaGetDeviceCount(&device_count);
        if (ret != CUDA_SUCCESS || device_count == 0) {
            return Status::OK();
        }

        LOG(INFO) << "Found " << device_count << " " << name() << " device(s)";
        for (int i = 0; i < device_count; ++i) {
            Topology::MemEntry entry;
            entry.name = getGpuPrefix() + std::to_string(i);
            mem_list.push_back(entry);
        }
#endif
        return Status::OK();
    }

    Status allocate(void** pptr, size_t size, MemoryOptions& options) override {
#if defined(HAS_DEVICE_SUPPORT)
        LocationParser location(options.location);
        auto gpu_prefix = getGpuPrefix();

        // Check if location matches GPU type
        if (!location.type().empty() &&
            gpu_prefix.find(location.type() + ":") == 0) {
            int device = 0;
            auto err = cudaGetDevice(&device);
            if (err != CUDA_SUCCESS)
                return Status::InternalError("cudaGetDevice failed");
            err = cudaSetDevice(location.index());
            if (err != CUDA_SUCCESS)
                return Status::InternalError("cudaSetDevice failed");
            err = cudaMalloc(pptr, size);
            if (err != CUDA_SUCCESS)
                return Status::InternalError("cudaMalloc failed");
            cudaSetDevice(device);
            return Status::OK();
        }
#endif

        // CPU allocation (common to all platforms)
        int socket_id = 0;
        LocationParser location(options.location);
        if (location.type() == "cpu") socket_id = location.index();
        *pptr = numa_alloc_onnode(size, socket_id);
        return *pptr ? Status::OK()
                     : Status::InternalError("numa_alloc failed");
    }

    Status free(void* ptr, size_t size) override {
#if defined(HAS_DEVICE_SUPPORT)
        cudaPointerAttributes attr;
        auto ret = cudaPointerGetAttributes(&attr, ptr);
        if (ret == CUDA_SUCCESS && attr.type == cudaMemoryTypeDevice) {
            auto err = cudaFree(ptr);
            return (err == CUDA_SUCCESS)
                       ? Status::OK()
                       : Status::InternalError("cudaFree failed");
        }
#endif
        numa_free(ptr, size);
        return Status::OK();
    }

    Status copy(void* dst, void* src, size_t length) override {
#if defined(HAS_DEVICE_SUPPORT)
        auto err = cudaMemcpy(dst, src, length, cudaMemcpyDefault);
        return (err == CUDA_SUCCESS)
                   ? Status::OK()
                   : Status::InternalError("cudaMemcpy failed");
#else
        memcpy(dst, src, length);
        return Status::OK();
#endif
    }

    MemoryType getMemoryType(void* addr) override {
#if defined(HAS_DEVICE_SUPPORT)
        cudaPointerAttributes attr;
        auto ret = cudaPointerGetAttributes(&attr, addr);
        if (ret == CUDA_SUCCESS && attr.type == cudaMemoryTypeDevice) {
#if defined(USE_CUDA)
            return MTYPE_CUDA;
#elif defined(USE_ASCEND)
            return MTYPE_ASCEND;
#elif defined(USE_HIP)
            return MTYPE_HIP;
#elif defined(USE_MUSA)
            return MTYPE_MUSA;
#elif defined(USE_MACA)
            return MTYPE_MACA;
#else
            return MTYPE_CUDA;  // Fallback
#endif
        }
#endif
        (void)addr;
        return MTYPE_CPU;
    }

    const std::vector<RangeLocation> getLocation(void* start, size_t len,
                                                 bool skip_prefault) override {
        std::vector<RangeLocation> locations;
#if defined(HAS_DEVICE_SUPPORT)
        cudaPointerAttributes attr;
        auto ret = cudaPointerGetAttributes(&attr, start);

        if (ret == CUDA_SUCCESS && attr.type == cudaMemoryTypeDevice) {
            int device = 0;
            cudaGetDevice(&device);
            RangeLocation rl;
            rl.start = reinterpret_cast<uint64_t>(start);
            rl.len = len;
            rl.location = getGpuPrefix() + std::to_string(device);
            locations.push_back(rl);
            return locations;
        }
#endif
        return locations;
    }

    int getDeviceCount() const override {
#if defined(HAS_DEVICE_SUPPORT)
        int count = 0;
        cudaGetDeviceCount(&count);
        return count;
#else
        return 0;
#endif
    }

    std::string getPrefix() const override {
#if defined(HAS_DEVICE_SUPPORT)
        return getGpuPrefix();
#else
        return "cpu:";
#endif
    }

   private:
    std::shared_ptr<Config> config_;
};

// Export the backend - symbol name is the same for all platforms
extern "C" {
std::shared_ptr<mooncake::tent::IPlatformBackend> CreatePlatformBackend() {
    return std::make_shared<PlatformBackend>();
}
}

}  // namespace tent
}  // namespace mooncake
