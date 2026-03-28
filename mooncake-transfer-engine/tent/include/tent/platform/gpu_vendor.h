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
 * @file gpu_vendor.h
 * @brief GPU vendor abstraction layer for TENT
 *
 * Provides a unified CUDA-like API surface that maps to vendor-specific APIs.
 * Selects the appropriate vendor header based on compile-time flags.
 */

#ifndef TENT_PLATFORM_GPU_VENDOR_H
#define TENT_PLATFORM_GPU_VENDOR_H

#include <string>

namespace mooncake {
namespace tent {

/**
 * @brief Get the GPU memory prefix for location strings
 */
inline std::string getGpuPrefix() {
#if defined(USE_HIP)
    return "hip:";
#elif defined(USE_MUSA)
    return "musa:";
#elif defined(USE_MACA)
    return "maca:";
#elif defined(USE_ASCEND) || defined(USE_ASCEND_DIRECT)
    return "ascend:";
#elif defined(USE_CUDA)
    return "cuda:";
#else
    return "cpu:";
#endif
}

/**
 * @brief Get the GPU vendor name
 */
inline std::string getGpuVendorName() {
#if defined(USE_HIP)
    return "AMD HIP";
#elif defined(USE_MUSA)
    return "Moore Threads MUSA";
#elif defined(USE_MACA)
    return "Iluvatar MACA";
#elif defined(USE_ASCEND) || defined(USE_ASCEND_DIRECT)
    return "Huawei Ascend";
#elif defined(USE_CUDA)
    return "NVIDIA CUDA";
#else
    return "CPU-only";
#endif
}

} // namespace tent
} // namespace mooncake

// ============================================================
// Vendor-specific API mappings
// ============================================================

#if defined(USE_HIP)
    #include "tent/platform/gpu_vendor/hip.h"
#elif defined(USE_MUSA)
    #include "tent/platform/gpu_vendor/musa.h"
#elif defined(USE_MACA)
    #include "tent/platform/gpu_vendor/maca.h"
#elif defined(USE_ASCEND) || defined(USE_ASCEND_DIRECT)
    #include "tent/platform/gpu_vendor/ascend.h"
#elif defined(USE_CUDA)
    #include "tent/platform/gpu_vendor/cuda.h"
#else
    #include "tent/platform/gpu_vendor/cpu.h"
#endif

#endif  // TENT_PLATFORM_GPU_VENDOR_H
