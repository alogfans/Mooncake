// Copyright 2024 KVCache.AI
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

#ifndef TENT_RANDOM_H
#define TENT_RANDOM_H

#include <atomic>

#include "tent/common/utils/os.h"

namespace mooncake {
namespace tent {
class SimpleRandom {
   public:
    SimpleRandom(uint32_t seed) : current(seed) {}

    static SimpleRandom& Get() {
        static std::atomic<uint64_t> g_incr_val(0);
        thread_local SimpleRandom g_random(getCurrentTimeInNano() +
                                           g_incr_val.fetch_add(1));
        return g_random;
    }

    uint32_t next() {
        current = (a * current + c) & m;
        return current;
    }

    uint32_t next(uint32_t max) { return (next() >> 12) % max; }

   private:
    uint32_t current;
    static const uint32_t a = 1664525;
    static const uint32_t c = 1013904223;
    static const uint32_t m = 0xFFFFFFFF;
};

static inline uint64_t SeedOnce() {
    // Try to leverage existing util; fallback to time-based seed.
    // Replace with your project's canonical seed source if available.
    uint64_t s = getCurrentTimeInNano();
    // xorshift mix
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    return s ? s : 88172645463325252ull;
}

static inline uint64_t XorShift64(uint64_t& s) {
    // xorshift64*
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    return s * 2685821657736338717ull;
}

static inline double Rand01(uint64_t& s) {
    // [0,1)
    const uint64_t r = XorShift64(s);
    return (r >> 11) * (1.0 / 9007199254740992.0);  // 2^53
}

static inline size_t RandMod(uint64_t& s, size_t n) {
    return (n == 0) ? 0 : static_cast<size_t>(XorShift64(s) % n);
}

}  // namespace tent
}  // namespace mooncake

#endif  // TENT_RANDOM_H