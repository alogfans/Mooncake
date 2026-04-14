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

#ifndef TENT_SLICE_H
#define TENT_SLICE_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <list>
#include <mutex>
#include <new>
#include <thread>
#include <type_traits>
#include <vector>

#include "tent/runtime/transport.h"
#include "tent/runtime/slab.h"
#include "tent/common/types.h"  // for PRIO_* constants

namespace mooncake {
namespace tent {
struct RdmaSlice;

struct RdmaSliceList {
    RdmaSlice* first = nullptr;
    int num_slices = 0;
};

// Forward declaration
struct RdmaTask;

class RdmaEndPoint;

struct RdmaSlice {
    void* source_addr = nullptr;
    uint64_t target_addr = 0;
    size_t length = 0;

    RdmaTask* task = nullptr;
    RdmaSlice* next = nullptr;

    uint32_t source_lkey = 0;
    uint32_t target_rkey = 0;
    int source_dev_id = -1;
    int target_dev_id = -1;

    RdmaEndPoint* ep_weak_ptr = nullptr;
    TransferStatusEnum word = TransferStatusEnum::INITIAL;
    int qp_index = 0;
    int retry_count = 0;
    bool failed = false;

    // Latency breakdown: 5 stages of end-to-end transfer
    // Stage 1: Submit -> Enqueue
    uint64_t user_submit_ts = 0;  // When user called submit()
    uint64_t enqueue_ts = 0;      // When slice entered worker queue

    // Stage 2: Queue Wait (enqueue -> dequeue)
    uint64_t dequeue_ts = 0;  // When worker dequeued the slice

    // Stage 3: Dequeue -> Post Send
    uint64_t post_send_start_ts = 0;  // Before ibv_post_send()
    uint64_t post_send_end_ts = 0;    // After ibv_post_send()

    // Stage 4: Post Send -> Poll Success
    uint64_t complete_ts = 0;  // When CQE was polled

    // Stage 5: Poll Success -> API Ready
    uint64_t api_ready_ts = 0;  // When status became queryable via API

    int priority = PRIO_HIGH;      // QoS priority
    uint64_t device_mask = ~0ULL;  // Device mask for quota allocation

    // Batch allocation context for unified quota scheduling
    std::string source_location;         // Source memory location for quota
    uint32_t num_slices_in_request = 1;  // Total slices in the request
    uint32_t slice_index_in_request =
        0;                // Index of this slice within the request
    int device_rank = 0;  // Rank (0/1/2) of selected device
};

using RdmaSliceStorage = Slab<RdmaSlice>;
using RdmaTaskStorage = Slab<RdmaTask>;

struct RdmaTask {
    int num_slices;
    Request request;
    volatile TransferStatusEnum status_word;
    volatile size_t transferred_bytes;
    volatile int success_slices;
    volatile int resolved_slices;
    volatile TransferStatusEnum first_error = PENDING;

    // Reference counting for independent lifecycle management
    // When ref_count reaches 0, the task is deallocated from Slab
    std::atomic<int> ref_count{0};

    void ref() { ref_count.fetch_add(1, std::memory_order_relaxed); }
    void deref() {
        if (ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            RdmaTaskStorage::Get().deallocate(this);
        }
    }
};

static inline void updateSliceStatus(RdmaSlice* slice,
                                     TransferStatusEnum status) {
    if (status == PENDING) return;
    RdmaTask* task = slice->task;
    if (!__sync_bool_compare_and_swap(&slice->word, PENDING, status)) return;
    if (status == COMPLETED) {
        __sync_fetch_and_add(&task->transferred_bytes, slice->length);
        __sync_fetch_and_add(&task->success_slices, 1);
    } else {
        __sync_bool_compare_and_swap(&task->first_error, PENDING, status);
    }
    int resolved = __sync_add_and_fetch(&task->resolved_slices, 1);
    if (resolved >= task->num_slices) {
        TransferStatusEnum final_st = (task->success_slices == task->num_slices)
                                          ? COMPLETED
                                          : task->first_error;
        if (final_st == PENDING) final_st = FAILED;
        __sync_bool_compare_and_swap(&task->status_word, PENDING, final_st);
    }
}

}  // namespace tent
}  // namespace mooncake

#endif  // TENT_SLICE_H