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

#ifndef TENT_SHARED_QUOTA_H
#define TENT_SHARED_QUOTA_H

#include "quota.h"
#include "tent/common/status.h"
#include "tent/runtime/topology.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>
#include <string>
#include <cstring>
#include <atomic>
#include <vector>
#include <iostream>
#include <errno.h>
#include <thread>

namespace mooncake {
namespace tent {

static constexpr uint64_t SHM_MAGIC = 0x2025082772805203ULL;
static constexpr int SHM_VERSION = 10;

// Time slice configuration
// Slot 0:            HIGH only
// Slot 1:            MEDIUM + HIGH
// Slot 2:            ALL (LOW + MEDIUM + HIGH)
// Then repeat
static constexpr int NUM_SLOTS = 3;  // 3 slots per cycle

struct SharedHeader {
    uint64_t magic;
    int32_t version;

    // Global slot index (0, 1, 2) - legacy, for backward compatibility
    std::atomic<int> current_slot;

    // Per-device slot indices (0, 1, 2)
    // Each device advances independently
    // Max 16 devices supported
    static constexpr int MAX_DEVICES = 16;
    std::atomic<int> device_slots[MAX_DEVICES];

    pthread_mutex_t global_mutex;
};

class DeviceQuota;
class SharedQuotaManager {
   public:
    explicit SharedQuotaManager(DeviceQuota* local_quota);
    ~SharedQuotaManager();

    Status attach(const std::string& shm_name);
    Status detach();

    // Check if current process can send (global slot, for backward
    // compatibility)
    bool canSend();

    // Get device's current priority slot (0=HIGH, 1=MEDIUM, 2=LOW)
    int getDevicePriority(int device_id);

    // Set time slice duration in milliseconds (must be > 0)
    void setTimesliceUnitMs(int ms) { timeslice_unit_ms_ = ms; }
    int getTimesliceUnitMs() const { return timeslice_unit_ms_; }

   private:
    void startBackgroundThread();
    void stopBackgroundThread();
    void backgroundThreadLoop();
    Status initializeHeader();
    Status initMutex(pthread_mutex_t* m);

    // Check if a priority is allowed in the given slot
    bool isPriorityAllowedInSlot(int priority, int slot) const;

   private:
    std::string name_;
    SharedHeader* hdr_;
    int fd_;
    size_t size_;
    bool created_;
    DeviceQuota* local_quota_;
    int timeslice_unit_ms_ = 2;  // Default: 2ms per slot

    // Background thread
    std::thread background_thread_;
    std::atomic<bool> background_running_;
};

}  // namespace tent
}  // namespace mooncake

#endif  // TENT_SHARED_QUOTA_H
