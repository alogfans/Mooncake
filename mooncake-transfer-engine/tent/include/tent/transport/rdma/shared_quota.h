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

static constexpr int MAX_PID_SLOTS = 256;
static constexpr uint64_t SHM_MAGIC = 0x2025082772805203ULL;
static constexpr int SHM_VERSION = 7;

// Time slice configuration (in milliseconds)
static constexpr int TIMESLICE_UNIT_MS = 10;  // 10ms per unit
static constexpr int TIMESLICE_WEIGHT_HIGH = 9;   // HIGH: 9 units = 90ms
static constexpr int TIMESLICE_WEIGHT_MEDIUM = 3; // MEDIUM: 3 units = 30ms
static constexpr int TIMESLICE_WEIGHT_LOW = 1;    // LOW: 1 unit = 10ms
static constexpr int TIMESLICE_TOTAL_WEIGHT = TIMESLICE_WEIGHT_HIGH + TIMESLICE_WEIGHT_MEDIUM + TIMESLICE_WEIGHT_LOW;  // 13

struct ProcessEntry {
    pid_t pid;              // 0 = free slot
    int priority;           // PRIO_HIGH/PRIO_MEDIUM/PRIO_LOW
    uint64_t last_heartbeat_ns;  // Last heartbeat for liveness check
    uint8_t reserved[52];   // padding to 64 bytes
};

struct SharedHeader {
    uint64_t magic;
    int32_t version;

    // Global cycle start time
    // Cycle = HIGH(90ms) + MEDIUM(30ms) + LOW(10ms) = 130ms, then repeats
    std::atomic<uint64_t> cycle_start_ns;

    // Active processes
    ProcessEntry processes[MAX_PID_SLOTS];

    pthread_mutex_t global_mutex;
};

class DeviceQuota;
class SharedQuotaManager {
   public:
    explicit SharedQuotaManager(DeviceQuota* local_quota);
    ~SharedQuotaManager();

    Status attach(const std::string& shm_name);
    Status detach();

    // Check if current process can send (based on time slice)
    bool canSend();

    // Heartbeat to keep process alive
    void heartbeat();

   private:
    Status registerProcess();
    Status unregisterProcess();
    void startBackgroundThread();
    void stopBackgroundThread();
    void backgroundThreadLoop();
    Status initializeHeader();
    Status initMutex(pthread_mutex_t* m);

    // Get current priority based on time in cycle
    int getCurrentPriority(uint64_t now, uint64_t cycle_start) const;

   private:
    std::string name_;
    SharedHeader* hdr_;
    int fd_;
    size_t size_;
    bool created_;
    DeviceQuota* local_quota_;
    pid_t my_pid_;

    // Background thread
    std::thread background_thread_;
    std::atomic<bool> background_running_;
};

}  // namespace tent
}  // namespace mooncake

#endif  // TENT_SHARED_QUOTA_H
