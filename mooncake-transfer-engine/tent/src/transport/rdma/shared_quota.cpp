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

#include "tent/transport/rdma/shared_quota.h"
#include "tent/common/utils/os.h"
#include "tent/common/types.h"

#include <glog/logging.h>
#include <unistd.h>

namespace mooncake {
namespace tent {

SharedQuotaManager::SharedQuotaManager(DeviceQuota* local_quota)
    : hdr_(nullptr),
      fd_(-1),
      size_(sizeof(SharedHeader)),
      created_(false),
      local_quota_(local_quota),
      my_pid_(getpid()),
      background_running_(false) {}

SharedQuotaManager::~SharedQuotaManager() { detach(); }

Status SharedQuotaManager::initializeHeader() {
    std::memset(hdr_, 0, size_);

    Status s = initMutex(&hdr_->global_mutex);
    if (!s.ok()) return s;

    hdr_->version = SHM_VERSION;
    hdr_->magic = SHM_MAGIC;

    // Initialize cycle start to current time
    hdr_->cycle_start_ns.store(getCurrentTimeInNano(), std::memory_order_release);

    return Status::OK();
}

Status SharedQuotaManager::initMutex(pthread_mutex_t* m) {
    pthread_mutexattr_t attr;
    if (pthread_mutexattr_init(&attr) != 0) {
        return Status::InternalError("pthread_mutexattr_init failed");
    }
    if (pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED) != 0) {
        pthread_mutexattr_destroy(&attr);
        return Status::InternalError("pthread_mutexattr_setpshared failed");
    }
#if defined(PTHREAD_MUTEX_ROBUST)
    if (pthread_mutexattr_setrobust(&attr, PTHREAD_MUTEX_ROBUST) != 0) {
        pthread_mutexattr_destroy(&attr);
        return Status::InternalError("pthread_mutexattr_setrobust failed");
    }
#endif
    if (pthread_mutex_init(m, &attr) != 0) {
        pthread_mutexattr_destroy(&attr);
        return Status::InternalError("pthread_mutex_init failed");
    }
    pthread_mutexattr_destroy(&attr);
    return Status::OK();
}

int SharedQuotaManager::getCurrentPriority(uint64_t now, uint64_t cycle_start) const {
    // Calculate position in current cycle
    uint64_t pos_ns = (now - cycle_start) % (TIMESLICE_TOTAL_WEIGHT * TIMESLICE_UNIT_MS * 1000000ull);

    // Determine which priority's time slot we're in
    // HIGH: [0, 9*10ms), MEDIUM: [9*10ms, 12*10ms), LOW: [12*10ms, 13*10ms)
    uint64_t high_end = TIMESLICE_WEIGHT_HIGH * TIMESLICE_UNIT_MS * 1000000ull;
    if (pos_ns < high_end) return static_cast<int>(PRIO_HIGH);

    uint64_t medium_end = (TIMESLICE_WEIGHT_HIGH + TIMESLICE_WEIGHT_MEDIUM) * TIMESLICE_UNIT_MS * 1000000ull;
    if (pos_ns < medium_end) return static_cast<int>(PRIO_MEDIUM);

    return static_cast<int>(PRIO_LOW);
}

Status SharedQuotaManager::attach(const std::string& shm_name) {
    name_ = shm_name;

    fd_ = shm_open(name_.c_str(), O_RDWR | O_CREAT, 0666);
    if (fd_ < 0) {
        return Status::InternalError("shm_open failed: " +
                                     std::string(std::strerror(errno)));
    }

    if (ftruncate(fd_, static_cast<off_t>(size_)) != 0) {
        int e = errno;
        close(fd_);
        fd_ = -1;
        return Status::InternalError("ftruncate failed: " + std::string(std::strerror(e)));
    }

    void* ptr = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (ptr == MAP_FAILED) {
        int e = errno;
        close(fd_);
        fd_ = -1;
        return Status::InternalError("mmap failed: " + std::string(std::strerror(e)));
    }

    hdr_ = reinterpret_cast<SharedHeader*>(ptr);

    if (hdr_->magic != SHM_MAGIC || hdr_->version != SHM_VERSION) {
        created_ = true;
        Status s = initializeHeader();
        if (!s.ok()) {
            munmap(ptr, size_);
            close(fd_);
            hdr_ = nullptr;
            fd_ = -1;
            return s;
        }
    } else {
        created_ = false;
    }

    Status s = registerProcess();
    if (!s.ok()) {
        detach();
        return s;
    }

    startBackgroundThread();

    return Status::OK();
}

Status SharedQuotaManager::detach() {
    stopBackgroundThread();

    if (hdr_) {
        unregisterProcess();
        munmap(hdr_, size_);
        hdr_ = nullptr;
    }

    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }

    return Status::OK();
}

Status SharedQuotaManager::registerProcess() {
    if (!hdr_) return Status::InvalidArgument("not attached");

    int my_priority = static_cast<int>(PRIO_HIGH);  // TODO: from config

    int rc = pthread_mutex_lock(&hdr_->global_mutex);
    if (rc != 0) {
        return Status::InternalError("mutex lock failed: " + std::string(std::strerror(rc)));
    }

    // Find empty slot or existing slot
    int empty_slot = -1;
    for (int i = 0; i < MAX_PID_SLOTS; ++i) {
        if (hdr_->processes[i].pid == my_pid_) {
            hdr_->processes[i].last_heartbeat_ns = getCurrentTimeInNano();
            pthread_mutex_unlock(&hdr_->global_mutex);
            return Status::OK();
        }
        if (hdr_->processes[i].pid == 0 && empty_slot < 0) {
            empty_slot = i;
        }
    }

    if (empty_slot >= 0) {
        hdr_->processes[empty_slot].pid = my_pid_;
        hdr_->processes[empty_slot].priority = my_priority;
        hdr_->processes[empty_slot].last_heartbeat_ns = getCurrentTimeInNano();
        LOG(INFO) << "Registered process " << my_pid_ << " with priority " << my_priority;
    } else {
        pthread_mutex_unlock(&hdr_->global_mutex);
        return Status::InternalError("no free process slot");
    }

    pthread_mutex_unlock(&hdr_->global_mutex);
    return Status::OK();
}

Status SharedQuotaManager::unregisterProcess() {
    if (!hdr_) return Status::InvalidArgument("not attached");

    int rc = pthread_mutex_lock(&hdr_->global_mutex);
    if (rc != 0) return Status::OK();

    for (int i = 0; i < MAX_PID_SLOTS; ++i) {
        if (hdr_->processes[i].pid == my_pid_) {
            hdr_->processes[i].pid = 0;
            LOG(INFO) << "Unregistered process " << my_pid_;
            break;
        }
    }

    pthread_mutex_unlock(&hdr_->global_mutex);
    return Status::OK();
}

bool SharedQuotaManager::canSend() {
    if (!hdr_) return true;

    uint64_t now = getCurrentTimeInNano();

    // Get my priority
    int my_priority = static_cast<int>(PRIO_HIGH);  // TODO: from config
    for (int i = 0; i < MAX_PID_SLOTS; ++i) {
        if (hdr_->processes[i].pid == my_pid_) {
            my_priority = hdr_->processes[i].priority;
            break;
        }
    }

    // Get current priority based on time in cycle
    uint64_t cycle_start = hdr_->cycle_start_ns.load(std::memory_order_acquire);
    int current_prio = getCurrentPriority(now, cycle_start);

    // Only allow if priority matches
    // Same priority processes can share, no explicit ownership needed
    return (current_prio == my_priority);
}

void SharedQuotaManager::heartbeat() {
    if (!hdr_) return;

    uint64_t now = getCurrentTimeInNano();

    int rc = pthread_mutex_lock(&hdr_->global_mutex);
    if (rc != 0) return;

    for (int i = 0; i < MAX_PID_SLOTS; ++i) {
        if (hdr_->processes[i].pid == my_pid_) {
            hdr_->processes[i].last_heartbeat_ns = now;
            break;
        }
    }

    pthread_mutex_unlock(&hdr_->global_mutex);
}

void SharedQuotaManager::startBackgroundThread() {
    if (background_running_.exchange(true)) return;

    background_thread_ = std::thread([this]() {
        backgroundThreadLoop();
    });
}

void SharedQuotaManager::stopBackgroundThread() {
    if (!background_running_.exchange(false)) return;

    if (background_thread_.joinable()) {
        background_thread_.join();
    }
}

void SharedQuotaManager::backgroundThreadLoop() {
    const uint64_t SLEEP_INTERVAL_US = 1000;  // 1ms

    while (background_running_.load(std::memory_order_relaxed)) {
        // Background thread doesn't need to do anything actively
        // Time is calculated on-the-fly from cycle_start_ns
        // Just sleep to keep thread alive
        usleep(SLEEP_INTERVAL_US);
    }
}

}  // namespace tent
}  // namespace mooncake
