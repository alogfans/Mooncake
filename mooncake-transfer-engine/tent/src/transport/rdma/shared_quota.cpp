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

namespace mooncake {
namespace tent {
SharedQuotaManager::SharedQuotaManager(DeviceQuota* local_quota)
    : hdr_(nullptr),
      fd_(-1),
      size_(sizeof(SharedHeader)),
      created_(false),
      local_quota_(local_quota) {}

SharedQuotaManager::~SharedQuotaManager() { detach(); }

Status SharedQuotaManager::attach(const std::string& shm_name) {
    name_ = shm_name;

    // Open or create shared memory (mode 0666)
    fd_ = shm_open(name_.c_str(), O_RDWR | O_CREAT, 0666);
    if (fd_ < 0) {
        return Status::InternalError("shm_open failed: " +
                                     std::string(strerror(errno)));
    }

    // Ensure size
    if (ftruncate(fd_, static_cast<off_t>(size_)) != 0) {
        int e = errno;
        close(fd_);
        fd_ = -1;
        return Status::InternalError("ftruncate failed: " +
                                     std::string(strerror(e)));
    }

    // mmap
    void* ptr =
        mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (ptr == MAP_FAILED) {
        int e = errno;
        close(fd_);
        fd_ = -1;
        return Status::InternalError("mmap failed: " +
                                     std::string(strerror(e)));
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

    Status s = attachProcess();
    if (!s.ok()) return s;
    return Status::OK();
}

Status SharedQuotaManager::detach() {
    if (hdr_) {
        detachProcess();
        munmap(hdr_, size_);
        hdr_ = nullptr;
    }
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
    return Status::OK();
}

Status SharedQuotaManager::initializeHeader() {
    memset(hdr_, 0, size_);
    Status s = initMutex(&hdr_->global_mutex);
    if (!s.ok()) {
        return s;
    }
    hdr_->version = SHM_VERSION;
    hdr_->magic = SHM_MAGIC;
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

// attempt to acquire global lock; handle EOWNERDEAD
int SharedQuotaManager::lock() {
    if (!hdr_) return EINVAL;
    int rc = pthread_mutex_lock(&hdr_->global_mutex);
    if (rc == 0) return 0;
    if (rc == EOWNERDEAD) {
        // make consistent so others can continue
#if defined(PTHREAD_MUTEX_ROBUST)
        int rc2 = pthread_mutex_consistent(&hdr_->global_mutex);
        if (rc2 != 0) {
            return rc2;
        }
#endif
        // We hold the lock now — repair data if needed
        reclaimDeadPidsInternal();
        return 0;
    }
    return rc;
}

int SharedQuotaManager::unlock() {
    if (!hdr_) return EINVAL;
    return pthread_mutex_unlock(&hdr_->global_mutex);
}

void SharedQuotaManager::reclaimDeadPidsInternal() {
    if (!hdr_) return;
    // Assumes caller holds lock (but we also call from lock() on EOWNERDEAD)
    for (int i = 0; i < hdr_->num_devices; ++i) {
        // skip unused slots (dev_name empty)
        if (hdr_->devices[i].dev_name[0] == '\0') continue;
        SharedDeviceEntry& dev = hdr_->devices[i];
        for (int s = 0; s < MAX_PID_SLOTS; ++s) {
            pid_t p = dev.pid_usages[s].pid;
            if (p == 0) continue;
            if (!isPidAlive(p)) {
                // zero out slot — we'll recompute active_bytes in diffusion
                dev.pid_usages[s].pid = 0;
                dev.pid_usages[s].high_prio_bytes = 0;
                dev.pid_usages[s].medium_prio_bytes = 0;
                dev.pid_usages[s].low_prio_bytes = 0;
            }
        }
    }
}

bool SharedQuotaManager::isPidAlive(pid_t pid) {
    if (pid <= 0) return false;
    int r = kill(pid, 0);
    if (r == 0) return true;
    if (errno == ESRCH) return false;
    return true;  // other errors (EPERM) -> treat as alive
}

int SharedQuotaManager::findDeviceIdByNameLocked(const std::string& dev_name) {
    if (!hdr_) return -1;
    for (int i = 0; i < hdr_->num_devices; ++i) {
        if (hdr_->devices[i].dev_name[0] == '\0') continue;
        if (strncmp(hdr_->devices[i].dev_name, dev_name.c_str(),
                    sizeof(hdr_->devices[i].dev_name)) == 0)
            return i;
    }
    return -1;
}

PidUsage* SharedQuotaManager::findOrCreatePidSlotLocked(int dev_id, pid_t pid) {
    if (!hdr_) return nullptr;
    if (dev_id < 0 || dev_id >= hdr_->num_devices) return nullptr;
    SharedDeviceEntry& dev = hdr_->devices[dev_id];
    PidUsage* empty = nullptr;
    for (int s = 0; s < MAX_PID_SLOTS; ++s) {
        if (dev.pid_usages[s].pid == pid) return &dev.pid_usages[s];
        if (dev.pid_usages[s].pid == 0 && empty == nullptr)
            empty = &dev.pid_usages[s];
    }
    if (empty) {
        empty->pid = pid;
        empty->high_prio_bytes = 0;
        empty->medium_prio_bytes = 0;
        empty->low_prio_bytes = 0;
    }
    return empty;
}

PidUsage* SharedQuotaManager::findPidSlotLocked(int dev_id, pid_t pid) {
    if (!hdr_) return nullptr;
    if (dev_id < 0 || dev_id >= hdr_->num_devices) return nullptr;
    SharedDeviceEntry& dev = hdr_->devices[dev_id];
    for (int s = 0; s < MAX_PID_SLOTS; ++s) {
        if (dev.pid_usages[s].pid == pid) return &dev.pid_usages[s];
    }
    return nullptr;
}

Status SharedQuotaManager::attachProcess() {
    if (!hdr_) return Status::InvalidArgument("not attached");

    int rc = lock();
    if (rc != 0) {
        return Status::InternalError("failed to lock shared mutex: " +
                                     std::string(strerror(rc)));
    }

    auto topo = local_quota_->getTopology();
    for (size_t i = 0; i < topo->getNicCount(); ++i) {
        if (topo->getNicType(i) != Topology::NIC_RDMA) continue;
        auto dev_name = topo->getNicName(i);
        if (findDeviceIdByNameLocked(dev_name) >= 0) continue;
        int empty_idx = -1;
        for (int j = 0; j < MAX_DEVICES; ++j) {
            if (hdr_->devices[j].dev_name[0] == '\0') {
                empty_idx = j;
                break;
            }
        }
        if (empty_idx < 0) continue;
        strncpy(hdr_->devices[empty_idx].dev_name, dev_name.c_str(), 56);
        hdr_->devices[empty_idx].active_bytes = 0;
    }

    int count = 0;
    for (int i = 0; i < MAX_DEVICES; ++i)
        if (hdr_->devices[i].dev_name[0] != '\0') ++count;
    hdr_->num_devices = count;

    unlock();
    return Status::OK();
}

Status SharedQuotaManager::detachProcess() { return Status::OK(); }

Status SharedQuotaManager::diffusion() {
    if (!hdr_) return Status::InvalidArgument("not attached");
    pid_t pid = getpid();
    int rc = lock();
    if (rc != 0)
        return Status::InternalError("lock failed: " +
                                     std::string(strerror(rc)));
    for (int d = 0; d < hdr_->num_devices; ++d) {
        std::string dev_name = hdr_->devices[d].dev_name;
        auto dev_id = local_quota_->getTopology()->getNicId(dev_name);
        if (dev_name.empty() || dev_id < 0) continue;
        PidUsage* slot = findOrCreatePidSlotLocked(dev_id, pid);
        if (!slot) {
            unlock();
            return Status::InternalError("no free pid slot for device");
        }
        // Report per-priority active bytes
        slot->high_prio_bytes =
            local_quota_->getActiveBytesByPriority(dev_id, PRIO_HIGH);
        slot->medium_prio_bytes =
            local_quota_->getActiveBytesByPriority(dev_id, PRIO_MEDIUM);
        slot->low_prio_bytes =
            local_quota_->getActiveBytesByPriority(dev_id, PRIO_LOW);

        uint64_t high_sum = 0, med_sum = 0, low_sum = 0, total_sum = 0;
        for (int s = 0; s < MAX_PID_SLOTS; ++s) {
            pid_t p = hdr_->devices[d].pid_usages[s].pid;
            if (p == 0) continue;
            high_sum += hdr_->devices[d].pid_usages[s].high_prio_bytes;
            med_sum += hdr_->devices[d].pid_usages[s].medium_prio_bytes;
            low_sum += hdr_->devices[d].pid_usages[s].low_prio_bytes;
        }
        total_sum = high_sum + med_sum + low_sum;

        hdr_->devices[d].active_bytes = total_sum;
        hdr_->devices[d].high_prio_bytes = high_sum;
        hdr_->devices[d].medium_prio_bytes = med_sum;
        hdr_->devices[d].low_prio_bytes = low_sum;
        // For compatibility: set diffusion_active_bytes (sum of other
        // processes)
        uint64_t my_bytes = slot->high_prio_bytes + slot->medium_prio_bytes +
                            slot->low_prio_bytes;
        uint64_t diffusion_active_bytes =
            total_sum < my_bytes ? 0 : total_sum - my_bytes;
        local_quota_->setDiffusionActiveBytes(dev_id, diffusion_active_bytes);

        // Update timeslice state
        updateTimeslice(d);
    }

    unlock();
    return Status::OK();
}

uint64_t SharedQuotaManager::getHighPrioLoad(int dev_id) const {
    if (!hdr_ || dev_id < 0 || dev_id >= hdr_->num_devices) return 0;
    return hdr_->devices[dev_id].high_prio_bytes;
}

uint64_t SharedQuotaManager::getMediumPrioLoad(int dev_id) const {
    if (!hdr_ || dev_id < 0 || dev_id >= hdr_->num_devices) return 0;
    return hdr_->devices[dev_id].medium_prio_bytes;
}

uint64_t SharedQuotaManager::getLowPrioLoad(int dev_id) const {
    if (!hdr_ || dev_id < 0 || dev_id >= hdr_->num_devices) return 0;
    return hdr_->devices[dev_id].low_prio_bytes;
}

void SharedQuotaManager::updateTimeslice(int dev_idx) {
    if (!hdr_) return;
    uint64_t now = getCurrentTimeInNano();
    auto& ts = hdr_->devices[dev_idx].timeslice;

    // Initialize on first use
    uint64_t current_end = ts.slice_end_ns.load(std::memory_order_relaxed);
    if (current_end == 0) {
        ts.current_prio.store(PRIO_HIGH, std::memory_order_relaxed);
        ts.slice_end_ns.store(now + TIMESLICE_BASE_NS * QUOTA_WEIGHT[PRIO_HIGH],
                              std::memory_order_relaxed);
        return;
    }

    // Check if current timeslice has ended
    if (now >= current_end) {
        // Advance to next priority with pending work
        uint64_t next_prio = PRIO_HIGH;
        // Simple round-robin through priorities, skipping empty ones
        for (int p = 1; p < NUM_PRIORITIES; ++p) {
            int prio = (ts.current_prio.load(std::memory_order_relaxed) + p) %
                       NUM_PRIORITIES;
            if (hdr_->devices[dev_idx].high_prio_bytes > 0 &&
                prio == PRIO_HIGH) {
                next_prio = prio;
                break;
            }
            if (hdr_->devices[dev_idx].medium_prio_bytes > 0 &&
                prio == PRIO_MEDIUM) {
                next_prio = prio;
                break;
            }
            if (hdr_->devices[dev_idx].low_prio_bytes > 0 && prio == PRIO_LOW) {
                next_prio = prio;
                break;
            }
        }
        ts.current_prio.store(next_prio, std::memory_order_relaxed);
        ts.slice_end_ns.store(now + TIMESLICE_BASE_NS * QUOTA_WEIGHT[next_prio],
                              std::memory_order_relaxed);
    }
}

bool SharedQuotaManager::canSend(int dev_id, int priority) const {
    if (!hdr_ || dev_id < 0 || dev_id >= hdr_->num_devices)
        return true;  // Allow if no quota
    if (priority < 0 || priority >= NUM_PRIORITIES) return true;

    const auto& ts = hdr_->devices[dev_id].timeslice;
    uint64_t current_prio = ts.current_prio.load(std::memory_order_relaxed);
    uint64_t now = getCurrentTimeInNano();

    // Check if we're still in current timeslice
    if (now < ts.slice_end_ns.load(std::memory_order_relaxed))
        return current_prio == priority;

    // Timeslice ended, try to advance (non-blocking check)
    return false;  // Caller should call updateTimeslice
}

}  // namespace tent
}  // namespace mooncake
