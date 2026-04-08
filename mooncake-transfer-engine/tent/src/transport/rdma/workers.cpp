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

#include "tent/transport/rdma/workers.h"
#include "tent/transport/rdma/shared_quota.h"

#include <sys/epoll.h>

#include <cassert>

#include "tent/transport/rdma/endpoint_store.h"
#include "tent/common/utils/ip.h"
#include "tent/common/utils/string_builder.h"
#include "tent/common/utils/os.h"
#include "tent/common/utils/random.h"

namespace mooncake {
namespace tent {
thread_local int tl_wid = -1;

// Error classification for CQ completions
enum class ErrorClass { FLUSH, OTHER };

static ErrorClass classifyError(int ibv_wc_status) {
    if (ibv_wc_status == IBV_WC_WR_FLUSH_ERR) return ErrorClass::FLUSH;
    return ErrorClass::OTHER;
}

Workers::Workers(RdmaTransport* transport)
    : transport_(transport), num_workers_(0), running_(false) {
    device_quota_ = std::make_unique<DeviceQuota>();
    device_quota_->loadTopology(transport_->local_topology_);
    auto& conf = transport_->conf_;

    auto shared_quota_shm_path =
        conf->get("transports/rdma/shared_quota_shm_path", "");
    if (!shared_quota_shm_path.empty()) {
        auto status = device_quota_->enableSharedQuota(shared_quota_shm_path);
        if (!status.ok()) {
            LOG(WARNING) << "Failed to enable shared quota: "
                         << status.ToString();
        }
        // Configure timeslice duration for QoS scheduling
        int timeslice_unit_ms =
            conf->get("transports/rdma/timeslice_unit_ms", 2);
        device_quota_->getSharedQuota()->setTimesliceUnitMs(timeslice_unit_ms);
    }
    auto enable_quota = conf->get("transports/rdma/enable_quota", true);
    device_quota_->setEnableQuota(enable_quota);

    // Configure priority from config (for shared quota QoS)
    auto priority_str =
        conf->get("transports/rdma/priority", std::string("high"));
    int priority = 0;  // Default: PRIO_HIGH
    if (priority_str == "medium")
        priority = 1;
    else if (priority_str == "low")
        priority = 2;
    device_quota_->setPriority(priority);

    // Configure local priority timeslice (for intra-process QoS)
    // Uses rdtscp for high-precision timing
    timeslice_us_ = conf->get("transports/rdma/local_timeslice_us", 100);

    // Set slice calculation parameters (must match rdma_transport.cpp)
    device_quota_->setSliceParams(transport_->params_->workers.block_size);

    // Configure EWMA scheduling parameters
    DeviceQuota::SchedulingParams params;
    params.ewma_alpha = conf->get("transports/rdma/ewma_alpha", 0.01);
    params.enable_ewma_learning =
        conf->get("transports/rdma/enable_ewma_learning", true);

    // Configure device priority
    params.enable_device_priority =
        conf->get("transports/rdma/device_priority", false);
    if (params.enable_device_priority) {
        device_quota_->fillDevicePriorities();
    }

    // Configure rank weights from config (format: "w0,w1,w2")
    std::string weights_str =
        conf->get("transports/rdma/rank_weights", std::string("10.0,2.0,0.2"));
    if (!weights_str.empty()) {
        size_t pos = 0;
        for (size_t i = 0;
             i < Topology::DevicePriorityRanks && pos < weights_str.length();
             ++i) {
            size_t end = weights_str.find(',', pos);
            if (end == std::string::npos) end = weights_str.length();
            std::string val = weights_str.substr(pos, end - pos);
            params.rank_weights[i] = std::stod(val);
            pos = end + 1;
        }
    }

    device_quota_->setSchedulingParams(params);

    // Configure priority weights for QoS scheduling (format: "w0,w1,w2")
    // Default: High:Medium:Low = 9:3:1
    std::string prio_weights_str =
        conf->get("transports/rdma/priority_weights", std::string("9,3,1"));
    kTotalWeight = 0;
    if (!prio_weights_str.empty()) {
        size_t pos = 0;
        for (size_t i = 0;
             i < kNumPriorityLevels && pos < prio_weights_str.length(); ++i) {
            size_t end = prio_weights_str.find(',', pos);
            if (end == std::string::npos) end = prio_weights_str.length();
            std::string val = prio_weights_str.substr(pos, end - pos);
            kPriorityWeight[i] = std::stoi(val);
            kTotalWeight += kPriorityWeight[i];
            pos = end + 1;
        }
    } else {
        // Fallback to default values
        kPriorityWeight[0] = 9;
        kPriorityWeight[1] = 3;
        kPriorityWeight[2] = 1;
        kTotalWeight = 13;
    }
}

Workers::~Workers() {
    if (running_) stop();
}

Status Workers::start() {
    const static uint64_t kDefaultMaxTimeoutNs = 10000000000ull;
    if (!running_) {
        running_ = true;
        monitor_ = std::thread([this] { monitorThread(); });
        num_workers_ = transport_->params_->workers.num_workers;
        slice_timeout_ns_ = transport_->conf_->get(
            "transports/rdma/max_timeout_ns", kDefaultMaxTimeoutNs);
        worker_context_ = new WorkerContext[num_workers_];
        for (size_t id = 0; id < num_workers_; ++id) {
            worker_context_[id].thread =
                std::thread([this, id] { workerThread(id); });
        }
    }
    return Status::OK();
}

Status Workers::stop() {
    if (!running_) return Status::OK();
    running_ = false;
    for (size_t id = 0; id < num_workers_; ++id) {
        auto& worker = worker_context_[id];
        {
            std::lock_guard<std::mutex> lock(worker.mutex);
            worker.cv.notify_all();
        }
        worker.thread.join();
    }
    monitor_.join();
    delete[] worker_context_;
    worker_context_ = nullptr;
    return Status::OK();
}

Status Workers::submit(RdmaSliceList& slice_list, int worker_id) {
    if (worker_id < 0 || worker_id >= (int)num_workers_) {
        // If caller didn't specify the worker, find the least loaded one
        long min_inflight = INT64_MAX;
        int start_id = SimpleRandom::Get().next(num_workers_);
        for (size_t i = start_id; i < start_id + num_workers_; ++i) {
            auto current =
                worker_context_[i % num_workers_].inflight_slices.load(
                    std::memory_order_relaxed);
            if (current < min_inflight) {
                worker_id = i % num_workers_;
                min_inflight = current;
            }
        }
    }
    auto& worker = worker_context_[worker_id];

    // Get priority from first slice (all slices in list have same priority)
    int priority = 0;  // default high
    if (slice_list.first && slice_list.first->task) {
        priority =
            std::clamp(slice_list.first->priority, 0, kNumPriorityLevels - 1);
    }

    worker.queues[priority].push(slice_list);
    if (!worker.inflight_slices.fetch_add(slice_list.num_slices)) {
        std::lock_guard<std::mutex> lock(worker.mutex);
        if (worker.in_suspend) worker.cv.notify_all();
    }
    return Status::OK();
}

Status Workers::submit(RdmaSlice* slice) {
    RdmaSliceList slice_list;
    slice_list.first = slice;
    slice_list.num_slices = 1;
    return submit(slice_list);
}

Status Workers::cancel(RdmaSliceList& slice_list) {
    return Status::NotImplemented("cancel not implemented" LOC_MARK);
}

std::shared_ptr<RdmaEndPoint> Workers::getEndpoint(Workers::PostPath path) {
    std::string target_seg_name, target_dev_name;
    std::string rpc_server_addr;
    RouteHint hint;
    auto target_id = path.remote_segment_id;
    auto device_id = path.remote_device_id;
    auto& segment_manager = transport_->metadata_->segmentManager();
    if (target_id == LOCAL_SEGMENT_ID) {
        hint.segment = segment_manager.getLocal().get();
    } else {
        segment_manager.getRemoteCached(hint.segment, target_id);
    }
    if (hint.segment->type != SegmentType::Memory) return nullptr;
    hint.topo = &std::get<MemorySegmentDesc>(hint.segment->detail).topology;
    if (target_id != LOCAL_SEGMENT_ID) {
        rpc_server_addr = hint.segment->getMemory().rpc_server_addr;
    }
    target_seg_name = hint.segment->name;
    target_dev_name = hint.topo->getNicName(device_id);
    if (target_seg_name.empty() || target_dev_name.empty()) {
        LOG(ERROR) << "Empty target segment or device name";
        return nullptr;
    }
    auto context = transport_->context_set_[path.local_device_id].get();
    if (context->status() != RdmaContext::DEVICE_ENABLED) {
        // LOG(WARNING) << "Context " << context->name() << " is not serving";
        return nullptr;  // experimental: force to fail this slice and mark this
                         // connection unavailable
    }
    std::shared_ptr<RdmaEndPoint> endpoint;
    auto peer_name = MakeNicPath(target_seg_name, target_dev_name);
    endpoint = context->endpointStore()->getOrInsert(peer_name);
    if (endpoint && endpoint->status() == RdmaEndPoint::EP_RESET) {
        context->endpointStore()->remove(endpoint.get());
        endpoint = context->endpointStore()->getOrInsert(peer_name);
    }
    if (!endpoint) {
        LOG(ERROR) << "Cannot allocate endpoint " << peer_name;
        return nullptr;
    }
    if (endpoint->status() != RdmaEndPoint::EP_READY) {
        auto status = endpoint->connect(target_seg_name, target_dev_name,
                                        rpc_server_addr);
        if (!status.ok()) {
            thread_local uint64_t tl_last_output_ts = 0;
            uint64_t current_ts = getCurrentTimeInNano();
            if (current_ts - tl_last_output_ts > 10000000000ull) {
                tl_last_output_ts = current_ts;
                LOG(ERROR) << "Unable to connect endpoint " << peer_name << ": "
                           << status.ToString();
            }
            return nullptr;
        }
    }
    return endpoint;
}

void Workers::disableEndpoint(RdmaSlice* slice, int ibv_wc_status) {
    SegmentDesc* desc = nullptr;
    auto& segment_manager = transport_->metadata_->segmentManager();
    auto target_id = slice->task->request.target_id;
    if (target_id == LOCAL_SEGMENT_ID) {
        desc = segment_manager.getLocal().get();
    } else {
        auto status = segment_manager.getRemoteCached(desc, target_id);
        if (!status.ok()) return;
    }
    if (desc) {
        auto& worker = worker_context_[tl_wid];
        auto& rail = worker.rails[desc->machine_id];
        rail.markFailed(slice->source_dev_id, slice->target_dev_id,
                        ibv_wc_status);
        rail.markDegraded(slice->source_dev_id, slice->target_dev_id);
    }
    // Remove endpoint from active store. It goes to the store's waiting_list
    // which holds its shared_ptr alive until all inflight WRs produce CQEs.
    // Never reset() here — that would kill other in-flight slices.
    // reclaim() will clean up once inflight_slices_ reaches 0.
    if (slice->ep_weak_ptr) {
        auto ep = slice->ep_weak_ptr;
        ep->context().endpointStore()->remove(ep);
    }
}

void Workers::asyncPostSend() {
    auto& worker = worker_context_[tl_wid];
    std::vector<RdmaSliceList> result;

    auto shared_quota =
        device_quota_ ? device_quota_->getSharedQuota() : nullptr;

    // Multi-process: check if this process can send (global slot)
    // Single-process: always true, priority handled locally below
    bool can_send = shared_quota ? shared_quota->canSend() : true;

    if (can_send) {
        // Local time-based priority scheduling (using rdtscp):
        // Slot 0 (0-100us): HIGH requests only
        // Slot 1 (100-200us): MEDIUM + HIGH requests
        // Slot 2 (200-300us): ALL requests (LOW + MEDIUM + HIGH)
        // Then repeat (finer granularity than shared quota)
        // This ensures HIGH gets preference while LOW doesn't starve.
        int current_slot = getCurrentPrioritySlot();

        // Try to dispatch from allowed priorities in current time slot
        // Request can run if its priority <= current_slot
        // HIGH(0): can run in all slots (0,1,2)
        // MEDIUM(1): can run in slots 1,2
        // LOW(2): can run only in slot 2
        for (int prio = 0; prio < kNumPriorityLevels; ++prio) {
            if (prio > current_slot) continue;  // Not allowed in this slot
            worker.queues[prio].pop(result);
            if (!result.empty()) break;  // Dispatched one batch, done
        }
    }

    for (auto& slice_list : result) {
        if (slice_list.num_slices == 0) continue;
        auto slice = slice_list.first;
        for (int id = 0; id < slice_list.num_slices; ++id) {
            auto status = generatePostPath(slice);
            if (!status.ok()) {
                LOG(ERROR) << "Failed to generate post path for slice " << slice
                           << ": " << status.ToString();
                updateSliceStatus(slice, FAILED);
            } else {
                PostPath path{
                    .local_device_id = slice->source_dev_id,
                    .remote_segment_id = slice->task->request.target_id,
                    .remote_device_id = slice->target_dev_id};
                worker.requests[path].push_back(slice);
            }
            slice = slice->next;
        }
    }

    for (auto& entry : worker.requests) {
        auto& path = entry.first;
        auto& slices = entry.second;
        if (slices.empty()) continue;
        auto endpoint = getEndpoint(path);
        if (!endpoint) {
            std::vector<RdmaSlice*> clone;
            slices.swap(clone);
            for (auto slice : clone) {
                slice->retry_count++;
                if (slice->retry_count >=
                    transport_->params_->workers.max_retry_count) {
                    LOG(WARNING)
                        << "Slice " << slice << " failed: retry count exceeded";
                    disableEndpoint(slice, IBV_WC_REM_ACCESS_ERR);
                    updateSliceStatus(slice, FAILED);
                } else {
                    submit(slice);
                }
            }
            continue;
        }

        int num_submitted = endpoint->submitSlices(slices, tl_wid);
        for (int id = 0; id < num_submitted; ++id) {
            auto slice = slices[id];
            if (slice->failed) {
                slice->retry_count++;
                if (slice->retry_count >=
                    transport_->params_->workers.max_retry_count) {
                    LOG(WARNING)
                        << "Slice " << slice << " failed: retry count exceeded";
                    disableEndpoint(slice, IBV_WC_RETRY_EXC_ERR);
                    updateSliceStatus(slice, FAILED);
                } else {
                    submit(slice);
                }
            } else {
                slice->submit_ts = getCurrentTimeInNano();
            }
        }

        if (num_submitted) {
            worker.inflight_slice_set.insert(slices.begin(),
                                             slices.begin() + num_submitted);
            slices.erase(slices.begin(), slices.begin() + num_submitted);
        }
    }
}

void Workers::asyncPollCq() {
    auto& worker = worker_context_[tl_wid];
    const static size_t kPollCount = 64;
    int num_contexts = (int)transport_->context_set_.size();
    int num_cq_list = transport_->params_->device.num_cq_list;
    int num_slices = 0;

    uint64_t current_ts = getCurrentTimeInNano();
    std::vector<RdmaSlice*> slice_to_remove;
    for (auto& slice : worker.inflight_slice_set) {
        if (slice->word != PENDING) continue;
        if (current_ts - slice->enqueue_ts > slice_timeout_ns_) {
            auto ep = slice->ep_weak_ptr;
            LOG(WARNING) << "Slice " << slice
                         << " failed: transfer timeout (software)";
            auto num_slices = ep->acknowledge(slice, TIMEOUT);
            disableEndpoint(slice, IBV_WC_RESP_TIMEOUT_ERR);
            worker.inflight_slices.fetch_sub(num_slices);
            slice_to_remove.push_back(slice);
        }
    }
    for (auto& slice : slice_to_remove) worker.inflight_slice_set.erase(slice);

    for (int index = 0; index < num_contexts; index++) {
        auto& context = transport_->context_set_[index];
        auto cq = context->cq(tl_wid % num_cq_list);
        ibv_wc wc[kPollCount];
        int nr_poll = cq->poll(kPollCount, wc);
        if (nr_poll < 0) continue;
        auto poll_ts = getCurrentTimeInNano();
        for (int i = 0; i < nr_poll; ++i) {
            auto slice = (RdmaSlice*)wc[i].wr_id;
            worker.inflight_slice_set.erase(slice);
            auto ep = slice->ep_weak_ptr;
            double enqueue_lat =
                (slice->submit_ts - slice->enqueue_ts) / 1000.0;
            double inflight_lat = (poll_ts - slice->submit_ts) / 1000.0;
            // Use inflight_lat (pure transfer time) to exclude queue wait time
            // This reflects actual device/NUMA performance, not scheduling
            // delay
            double transfer_lat_sec = inflight_lat / 1e6;
            if (slice->retry_count == 0) {
                device_quota_->release(slice->source_dev_id, slice->length,
                                       transfer_lat_sec);
            }
            if (slice->word != PENDING) continue;
            if (wc[i].status != IBV_WC_SUCCESS) {
                auto error_class = classifyError(wc[i].status);
                if (error_class != ErrorClass::FLUSH) {
                    LOG(INFO) << "Detected error WQE for slice " << slice
                              << " (opcode: " << slice->task->request.opcode
                              << ", source_addr: " << (void*)slice->source_addr
                              << ", dest_addr: " << (void*)slice->target_addr
                              << ", length: " << slice->length
                              << ", local_nic: " << context->name()
                              << "): " << ibv_wc_status_str(wc[i].status);
                }

                bool should_fail = false;
                if (error_class == ErrorClass::FLUSH) {
                    // Flush errors from reconnection don't consume retry
                    // budget. Force path re-selection by clearing device
                    // assignments.
                    num_slices += ep->acknowledge(slice, PENDING);
                    disableEndpoint(slice, wc[i].status);
                    slice->source_dev_id = -1;
                    slice->target_dev_id = -1;
                    slice->ep_weak_ptr = nullptr;
                } else {
                    // OTHER errors: consume retry budget
                    slice->retry_count++;
                    if (slice->retry_count >=
                        transport_->params_->workers.max_retry_count) {
                        LOG(WARNING) << "Slice " << slice
                                     << " failed: retry count exceeded";
                        should_fail = true;
                    } else {
                        num_slices += ep->acknowledge(slice, PENDING);
                        disableEndpoint(slice, wc[i].status);
                    }
                }

                if (should_fail) {
                    num_slices += ep->acknowledge(slice, FAILED);
                    disableEndpoint(slice, wc[i].status);
                } else {
                    submit(slice);
                }
            } else {
                num_slices += ep->acknowledge(slice, COMPLETED);
                worker.perf.inflight_lat.add(inflight_lat);
                worker.perf.enqueue_lat.add(enqueue_lat);
            }
        }
    }
    if (num_slices) {
        worker.inflight_slices.fetch_sub(num_slices);
    }
    // Periodically reclaim endpoints from store waiting lists.
    // Endpoints in waiting_list have been removed from active use but are kept
    // alive until inflight_slices_ == 0 (all CQEs received). reclaim() safely
    // destroys them once no more completions are pending.
    static constexpr uint64_t kReclaimIntervalNs = 5000000000ull;  // 5 seconds
    uint64_t reclaim_ts = getCurrentTimeInNano();
    if (reclaim_ts - worker.last_reclaim_ts > kReclaimIntervalNs) {
        worker.last_reclaim_ts = reclaim_ts;
        for (auto& context : transport_->context_set_) {
            context->endpointStore()->reclaim();
        }
    }
}

void Workers::showLatencyInfo() {
    auto& worker = worker_context_[tl_wid];
    LOG(INFO) << "[W" << tl_wid << "] enqueue count "
              << worker.perf.enqueue_lat.count() << " avg "
              << worker.perf.enqueue_lat.avg() << " p99 "
              << worker.perf.enqueue_lat.p99() << " p999 "
              << worker.perf.enqueue_lat.p999();
    LOG(INFO) << "[W" << tl_wid << "] submit count "
              << worker.perf.inflight_lat.count() << " avg "
              << worker.perf.inflight_lat.avg() << " p99 "
              << worker.perf.inflight_lat.p99() << " p999 "
              << worker.perf.inflight_lat.p999();
    worker.perf.enqueue_lat.clear();
    worker.perf.inflight_lat.clear();
}

void Workers::workerThread(int thread_id) {
    bindToSocket(thread_id % numa_num_configured_nodes());
    tl_wid = thread_id;
    auto& worker = worker_context_[thread_id];

    uint64_t grace_ts = 0;
    uint64_t last_perf_logging_ts = 0;
    while (running_) {
        auto current_ts = getCurrentTimeInNano();
        auto inflight_slices =
            worker.inflight_slices.load(std::memory_order_relaxed);
        if (inflight_slices ||
            current_ts - grace_ts <
                transport_->params_->workers.grace_period_ns) {
            asyncPostSend();
            asyncPollCq();
            if (inflight_slices) grace_ts = current_ts;
            const static uint64_t ONE_SECOND = 1000000000;
            if (transport_->params_->workers.show_latency_info &&
                current_ts - last_perf_logging_ts > ONE_SECOND) {
                showLatencyInfo();
                last_perf_logging_ts = current_ts;
            }
        } else {
            std::unique_lock<std::mutex> lock(worker.mutex);
            worker.in_suspend = true;
            worker.cv.wait(lock, [&]() -> bool {
                return !running_ || worker.inflight_slices.load(
                                        std::memory_order_acquire) > 0;
            });
            worker.in_suspend = false;
        }
    }
}

int Workers::handleContextEvents(std::shared_ptr<RdmaContext>& context) {
    ibv_async_event event;
    if (ibv_get_async_event(context->nativeContext(), &event) < 0) return -1;
    LOG(WARNING) << "Received context async event "
                 << ibv_event_type_str(event.event_type) << " for context "
                 << context->name();
    if (event.event_type == IBV_EVENT_QP_FATAL ||
        event.event_type == IBV_EVENT_WQ_FATAL) {
        auto endpoint = (RdmaEndPoint*)event.element.qp->qp_context;
        context->endpointStore()->remove(endpoint);
    } else if (event.event_type == IBV_EVENT_CQ_ERR) {
        context->pause();
        context->resume();
        LOG(WARNING) << "Action: " << context->name() << " restarted";
    } else if (event.event_type == IBV_EVENT_DEVICE_FATAL ||
               event.event_type == IBV_EVENT_PORT_ERR) {
        context->pause();
        LOG(WARNING) << "Action: " << context->name() << " down";
    } else if (event.event_type == IBV_EVENT_PORT_ACTIVE) {
        context->resume();
        LOG(WARNING) << "Action: " << context->name() << " up";
    }
    ibv_ack_async_event(&event);
    return 0;
}

void Workers::monitorThread() {
    while (running_) {
        for (auto& context : transport_->context_set_) {
            struct epoll_event event;
            if (context->eventFd() < 0) continue;
            int num_events = epoll_wait(context->eventFd(), &event, 1, 100);
            if (num_events < 0) {
                PLOG(ERROR) << "epoll_wait()";
                continue;
            }
            if (num_events == 0) continue;
            if (!(event.events & EPOLLIN)) continue;
            if (event.data.fd == context->nativeContext()->async_fd)
                handleContextEvents(context);
        }
    }
}

Status Workers::getRouteHint(RouteHint& hint, SegmentID segment_id,
                             uint64_t addr, uint64_t length) {
    auto& segment_manager = transport_->metadata_->segmentManager();
    if (segment_id == LOCAL_SEGMENT_ID) {
        hint.segment = segment_manager.getLocal().get();
    } else {
        CHECK_STATUS(segment_manager.getRemoteCached(hint.segment, segment_id));
    }
    hint.buffer = hint.segment->findBuffer(addr, length);
    if (!hint.buffer) {
        return Status::AddressNotRegistered(
            "No matched buffer in given address range" LOC_MARK);
    }
    if (hint.segment->type != SegmentType::Memory)
        return Status::AddressNotRegistered("Segment type not memory" LOC_MARK);
    hint.topo = &std::get<MemorySegmentDesc>(hint.segment->detail).topology;
    std::string location = hint.buffer->location;
    if (!hint.buffer->regions.empty()) {
        size_t offset = hint.buffer->addr;
        size_t best_overlap = 0;
        size_t target_start = addr;
        size_t target_end = addr + length;
        for (auto& entry : hint.buffer->regions) {
            size_t region_start = offset;
            size_t region_end = offset + entry.size;
            size_t overlap_start = std::max(region_start, target_start);
            size_t overlap_end = std::min(region_end, target_end);
            size_t overlap = (overlap_end > overlap_start)
                                 ? (overlap_end - overlap_start)
                                 : 0;
            if (overlap > best_overlap) {
                best_overlap = overlap;
                location = entry.location;
            }
            offset += entry.size;
        }
    }
    auto mem_id = hint.topo->getMemId(location);
    if (mem_id < 0) mem_id = hint.topo->getMemId(kWildcardLocation);
    hint.topo_entry = hint.topo->getMemEntry(mem_id);
    return Status::OK();
}

int Workers::getDeviceRank(const RouteHint& hint, int device_id) {
    for (size_t rank = 0; rank < Topology::DevicePriorityRanks; ++rank) {
        auto& list = hint.topo_entry->device_list[rank];
        for (auto& entry : list)
            if (entry == device_id) return rank;
    }
    return -1;
}

Status Workers::selectOptimalDevice(RouteHint& source, RouteHint& target,
                                    RdmaSlice* slice) {
    auto& worker = worker_context_[tl_wid];

    // Device should already be assigned by quota
    if (slice->source_dev_id < 0) {
        return Status::InvalidArgument("Source device not assigned" LOC_MARK);
    }

    // Load rail topology and select target device
    auto& rail = worker.rails[target.segment->machine_id];
    if (!rail.ready() || target.topo != rail.remote())
        rail.load(source.topo, target.topo);

    // Select target device (only if not already set by retry logic)
    if (slice->target_dev_id < 0) {
        int mapped_dev_id = rail.findBestRemoteDevice(
            slice->source_dev_id, target.topo_entry->numa_node);

        // Try to use mapped device, fallback to any available device
        bool found_target = false;
        for (size_t rank = 0; rank < Topology::DevicePriorityRanks; ++rank) {
            const auto& list = target.topo_entry->device_list[rank];
            if (list.empty()) continue;
            auto it = std::find(list.begin(), list.end(), mapped_dev_id);
            if (it != list.end()) {
                slice->target_dev_id = mapped_dev_id;
                found_target = true;
                break;
            }
        }

        // Fallback: pick any device from target
        if (!found_target) {
            for (size_t rank = 0; rank < Topology::DevicePriorityRanks;
                 ++rank) {
                const auto& list = target.topo_entry->device_list[rank];
                if (list.empty()) continue;
                slice->target_dev_id =
                    list[SimpleRandom::Get().next(list.size())];
                break;
            }
        }
    }

    if (slice->target_dev_id < 0)
        return Status::DeviceNotFound(
            "No target device could access the slice memory region" LOC_MARK);

    // Verify rail availability
    if (!rail.available(slice->source_dev_id, slice->target_dev_id)) {
        return selectFallbackDevice(source, target, slice);
    }

    return Status::OK();
}

int Workers::getDeviceByFlatIndex(const RouteHint& hint, size_t flat_idx) {
    for (size_t rank = 0; rank < Topology::DevicePriorityRanks; ++rank) {
        auto& list = hint.topo_entry->device_list[rank];
        if (flat_idx < list.size()) return list[flat_idx];
        flat_idx -= list.size();
    }
    return -1;
}

Status Workers::selectFallbackDevice(RouteHint& source, RouteHint& target,
                                     RdmaSlice* slice) {
    bool same_machine =
        (source.segment->machine_id == target.segment->machine_id);

    std::vector<int> source_devices, target_devices;
    for (size_t rank = 0; rank < Topology::DevicePriorityRanks; ++rank) {
        for (int dev : source.topo_entry->device_list[rank]) {
            source_devices.push_back(dev);
        }
        for (int dev : target.topo_entry->device_list[rank]) {
            target_devices.push_back(dev);
        }
    }

    auto& worker = worker_context_[tl_wid];
    auto& rail = worker.rails[target.segment->machine_id];
    if (!rail.ready() || target.topo != rail.remote())
        rail.load(source.topo, target.topo);

    // Use retry_count to rotate through source devices
    // For each source device, prefer the associated (direct rail) remote device
    for (int pass = 0; pass < 2; ++pass) {
        bool prefer_healthy = (pass == 0);

        // Rotate through all source devices using retry_count as offset
        size_t start_sidx = source_devices.empty()
                                ? 0
                                : slice->retry_count % source_devices.size();
        for (size_t offset = 0; offset < source_devices.size(); ++offset) {
            size_t sidx = (start_sidx + offset) % source_devices.size();
            int sdev = source_devices[sidx];

            // Find the best matching target device for this source
            int mapped_tdev =
                rail.findBestRemoteDevice(sdev, target.topo_entry->numa_node);

            // Try mapped device first, then try all others
            std::vector<int> target_candidates;
            if (mapped_tdev >= 0) {
                target_candidates.push_back(mapped_tdev);
            }
            for (int tdev : target_devices) {
                if (tdev != mapped_tdev) {
                    target_candidates.push_back(tdev);
                }
            }

            for (int tdev : target_candidates) {
                bool reachable = true;
                bool is_degraded = false;

                if (same_machine) {
                    reachable = (sdev == tdev);  // loopback is safe
                } else {
                    reachable = rail.available(sdev, tdev);
                    if (reachable) {
                        is_degraded = rail.isDegraded(sdev, tdev);
                    }
                }

                // In first pass, skip degraded rails
                if (prefer_healthy && is_degraded) {
                    continue;
                }

                if (reachable) {
                    slice->source_dev_id = sdev;
                    slice->target_dev_id = tdev;
                    return Status::OK();
                }
            }
        }
    }

    LOG(ERROR) << "No available path found for slice " << slice;
    return Status::DeviceNotFound("No available path" LOC_MARK);
}

Status Workers::generatePostPath(RdmaSlice* slice) {
    RouteHint source, target;
    CHECK_STATUS(getRouteHint(source, LOCAL_SEGMENT_ID,
                              (uint64_t)slice->source_addr, slice->length));

    auto target_id = slice->task->request.target_id;
    CHECK_STATUS(getRouteHint(target, target_id, (uint64_t)slice->target_addr,
                              slice->length));

    if (slice->retry_count == 0)
        CHECK_STATUS(selectOptimalDevice(source, target, slice));
    else
        CHECK_STATUS(selectFallbackDevice(source, target, slice));
    slice->source_lkey = source.buffer->lkey[slice->source_dev_id];
    slice->target_rkey = target.buffer->rkey[slice->target_dev_id];
    return Status::OK();
}

int Workers::getCurrentPrioritySlot() const {
    // Get current time in nanoseconds using rdtscp
    uint64_t now_ns = getFastTimeNanos();
    // Convert to microseconds and divide by timeslice duration
    // Slot 0 (0-100us): HIGH only
    // Slot 1 (100-200us): MEDIUM + HIGH
    // Slot 2 (200-300us): ALL (LOW + MEDIUM + HIGH)
    // Then repeat
    uint64_t timeslice_ns = timeslice_us_ * 1000;
    int slot = static_cast<int>((now_ns / timeslice_ns) % kNumPriorityLevels);
    return slot;
}

}  // namespace tent
}  // namespace mooncake
