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

#include "tent/transport/rdma/rdma_transport.h"
#include "tent/transport/rdma/ibv_loader.h"
#include "tent/transport/rdma/quota.h"

#include <glog/logging.h>
#include <sys/mman.h>
#include <sys/time.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <future>
#include <limits>
#include <sstream>
#include <thread>

#include "tent/common/status.h"
#include "tent/common/utils/ip.h"
#include "tent/transport/rdma/buffers.h"
#include "tent/transport/rdma/endpoint_store.h"
#include "tent/transport/rdma/workers.h"
#include "tent/common/utils/string_builder.h"
#include "tent/runtime/platform.h"
#include "tent/runtime/topology.h"
#include "tent/common/utils/random.h"
#include "tent/thirdparty/nlohmann/json.h"

#define SET_DEVICE(key, param) \
    param = conf->get("transports/rdma/device/" #key, param)

#define SET_ENDPOINT(key, param) \
    param = conf->get("transports/rdma/endpoint/" #key, param)

#define SET_WORKERS(key, param) \
    param = conf->get("transports/rdma/workers/" #key, param)

namespace mooncake {
namespace tent {

namespace {

constexpr uint64_t kDefaultRdmaQuiesceTimeoutNs = 10000000000ull;

uint16_t getRdmaBindDefaultPort(const Config& config) {
    constexpr const char* kKey = "rpc_server_port";
    if (!config.contains(kKey)) return 0;

    json raw_value = config.get<json>(kKey, json());
    if (raw_value.is_number_integer() || raw_value.is_number_unsigned()) {
        long long value = raw_value.get<long long>();
        if (value >= 0 && value <= static_cast<long long>(
                                       std::numeric_limits<uint16_t>::max())) {
            return static_cast<uint16_t>(value);
        }
        return 0;
    }

    if (raw_value.is_string()) {
        const std::string string_value = raw_value.get<std::string>();
        char* end = nullptr;
        errno = 0;
        unsigned long value = std::strtoul(string_value.c_str(), &end, 10);
        if (errno == 0 && end != string_value.c_str() && *end == '\0' &&
            value <= std::numeric_limits<uint16_t>::max()) {
            return static_cast<uint16_t>(value);
        }
    }

    return 0;
}

}  // namespace

static Status configureLaneCount(std::shared_ptr<Config> conf,
                                 std::shared_ptr<RdmaParams> params) {
    constexpr int kUnset = -1;
    const int num_lanes = conf->get("transports/rdma/num_lanes", kUnset);
    const int num_cq_list =
        conf->get("transports/rdma/device/num_cq_list", kUnset);
    const int qp_mul_factor =
        conf->get("transports/rdma/endpoint/qp_mul_factor", kUnset);
    const int num_workers =
        conf->get("transports/rdma/workers/num_workers", kUnset);

    auto validate_positive = [](int value, const char* name) -> Status {
        if (value == kUnset || value > 0) return Status::OK();
        std::stringstream ss;
        ss << "Invalid RDMA " << name << ": " << value
           << ", expected a positive integer";
        return Status::InvalidArgument(ss.str() + LOC_MARK);
    };

    auto status = validate_positive(num_lanes, "num_lanes");
    if (!status.ok()) return status;
    status = validate_positive(num_cq_list, "device.num_cq_list");
    if (!status.ok()) return status;
    status = validate_positive(qp_mul_factor, "endpoint.qp_mul_factor");
    if (!status.ok()) return status;
    status = validate_positive(num_workers, "workers.num_workers");
    if (!status.ok()) return status;

    int lane_count = params->num_lanes;
    if (num_lanes != kUnset) {
        lane_count = num_lanes;
    } else if (num_cq_list != kUnset) {
        lane_count = num_cq_list;
    } else if (qp_mul_factor != kUnset) {
        lane_count = qp_mul_factor;
    } else if (num_workers != kUnset) {
        lane_count = num_workers;
    }

    auto validate_match = [lane_count](int value, const char* name) -> Status {
        if (value == kUnset || value == lane_count) return Status::OK();
        std::stringstream ss;
        ss << "Inconsistent RDMA lane configuration: " << name << "=" << value
           << " but expected lane count " << lane_count
           << " so worker/QP/CQ counts stay aligned";
        return Status::InvalidArgument(ss.str() + LOC_MARK);
    };

    status = validate_match(num_cq_list, "device.num_cq_list");
    if (!status.ok()) return status;
    status = validate_match(qp_mul_factor, "endpoint.qp_mul_factor");
    if (!status.ok()) return status;
    status = validate_match(num_workers, "workers.num_workers");
    if (!status.ok()) return status;

    if (num_cq_list != kUnset || qp_mul_factor != kUnset ||
        num_workers != kUnset) {
        LOG(WARNING) << "Legacy RDMA parallelism knobs "
                     << "(device.num_cq_list, endpoint.qp_mul_factor, "
                     << "workers.num_workers) are deprecated; prefer "
                     << "transports/rdma/num_lanes";
    }

    params->num_lanes = lane_count;
    params->device.num_cq_list = lane_count;
    params->endpoint.qp_mul_factor = lane_count;
    params->workers.num_workers = lane_count;
    return Status::OK();
}

static Status convertConfToRdmaParams(std::shared_ptr<Config> conf,
                                      std::shared_ptr<RdmaParams> params) {
    auto status = configureLaneCount(conf, params);
    if (!status.ok()) return status;

    SET_DEVICE(num_comp_channels, params->device.num_comp_channels);
    SET_DEVICE(port, params->device.port);
    SET_DEVICE(gid_index, params->device.gid_index);
    SET_DEVICE(max_cqe, params->device.max_cqe);

    SET_ENDPOINT(endpoint_store_cap, params->endpoint.endpoint_store_cap);
    SET_ENDPOINT(max_sge, params->endpoint.max_sge);
    SET_ENDPOINT(max_qp_wr, params->endpoint.max_qp_wr);
    SET_ENDPOINT(max_inline_bytes, params->endpoint.max_inline_bytes);
    SET_ENDPOINT(pkey_index, params->endpoint.pkey_index);
    SET_ENDPOINT(hop_limit, params->endpoint.hop_limit);
    SET_ENDPOINT(flow_label, params->endpoint.flow_label);
    SET_ENDPOINT(traffic_class, params->endpoint.traffic_class);
    SET_ENDPOINT(service_level, params->endpoint.service_level);
    SET_ENDPOINT(src_path_bits, params->endpoint.src_path_bits);
    SET_ENDPOINT(static_rate, params->endpoint.static_rate);
    SET_ENDPOINT(rq_psn, params->endpoint.rq_psn);
    SET_ENDPOINT(max_dest_rd_atomic, params->endpoint.max_dest_rd_atomic);
    SET_ENDPOINT(min_rnr_timer, params->endpoint.min_rnr_timer);
    SET_ENDPOINT(sq_psn, params->endpoint.sq_psn);
    SET_ENDPOINT(send_timeout, params->endpoint.send_timeout);
    SET_ENDPOINT(send_retry_count, params->endpoint.send_retry_count);
    SET_ENDPOINT(send_rnr_count, params->endpoint.send_rnr_count);
    SET_ENDPOINT(max_rd_atomic, params->endpoint.max_rd_atomic);

    size_t mtu_val = conf->get("transports/rdma/endpoint/path_mtu", 4096);
    if (mtu_val == 4096)
        params->endpoint.path_mtu = IBV_MTU_4096;
    else if (mtu_val == 2048)
        params->endpoint.path_mtu = IBV_MTU_2048;
    else if (mtu_val == 1024)
        params->endpoint.path_mtu = IBV_MTU_1024;
    else
        params->endpoint.path_mtu = IBV_MTU_512;

    // Optional per-pool QP layout (RFC #2568 step 2). Each entry defines a
    // named pool with its own QP count and link-layer SL/TC;
    // SelectionPolicy.qp_pool references these by name. Absent/empty => single
    // default pool (unchanged). The pool SL/TC live here in the RDMA config,
    // not in SelectionPolicy, to keep the link-layer QoS definition in the
    // transport layer; policies only reference a pool by name.
    params->endpoint.qp_pools.clear();
    auto qp_pools_json =
        conf->getArray<nlohmann::json>("transports/rdma/endpoint/qp_pools");
    for (const auto& pool_json : qp_pools_json) {
        if (!pool_json.is_object()) {
            LOG(WARNING) << "Ignore non-object entry in qp_pools";
            continue;
        }
        if (!pool_json.contains("name") || !pool_json["name"].is_string()) {
            LOG(WARNING) << "Ignore qp_pool entry without a string 'name'";
            continue;
        }
        QpPoolSegment seg;
        seg.name = pool_json["name"].get<std::string>();
        seg.num_qp = pool_json.value("num_qp", 0);
        if (seg.num_qp <= 0) {
            LOG(WARNING) << "Ignore qp_pool '" << seg.name
                         << "' with non-positive num_qp " << seg.num_qp;
            continue;
        }
        seg.service_level = pool_json.value("service_level", -1);
        seg.traffic_class = pool_json.value("traffic_class", -1);
        params->endpoint.qp_pools.push_back(std::move(seg));
    }
    if (!params->endpoint.qp_pools.empty()) {
        LOG(INFO) << "Configured " << params->endpoint.qp_pools.size()
                  << " QP pool(s) for per-class link-layer isolation";
    }

    SET_WORKERS(max_retry_count, params->workers.max_retry_count);
    SET_WORKERS(block_size, params->workers.block_size);
    SET_WORKERS(grace_period_ns, params->workers.grace_period_ns);
    SET_WORKERS(rail_topo_path, params->workers.rail_topo_path);

    params->verbose = conf->get("verbose", false);
    params->log_slice_affinity =
        conf->get("transports/rdma/log_slice_affinity", false);
    return Status::OK();
}

static bool isGpuDirectRdmaSupported(std::shared_ptr<Config> conf) {
    auto disable_gpu_direct =
        conf->get("transports/rdma/disable_gpu_direct_rdma", false);
    if (disable_gpu_direct) {
        return false;
    }
    // Detect vendor GPUDirect/peer-memory drivers from /proc/modules.
    // NVIDIA: nvidia_peermem. AMD: peermem is built into amdgpu (linked with
    // ib_core), so the amdgpu module itself is the presence signal.
    std::ifstream modules("/proc/modules");
    std::string line;
    while (std::getline(modules, line)) {
        const auto name_end = line.find(' ');
        const auto name =
            name_end == std::string::npos ? line : line.substr(0, name_end);
        if (name == "nvidia_peermem" || name == "amdgpu") {
            return true;
        }
    }
    return false;
}

RdmaTransport::RdmaTransport()
    : installed_(false),
      notify_worker_running_(false),
      notify_poll_interval_us_(10) {}  // Start at 10us

RdmaTransport::~RdmaTransport() { uninstall(); }

size_t RdmaTransport::initializeContexts() {
    context_set_.clear();
    context_name_lookup_.clear();
    // One slot per NicID: dev_id arrives as a NicID and subscripts both this
    // and BufferDesc::lkey, so a compacted layout would name the wrong RNIC.
    // Skipped NICs keep an inert context, which consumers reject via status().
    context_set_.reserve(local_topology_->getNicCount());
    size_t context_count = 0;
    for (size_t i = 0; i < local_topology_->getNicCount(); ++i) {
        auto entry = local_topology_->getNicEntry(i);
        if (entry->type == Topology::NIC_RDMA) {
            auto context = std::make_shared<RdmaContext>(*this);
            if (context->construct(entry->name, params_) == 0) {
                context_name_lookup_[entry->name] = i;
                ++context_count;
                local_buffer_manager_.addDevice(context.get());
                context_set_.push_back(std::move(context));
                continue;
            }
            LOG(WARNING) << "Disable RDMA device " << entry->name << " because "
                         << "of initialization failure";
        }
        // A never-constructed context, not the one whose construct() failed:
        // the slot only has to stand in for the NicID, so it should not carry
        // a device name or an endpoint store it will never use.
        context_set_.push_back(std::make_shared<RdmaContext>(*this));
    }
    return context_count;
}

Status RdmaTransport::install(std::string& local_segment_name,
                              std::shared_ptr<ControlService> metadata,
                              std::shared_ptr<Topology> local_topology,
                              std::shared_ptr<Config> conf) {
    if (installed_) {
        return Status::InvalidArgument(
            "RDMA transport has been installed" LOC_MARK);
    }

    if (!IbvLoader::Instance().available()) {
        return Status::InvalidArgument("RDMA transport not available" LOC_MARK);
    }

    if (local_topology == nullptr ||
        !local_topology->getNicCount(Topology::NIC_RDMA)) {
        return Status::DeviceNotFound(
            "No RDMA device found in topology" LOC_MARK);
    }

    conf_ = conf;
    params_ = std::make_shared<RdmaParams>();
    auto param_status = convertConfToRdmaParams(conf_, params_);
    if (!param_status.ok()) return param_status;
    metadata_ = metadata;
    local_segment_name_ = local_segment_name;
    local_topology_ = local_topology;

    // In dual-NIC environments (e.g. separate TCP and RDMA interfaces),
    // transports/rdma/bind_address allows NIC paths to use an
    // RDMA-reachable IP while local_segment_name_ keeps the
    // TCP-reachable address for P2P.
    const auto rdma_bind_addr = conf_->get("transports/rdma/bind_address", "");
    if (!rdma_bind_addr.empty()) {
        const uint16_t default_port = getRdmaBindDefaultPort(*conf_);
        auto [host_name, port] =
            parseHostNameWithPort(local_segment_name, default_port);
        rdma_server_name_ = rdma_bind_addr + ":" + std::to_string(port);
        LOG(INFO) << "RdmaTransport(TENT): using RDMA bind address "
                  << rdma_server_name_
                  << " (TCP address: " << local_segment_name_ << ")";
    } else {
        rdma_server_name_ = local_segment_name_;
    }

    local_buffer_manager_.setTopology(local_topology);
    const bool context_empty = initializeContexts() == 0;
    const bool topology_empty = local_topology_->empty();
    if (context_empty || topology_empty) {
        const char* error_message = "No RDMA device initialized successfully";
        uninstall();
        return Status::DeviceNotFound(std::string(error_message) + LOC_MARK);
    }

    if (conf_->get("verbose", false)) {
        local_topology_->print();
    }
    setupLocalSegment();

    metadata_->setBootstrapRdmaCallback(
        std::bind(&RdmaTransport::onSetupRdmaConnections, this,
                  std::placeholders::_1, std::placeholders::_2));

    workers_ = std::make_unique<Workers>(this);
    workers_->start();

    // Start notification worker thread
    notify_worker_running_ = true;
    notify_worker_ = std::thread(&RdmaTransport::notifyWorkerThread, this);

    installed_ = true;
    caps.dram_to_dram = true;
    if (isGpuDirectRdmaSupported(conf_)) {
        caps.dram_to_gpu = true;
        caps.gpu_to_dram = true;
        caps.gpu_to_gpu = true;
    }
    return Status::OK();
}

Status RdmaTransport::quiesce() {
    uint64_t timeout_ns = kDefaultRdmaQuiesceTimeoutNs;
    if (conf_) {
        timeout_ns = conf_->get("transports/rdma/max_timeout_ns", timeout_ns);
    }
    Status drain = Status::OK();
    if (workers_) {
        drain = workers_->quiesce(timeout_ns);
        if (!drain.ok()) {
            LOG(ERROR) << "RDMA workers quiesce failed: " << drain.ToString();
        }
    }
    const uint64_t deadline_ns = getCurrentTimeInNano() + timeout_ns;
    while (true) {
        bool direct_busy = false;
        for (size_t i = 0; i < context_set_.size(); ++i) {
            pollDirectCompletions(static_cast<int>(i));
            auto& context = context_set_[i];
            if (context && context->hasDirectLaneOwner()) direct_busy = true;
        }
        if (!direct_busy) break;
        if (static_cast<uint64_t>(getCurrentTimeInNano()) >= deadline_ns) {
            Status direct_drain = Status::InternalError(
                "RDMA direct quiesce timed out with in-flight direct work" LOC_MARK);
            LOG(ERROR) << direct_drain.ToString();
            if (drain.ok()) drain = direct_drain;
            break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    const Status sync =
        Platform::getLoader().synchronizeDevices(local_topology_.get());
    if (!sync.ok()) {
        LOG(WARNING) << "RDMA dest-GPU sync during quiesce failed: "
                     << sync.ToString();
    }
    return drain;
}

Status RdmaTransport::uninstall() {
    // ControlService may still receive BootstrapRdma RPCs while uninstall is
    // running. Unregister and drain the callback before destroying workers,
    // contexts, and other state used by onSetupRdmaConnections(). Keep this
    // outside installed_ so partially-installed transports are covered too.
    if (metadata_) metadata_->setBootstrapRdmaCallback(nullptr);

    // Drain CQ and dest-GPU visibility while MRs and QPs are still alive.
    // Idempotent when deconstruct() already called quiesce().
    (void)quiesce();

    if (installed_) {
        // Stop notification worker thread
        notify_worker_running_ = false;
        if (notify_worker_.joinable()) {
            notify_worker_.join();
        }

        workers_.reset();
        metadata_.reset();
        local_buffer_manager_.clear();
        context_set_.clear();
        context_name_lookup_.clear();
        installed_ = false;
    }
    return Status::OK();
}

Status RdmaTransport::allocateSubBatch(SubBatchRef& batch, size_t max_size) {
    auto rdma_batch = Slab<RdmaSubBatch>::Get().allocate();
    if (!rdma_batch)
        return Status::InternalError(
            "Unable to allocate RDMA sub-batch" LOC_MARK);
    batch = rdma_batch;
    rdma_batch->task_list.reserve(max_size);
    rdma_batch->max_size = max_size;
    return Status::OK();
}

Status RdmaTransport::freeSubBatch(SubBatchRef& batch) {
    auto rdma_batch = dynamic_cast<RdmaSubBatch*>(batch);
    if (!rdma_batch)
        return Status::InvalidArgument("Invalid RDMA sub-batch" LOC_MARK);
    for (auto* task : rdma_batch->task_list) {
        task->deref();  // Release batch's reference to the task
    }
    rdma_batch->task_list.clear();
    for (auto slice : rdma_batch->slice_chain) {
        while (slice) {
            auto next = slice->next;
            RdmaSliceStorage::Get().deallocate(slice);
            slice = next;
        }
    }
    Slab<RdmaSubBatch>::Get().deallocate(rdma_batch);
    batch = nullptr;
    return Status::OK();
}

static inline uint64_t roundup(uint64_t a, uint64_t b) {
    return (a % b == 0) ? a : (a / b + 1) * b;
}

static std::string bufferLocationForRange(const BufferDesc& buffer,
                                          uint64_t addr, uint64_t length) {
    std::string location = buffer.location;
    if (buffer.regions.empty()) return location;

    uint64_t offset = buffer.addr;
    uint64_t best_overlap = 0;
    const uint64_t target_start = addr;
    const uint64_t target_end = addr + length;
    for (const auto& entry : buffer.regions) {
        const uint64_t region_start = offset;
        const uint64_t region_end = offset + entry.size;
        const uint64_t overlap_start = std::max(region_start, target_start);
        const uint64_t overlap_end = std::min(region_end, target_end);
        const uint64_t overlap =
            overlap_end > overlap_start ? overlap_end - overlap_start : 0;
        if (overlap > best_overlap) {
            best_overlap = overlap;
            location = entry.location;
        }
        offset += entry.size;
    }
    return location;
}

static bool memEntryContainsDevice(const Topology::MemEntry* entry,
                                   int device_id) {
    if (!entry) return false;
    for (size_t rank = 0; rank < Topology::DevicePriorityRanks; ++rank) {
        const auto& list = entry->device_list[rank];
        if (std::find(list.begin(), list.end(), device_id) != list.end())
            return true;
    }
    return false;
}

static int chooseTargetDevice(const Topology::MemEntry* entry,
                              const BufferDesc& buffer,
                              int preferred_device) {
    if (preferred_device >= 0 &&
        memEntryContainsDevice(entry, preferred_device) &&
        static_cast<size_t>(preferred_device) < buffer.rkey.size()) {
        return preferred_device;
    }
    if (!entry) return -1;
    for (size_t rank = 0; rank < Topology::DevicePriorityRanks; ++rank) {
        for (int device_id : entry->device_list[rank]) {
            if (device_id >= 0 &&
                static_cast<size_t>(device_id) < buffer.rkey.size()) {
                return device_id;
            }
        }
    }
    return -1;
}

int RdmaTransport::contextIndexForDevice(int device_id) const {
    if (!local_topology_) return -1;
    const auto* nic = local_topology_->getNicEntry(device_id);
    if (!nic) return -1;
    auto it = context_name_lookup_.find(nic->name);
    if (it == context_name_lookup_.end()) return -1;
    return it->second;
}

void RdmaTransport::pollDirectCompletions(int context_index) {
    if (context_index < 0 || context_index >= (int)context_set_.size()) return;
    auto& context = context_set_[context_index];
    if (!context || !context->directCq()) return;

    ibv_wc wc[8];
    int nr_poll = context->directCq()->poll(8, wc);
    if (nr_poll <= 0) return;

    for (int i = 0; i < nr_poll; ++i) {
        auto* slice = reinterpret_cast<RdmaSlice*>(wc[i].wr_id);
        if (!slice || !slice->task) continue;
        if (auto ep = slice->ep_weak_ptr.lock()) {
            ep->completeDirectSlice(slice);
            if (wc[i].status != IBV_WC_SUCCESS) {
                ep->resetConnection("Direct QP completion error");
            }
        }
        if (slice->task->direct) {
            context->releaseDirectLane(slice->task);
        }
        updateSliceStatus(slice,
                          wc[i].status == IBV_WC_SUCCESS ? COMPLETED : FAILED);
    }
}

bool RdmaTransport::trySubmitDirect(RdmaSubBatch* rdma_batch,
                                    const Request& request) {
    if (!rdma_batch || request.length == 0) return false;
    if (rdma_batch->task_list.size() + 1 > rdma_batch->max_size) return false;
    const size_t slice_size = params_->workers.block_size;
    if (slice_size == 0 || request.length > slice_size) return false;

    SegmentDescRef source_pin, target_pin;
    BufferDesc* source_buffer = nullptr;
    BufferDesc* target_buffer = nullptr;
    const Topology* source_topo = nullptr;
    const Topology* target_topo = nullptr;
    std::string source_location;
    std::string target_location;
    auto& segment_manager = metadata_->segmentManager();

    auto source_status = segment_manager.withCachedSegment(
        LOCAL_SEGMENT_ID, source_pin, [&](SegmentDesc* segment) {
            if (segment->type != SegmentType::Memory)
                return Status::NeedsRefreshCache(
                    "Local segment type is not Memory" LOC_MARK);
            source_buffer = segment->findBuffer(
                reinterpret_cast<uint64_t>(request.source), request.length);
            if (!source_buffer)
                return Status::NeedsRefreshCache(
                    "No matched local buffer for direct RDMA" LOC_MARK);
            source_topo = &std::get<MemorySegmentDesc>(segment->detail).topology;
            source_location = bufferLocationForRange(
                *source_buffer, reinterpret_cast<uint64_t>(request.source),
                request.length);
            return Status::OK();
        });
    if (!source_status.ok()) return false;

    auto target_status = segment_manager.withCachedSegment(
        request.target_id, target_pin, [&](SegmentDesc* segment) {
            if (segment->type != SegmentType::Memory)
                return Status::NeedsRefreshCache(
                    "Target segment type is not Memory" LOC_MARK);
            target_buffer =
                segment->findBuffer(request.target_offset, request.length);
            if (!target_buffer)
                return Status::NeedsRefreshCache(
                    "No matched target buffer for direct RDMA" LOC_MARK);
            target_topo = &std::get<MemorySegmentDesc>(segment->detail).topology;
            target_location = bufferLocationForRange(
                *target_buffer, request.target_offset, request.length);
            return Status::OK();
        });
    if (!target_status.ok()) return false;

    auto source_mem_id = source_topo->getMemId(source_location);
    if (source_mem_id < 0)
        source_mem_id = source_topo->getMemId(kWildcardLocation);
    const auto* source_mem_entry = source_topo->getMemEntry(source_mem_id);
    auto target_mem_id = target_topo->getMemId(target_location);
    if (target_mem_id < 0)
        target_mem_id = target_topo->getMemId(kWildcardLocation);
    const auto* target_mem_entry = target_topo->getMemEntry(target_mem_id);
    if (!source_mem_entry || !target_mem_entry) return false;

    int source_dev_id = -1;
    int context_index = -1;
    for (size_t rank = 0; rank < Topology::DevicePriorityRanks; ++rank) {
        for (int dev_id : source_mem_entry->device_list[rank]) {
            if (dev_id < 0 || dev_id >= 64) continue;
            if ((rdma_batch->device_mask & (1ULL << dev_id)) == 0) continue;
            if (static_cast<size_t>(dev_id) >= source_buffer->lkey.size())
                continue;
            int candidate_context = contextIndexForDevice(dev_id);
            if (candidate_context < 0 ||
                candidate_context >= (int)context_set_.size())
                continue;
            auto& context = context_set_[candidate_context];
            if (!context || context->status() != RdmaContext::DEVICE_ENABLED ||
                !context->directCq())
                continue;
            source_dev_id = dev_id;
            context_index = candidate_context;
            break;
        }
        if (source_dev_id >= 0) break;
    }
    if (source_dev_id < 0) return false;

    const int target_dev_id =
        chooseTargetDevice(target_mem_entry, *target_buffer, source_dev_id);
    if (target_dev_id < 0) return false;

    auto* task = RdmaTaskStorage::Get().allocate();
    if (!task) return false;
    auto& context = context_set_[context_index];
    if (!context->tryAcquireDirectLane(task)) {
        RdmaTaskStorage::Get().deallocate(task);
        return false;
    }

    auto* slice = RdmaSliceStorage::Get().allocate();
    if (!slice) {
        context->releaseDirectLane(task);
        RdmaTaskStorage::Get().deallocate(task);
        return false;
    }

    auto endpoint =
        getEndpointForContextIndex(context_index, request.target_id, target_dev_id);
    if (!endpoint || !endpoint->isDirectReady()) {
        RdmaSliceStorage::Get().deallocate(slice);
        context->releaseDirectLane(task);
        RdmaTaskStorage::Get().deallocate(task);
        return false;
    }

    task->num_slices = 1;
    task->request = request;
    task->device_mask = rdma_batch->device_mask;
    task->qp_pool = rdma_batch->qp_pool;
    task->status_word = PENDING;
    task->transferred_bytes = 0;
    task->success_slices.store(0, std::memory_order_relaxed);
    task->resolved_slices.store(0, std::memory_order_relaxed);
    task->first_error = PENDING;
    task->direct = true;
    task->direct_context_index = context_index;
    task->cancel_requested.store(false, std::memory_order_relaxed);
    task->ref();
    task->ref();

    slice->source_addr = request.source;
    slice->target_addr = request.target_offset;
    slice->length = request.length;
    slice->task = task;
    slice->next = nullptr;
    slice->source_lkey = source_buffer->lkey[source_dev_id];
    slice->target_rkey = target_buffer->rkey[target_dev_id];
    slice->source_dev_id = source_dev_id;
    slice->target_dev_id = target_dev_id;
    slice->ep_weak_ptr.reset();
    slice->word = PENDING;
    slice->qp_index = -1;
    slice->owner_worker.store(-1, std::memory_order_relaxed);
    slice->counted_lane.store(-1, std::memory_order_relaxed);
    slice->charged_dev.store(-1, std::memory_order_relaxed);
    slice->posted_dev.store(-1, std::memory_order_relaxed);
    slice->retry_count = 0;
    slice->last_fallback_idx = -1;
    slice->failed = false;
    slice->enqueue_ts = getCurrentTimeInNano();
    slice->submit_ts = slice->enqueue_ts;
    slice->priority = request.priority;

    auto post_status = endpoint->submitDirectSlice(slice);
    if (!post_status.ok()) {
        context->releaseDirectLane(task);
        task->deref();
        task->deref();
        RdmaSliceStorage::Get().deallocate(slice);
        return false;
    }

    rdma_batch->task_list.push_back(task);
    rdma_batch->slice_chain.push_back(slice);
    return true;
}

Status RdmaTransport::submitTransferTasks(
    SubBatchRef batch, const std::vector<Request>& request_list) {
    auto rdma_batch = dynamic_cast<RdmaSubBatch*>(batch);
    if (!rdma_batch)
        return Status::InvalidArgument("Invalid RDMA sub-batch" LOC_MARK);
    if (request_list.size() + rdma_batch->task_list.size() >
        rdma_batch->max_size)
        return Status::TooManyRequests("Exceed batch capacity" LOC_MARK);

    const size_t default_block_size = params_->workers.block_size;
    const int num_workers = params_->workers.num_workers;
    std::vector<RdmaSliceList> slice_lists(num_workers);
    std::vector<RdmaSlice*> slice_tails(num_workers, nullptr);
    auto enqueue_ts = getCurrentTimeInNano();

    // Distribute starting worker across threads to avoid contention
    static std::atomic<int> g_caller_threads(0);
    thread_local int tl_caller_id = g_caller_threads.fetch_add(1);
    int next_worker_idx = tl_caller_id;
    for (auto& request : request_list) {
        auto opcode = request.opcode;
        auto type = Platform::getLoader().getMemoryType(request.source);
        size_t max_slice_count = 64;
        if (type == MTYPE_CUDA || opcode == Request::WRITE)
            max_slice_count = 32;
        auto* task = RdmaTaskStorage::Get().allocate();
        rdma_batch->task_list.push_back(task);
        task->request = request;
        task->device_mask = rdma_batch->device_mask;
        task->qp_pool = rdma_batch->qp_pool;  // RFC #2568 step 3
        task->num_slices = 0;
        task->status_word = PENDING;
        task->transferred_bytes = 0;
        task->success_slices.store(0, std::memory_order_relaxed);
        task->resolved_slices.store(0, std::memory_order_relaxed);
        task->first_error = PENDING;
        task->direct = false;
        task->direct_context_index = -1;
        task->cancel_requested.store(false, std::memory_order_relaxed);
        task->ref();  // Batch holds a reference to the task

        const double merge_ratio = 0.25;
        uint64_t base_block = default_block_size;
        uint64_t num_slices = (request.length + base_block - 1) / base_block;
        num_slices = std::max<uint64_t>(
            1, std::min<uint64_t>(num_slices, max_slice_count));

        if (num_slices > 1) {
            uint64_t tail = request.length % base_block;
            if (tail > 0 &&
                tail < static_cast<uint64_t>(base_block * merge_ratio)) {
                num_slices = std::max<uint64_t>(1, num_slices - 1);
            }
        }

        uint64_t block_size = roundup(
            (request.length + num_slices - 1) / num_slices, default_block_size);

        std::vector<int> slice_dev_ids;
        // Only if a single request is enough, we perform aggregated allocation
        if (num_slices >= max_slice_count / 2) {
            std::string source_location = kWildcardLocation;
            auto source_locations =
                Platform::getLoader().getLocation(request.source, 1, true);
            if (!source_locations.empty()) {
                source_location = source_locations[0].location;
            }
            auto device_selector = workers_->getDeviceSelector();
            if (device_selector) {
                auto status = device_selector->allocate(
                    request.length, static_cast<uint32_t>(num_slices),
                    block_size, source_location, slice_dev_ids,
                    request.priority, batch->device_mask);
                if (!status.ok() || slice_dev_ids.empty()) {
                    LOG(WARNING) << "Device quota allocation failed: "
                                 << status.message();
                }
            }
        }

        uint64_t offset = 0;
        for (uint64_t slice_idx = 0; slice_idx < num_slices; ++slice_idx) {
            uint64_t length =
                std::min<uint64_t>(request.length - offset, block_size);
            auto slice = RdmaSliceStorage::Get().allocate();
            slice->source_addr = (char*)request.source + offset;
            slice->target_addr = request.target_offset + offset;
            slice->length = length;
            slice->task = task;
            slice->retry_count = 0;
            slice->last_fallback_idx = -1;
            slice->charged_dev = -1;
            slice->posted_dev = -1;
            slice->counted_lane = -1;
            slice->ep_weak_ptr.reset();
            slice->word = PENDING;
            slice->next = nullptr;
            slice->enqueue_ts = enqueue_ts;
            slice->priority = request.priority;  // Copy priority from request
            task->num_slices++;
            task->ref();  // Each slice holds a reference to the task
            if (slice_idx < slice_dev_ids.size()) {
                slice->source_dev_id = slice_dev_ids[slice_idx];
                slice->charged_dev = slice->source_dev_id;
            }
            offset += length;
            int part_id = next_worker_idx % num_workers;
            auto& list = slice_lists[part_id];
            auto& tail = slice_tails[part_id];
            list.num_slices++;
            next_worker_idx++;
            if (list.first) {
                tail->next = slice;
                tail = slice;
            } else {
                list.first = tail = slice;
            }
        }
    }

    for (int i = 0; i < num_workers; ++i) {
        if (slice_lists[i].first) {
            rdma_batch->slice_chain.push_back(slice_lists[i].first);
            workers_->submit(slice_lists[i], i);
        }
    }
    return Status::OK();
}

Status RdmaTransport::getTransferStatus(SubBatchRef batch, int task_id,
                                        TransferStatus& status) {
    auto rdma_batch = dynamic_cast<RdmaSubBatch*>(batch);
    if (task_id < 0 || task_id >= (int)rdma_batch->task_list.size()) {
        return Status::InvalidArgument("Invalid task ID" LOC_MARK);
    }
    auto* task = rdma_batch->task_list[task_id];
    if (task->direct && task->status_word == PENDING) {
        pollDirectCompletions(task->direct_context_index);
    }
    status = TransferStatus{task->status_word, task->transferred_bytes};
    return Status::OK();
}

Status RdmaTransport::cancelTransferTask(SubBatchRef batch, int task_id) {
    auto* rdma_batch = dynamic_cast<RdmaSubBatch*>(batch);
    if (!rdma_batch) {
        return Status::InvalidArgument("Invalid RDMA sub-batch" LOC_MARK);
    }
    if (task_id < 0 || task_id >= (int)rdma_batch->task_list.size()) {
        return Status::InvalidArgument("Invalid task ID" LOC_MARK);
    }
    auto* task = rdma_batch->task_list[task_id];
    if (task->status_word != PENDING) return Status::OK();
    if (task->direct) {
        task->cancel_requested.store(true, std::memory_order_release);
        pollDirectCompletions(task->direct_context_index);
        return Status::OK();
    }
    return workers_->cancel(task);
}

Status RdmaTransport::getNicLoadStats(std::vector<NicLoadStats>& stats) const {
    return workers_->getDeviceSelector()->getNicLoadStats(stats);
}

bool RdmaTransport::warmupMemory(void* addr, size_t length) {
    if (length < kMrWarmupMinBytes) return false;
    unsigned hwc = std::thread::hardware_concurrency();
    if (hwc < 4) return false;
    RdmaContext* warmup_ctx = nullptr;
    for (auto& ctx : context_set_) {
        if (ctx && ctx->status() == RdmaContext::DEVICE_ENABLED) {
            warmup_ctx = ctx.get();
            break;
        }
    }
    if (!warmup_ctx) return false;
    int ret = warmupMrRegistrationParallel(warmup_ctx, addr, length);
    if (ret != 0) {
        LOG(WARNING) << "MR warm-up failed (rc=" << ret
                     << "), falling back to cold registration";
        return false;
    }
    VLOG(1) << "MR warm-up succeeded for " << length << " bytes";
    return true;
}

Status RdmaTransport::addMemoryBuffer(BufferDesc& desc,
                                      const MemoryOptions& options) {
    CHECK_STATUS(local_buffer_manager_.addBuffer(desc, options));
    desc.transports.push_back(TransportType::RDMA);
    return Status::OK();
}

Status RdmaTransport::addMemoryBuffer(std::vector<BufferDesc>& desc_list,
                                      const MemoryOptions& options) {
    CHECK_STATUS(local_buffer_manager_.addBuffer(desc_list, options));
    for (auto& desc : desc_list) {
        desc.transports.push_back(TransportType::RDMA);
    }
    return Status::OK();
}

Status RdmaTransport::removeMemoryBuffer(BufferDesc& desc) {
    return local_buffer_manager_.removeBuffer(desc);
}

Status RdmaTransport::setupLocalSegment() {
    auto& manager = metadata_->segmentManager();
    CHECK_STATUS(manager.updateLocal([&](SegmentDesc& segment) -> Status {
        // Store RDMA server name for dual-NIC setups; when it differs from
        // local_segment_name_ the peer will use it for NIC path construction.
        if (rdma_server_name_ != local_segment_name_) {
            segment.rdma_server_name = rdma_server_name_;
        }
        auto& detail = std::get<MemorySegmentDesc>(segment.detail);
        for (auto& context : context_set_) {
            if (context->status() != RdmaContext::DEVICE_ENABLED) continue;
            DeviceDesc device_desc;
            device_desc.name = context->name();
            device_desc.lid = context->lid();
            device_desc.gid = context->gid();
            detail.devices.push_back(device_desc);
        }
        return Status::OK();
    }));
    return manager.synchronizeLocal();
}

int RdmaTransport::onSetupRdmaConnections(const BootstrapDesc& peer_desc,
                                          BootstrapDesc& local_desc) {
    auto local_nic_name = getNicNameFromNicPath(peer_desc.peer_nic_path);
    if (local_nic_name.empty() || !context_name_lookup_.count(local_nic_name)) {
        std::stringstream ss;
        ss << "No device found in local segment: " << local_nic_name;
        LOG(ERROR) << ss.str();
        local_desc.reply_msg = ss.str();
        return -1;
    }
    auto index = context_name_lookup_[local_nic_name];
    auto context = context_set_[index];
    auto ctx_status = context->status();
    if (ctx_status != RdmaContext::DEVICE_ENABLED &&
        ctx_status != RdmaContext::DEVICE_PAUSED) {
        std::stringstream ss;
        ss << "Device is down: " << peer_desc.local_nic_path;
        LOG(ERROR) << ss.str();
        local_desc.reply_msg = ss.str();
        return -1;
    }
    // Endpoints are never reset. A peer process that reused the same nic path
    // (same IP:port after a restart) hits an EP_READY endpoint whose QPs no
    // longer exist. accept() retires it and the next getOrInsert() creates a
    // fresh one. Do that retry inside this RPC so the initiator receives a
    // valid GID instead of an empty bootstrap reply.
    auto store = context->endpointStore();
    constexpr int kMaxAcceptAttempts = 2;
    for (int attempt = 0; attempt < kMaxAcceptAttempts; ++attempt) {
        auto endpoint = store->getOrInsert(peer_desc.local_nic_path);
        if (!endpoint) {
            std::stringstream ss;
            ss << "Cannot allocate endpoint: " << peer_desc.local_nic_path;
            LOG(ERROR) << ss.str();
            local_desc.reply_msg = ss.str();
            return -1;
        }
        local_desc = BootstrapDesc();
        auto status = endpoint->accept(peer_desc, local_desc);
        if (status.ok()) {
            local_desc.reply_msg.clear();
            return 0;
        }
        const auto ep_status = endpoint->status();
        const bool retired = ep_status == RdmaEndPoint::EP_DESTROYING ||
                             ep_status == RdmaEndPoint::EP_DESTROYED;
        if (retired) {
            store->remove(endpoint.get());
            if (attempt + 1 < kMaxAcceptAttempts) {
                LOG(INFO) << "Retrying RDMA bootstrap after retiring stale "
                             "endpoint for "
                          << peer_desc.local_nic_path;
                continue;
            }
        }
        LOG(ERROR) << status.ToString();
        local_desc.reply_msg = status.ToString();
        return -1;
    }

    return -1;
}

std::shared_ptr<RdmaEndPoint> RdmaTransport::getEndpoint(SegmentID target_id,
                                                         int device_id) {
    for (size_t i = 0; i < context_set_.size(); ++i) {
        auto& context = context_set_[i];
        if (context && context->status() == RdmaContext::DEVICE_ENABLED) {
            return getEndpointForContextIndex(static_cast<int>(i), target_id,
                                              device_id);
        }
    }
    return nullptr;
}

std::shared_ptr<RdmaEndPoint> RdmaTransport::getEndpointForContextIndex(
    int context_index, SegmentID target_id, int remote_device_id) {
    std::string rpc_server_addr, target_seg_name, target_dev_name,
        target_nic_path_name;

    auto status = metadata_->segmentManager().withCachedSegment(
        target_id, [&](SegmentDesc* segment) {
            if (segment->type != SegmentType::Memory) {
                return Status::NeedsRefreshCache(
                    "Segment type is not Memory" LOC_MARK);
            }

            if (target_id != LOCAL_SEGMENT_ID) {
                rpc_server_addr = segment->rpc_server_addr;
            }

            auto topo = &std::get<MemorySegmentDesc>(segment->detail).topology;
            target_seg_name = segment->name;
            target_nic_path_name = segment->nicPathServerName();
            target_dev_name = topo->getNicName(remote_device_id);
            if (target_seg_name.empty() || target_dev_name.empty()) {
                return Status::NeedsRefreshCache(
                    "Empty target segment or device name" LOC_MARK);
            }
            return Status::OK();
        });

    if (!status.ok()) {
        LOG(ERROR) << status.ToString();
        return nullptr;
    }

    if (context_index < 0 || context_index >= (int)context_set_.size()) {
        return nullptr;
    }
    auto* context = context_set_[context_index].get();
    if (!context || context->status() != RdmaContext::DEVICE_ENABLED)
        return nullptr;

    std::shared_ptr<RdmaEndPoint> endpoint;
    std::string peer_name = MakeNicPath(target_nic_path_name, target_dev_name);
    endpoint = context->endpointStore()->getOrInsert(peer_name);
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

Status RdmaTransport::sendNotification(SegmentID target_id,
                                       const Notification& notify) {
    auto endpoint = getEndpoint(target_id, LOCAL_SEGMENT_ID);
    if (!endpoint) {
        return Status::InternalError(
            "Endpoint not found for notification" LOC_MARK);
    }
    if (!endpoint->sendNotification(notify.name, notify.msg)) {
        return Status::InternalError("Failed to send notification" LOC_MARK);
    }
    return Status::OK();
}

Status RdmaTransport::receiveNotification(
    std::vector<Notification>& notify_list) {
    std::lock_guard<std::mutex> lock(notify_mutex_);
    if (notify_list_.empty()) {
        return Status::OK();
    }
    notify_list = std::move(notify_list_);
    notify_list_.clear();
    return Status::OK();
}

void RdmaTransport::addNotificationToQueue(const std::string& name,
                                           const std::string& msg) {
    std::lock_guard<std::mutex> lock(notify_mutex_);
    notify_list_.emplace_back(name, msg);
}

namespace {
// The notify QP carries its own host-memory send/recv buffers, so a local
// length/protection/WQE fault is confined to notification state. The data QPs
// of the same endpoint use separate WRs and MRs.
bool isNotifyLocalFault(ibv_wc_status status) {
    switch (status) {
        case IBV_WC_LOC_LEN_ERR:
        case IBV_WC_LOC_QP_OP_ERR:
        case IBV_WC_LOC_PROT_ERR:
        case IBV_WC_LOC_ACCESS_ERR:
        case IBV_WC_MW_BIND_ERR:
            return true;
        default:
            return false;
    }
}
}  // namespace

RdmaTransport::NotifyCompletionAction RdmaTransport::classifyNotifyCompletion(
    ibv_wc_status status, bool endpoint_alive, bool endpoint_ready) {
    // Every WR still posted on a retiring endpoint's notify QP flushes, which
    // is expected and must stay quiet.
    if (status == IBV_WC_WR_FLUSH_ERR && !endpoint_ready) {
        return NotifyCompletionAction::SkipSilently;
    }
    if (!endpoint_alive) return NotifyCompletionAction::ReportOnly;
    if (isNotifyLocalFault(status)) {
        return NotifyCompletionAction::DisableNotification;
    }
    return NotifyCompletionAction::RetireEndpoint;
}

int RdmaTransport::processNotifyCompletions() {
    int total_completions = 0;

    // Poll notification CQ from all contexts
    for (auto& context : context_set_) {
        auto notify_cq = context->notifyCq();
        if (!notify_cq) continue;

        ibv_wc wc[16];
        int completed = ibv_poll_cq(notify_cq->cq(), 16, wc);

        if (completed < 0) {
            PLOG(ERROR) << "Failed to poll notification CQ";
            continue;
        }

        if (completed == 0) continue;

        // Process each completion
        for (int i = 0; i < completed; ++i) {
            // Find endpoint by QP number before interpreting errors. A flush
            // completion after endpoint unpublication is expected during
            // retirement and should not flood logs.
            std::shared_ptr<RdmaEndPoint> endpoint;
            {
                RWSpinlock::ReadGuard guard(notify_endpoint_map_lock_);
                auto it = notify_qp_to_endpoint_.find(wc[i].qp_num);
                if (it != notify_qp_to_endpoint_.end()) {
                    endpoint = it->second.lock();
                }
            }

            if (wc[i].status != IBV_WC_SUCCESS) {
                // A failed completion leaves this notify QP unusable for good
                // and only the endpoint lifecycle builds a new one, so left
                // alone the endpoint stays EP_READY and every later
                // sendNotification() silently flushes. Retiring it also moves
                // the data QPs to ERR, so that is reserved for faults which may
                // mean the peer restarted or the path died. Both acting
                // branches re-take the notify_endpoint_map_lock_ ReadGuard
                // released above via unregisterNotifyQp(); the locally held
                // shared_ptr keeps the endpoint alive across the call.
                const bool endpoint_ready =
                    endpoint && endpoint->status() == RdmaEndPoint::EP_READY;
                auto action = classifyNotifyCompletion(
                    wc[i].status, endpoint != nullptr, endpoint_ready);
                if (action == NotifyCompletionAction::SkipSilently) continue;

                LOG(ERROR) << "Notification completion failed: " << wc[i].status
                           << ", qp_num=" << wc[i].qp_num;
                if (action == NotifyCompletionAction::DisableNotification) {
                    endpoint->disableNotification(
                        "notify QP local completion error");
                } else if (action == NotifyCompletionAction::RetireEndpoint) {
                    endpoint->resetConnection("notify QP completion error");
                }
                continue;
            }

            if (!endpoint) {
                LOG(WARNING) << "Received notification from unknown QP: "
                             << wc[i].qp_num;
                continue;
            }

            // Handle RECV completions: parse and add to transport queue
            if (wc[i].opcode == IBV_WC_RECV) {
                endpoint->handleNotifyRecv(wc[i].wr_id, wc[i].byte_len);
            } else if (wc[i].opcode == IBV_WC_SEND) {
                // Handle SEND completions: cleanup pending sends
                endpoint->handleNotifySendComplete(wc[i].wr_id);
            }
        }
    }

    return total_completions;
}

void RdmaTransport::registerNotifyQp(
    uint32_t qp_num, const std::shared_ptr<RdmaEndPoint>& endpoint) {
    RWSpinlock::WriteGuard guard(notify_endpoint_map_lock_);
    notify_qp_to_endpoint_[qp_num] = endpoint;
}

void RdmaTransport::unregisterNotifyQp(uint32_t qp_num) {
    RWSpinlock::WriteGuard guard(notify_endpoint_map_lock_);
    notify_qp_to_endpoint_.erase(qp_num);
}

void RdmaTransport::notifyWorkerThread() {
    while (notify_worker_running_) {
        processNotifyCompletions();
        usleep(notify_poll_interval_us_);
    }
}

double RdmaTransport::getEstimatedBandwidth() const {
    if (!workers_) return -1.0;
    auto* sel = workers_->getDeviceSelector();
    if (!sel) return -1.0;
    // The transmit estimate, not the selection EWMA: the admission queue
    // asks "how fast do bytes move once they are sent", and adds the wait
    // behind earlier work itself (DeadlineMlu's bytes_ahead), so the rate
    // must not fold that wait in the way the selection sample does.
    return sel->getAggregateTransmitBandwidth();
}

}  // namespace tent
}  // namespace mooncake
