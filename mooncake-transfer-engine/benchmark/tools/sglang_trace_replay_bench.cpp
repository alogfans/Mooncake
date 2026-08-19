// Copyright 2026 KVCache.AI
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

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <unistd.h>

#include "tent/common/config.h"
#include "tent/common/types.h"
#include "tent/transfer_engine.h"

DEFINE_string(mode, "replay", "target or replay");
DEFINE_string(scenario, "nonintent",
              "SGLang replay scenario: nonintent, intent, or qpool");
DEFINE_string(trace_file, "build/rdma_traffic.csv", "SGLang RDMA CSV trace");
DEFINE_string(local_segment_name, "",
              "TENT local segment base name. Empty uses the scenario default");
DEFINE_string(target_segment_name, "",
              "Decode/Store target segment base name. Empty uses the scenario "
              "default");
DEFINE_string(metadata_type, "http", "TENT metadata type");
DEFINE_string(metadata_servers, "http://qjh001:8080/metadata",
              "TENT metadata servers");
DEFINE_int32(rpc_server_port, 0,
             "TENT RPC server base port. 0 uses the scenario default");
DEFINE_string(cpu_location, "cpu:0", "DRAM location, e.g. cpu:0");
DEFINE_string(gpu_location, "cuda:{tp}",
              "GPU location template. cuda:0 maps TP i to cuda:i; use "
              "cuda:{tp} or rocm:{tp} to make the mapping explicit");
DEFINE_string(transport, "rdma", "TENT transport hint/registration type");
DEFINE_string(tent_conf_file, "",
              "Optional TENT config file loaded before benchmark overrides. "
              "Empty uses the scenario default");
DEFINE_uint64(buffer_size, 1ull << 30,
              "Bytes per local DRAM/GPU buffer on each process");
DEFINE_int32(tp_size, 4, "Tensor-parallel size; use 4 for 4P + 4D");
DEFINE_uint64(max_events, 0, "Maximum replayable trace events, 0 means all");
DEFINE_double(replay_scale, 1.0,
              "Trace time compression. 1.0 preserves sampled time");
DEFINE_double(duration_s, 300.0,
              "Replay duration in seconds. 0 replays the trace once; positive "
              "values repeat the trace until this duration");
DEFINE_bool(skip_control, false, "Skip metadata/control transfers");
DEFINE_uint64(control_max_bytes, 4096,
              "Transfers at or below this length are control transfers");
DEFINE_uint64(foreground_max_bytes, 0,
              "PD transfers at or below this non-control size are foreground; "
              "0 infers the minimum non-control PD length");
DEFINE_double(stats_skip_s, 10.0, "Skip warmup seconds in summary stats");
DEFINE_string(default_intent, "unspec", "Default TENT intent");
DEFINE_string(pd_intent, "",
              "Intent override for batch_transfer_sync. Empty uses the "
              "scenario default");
DEFINE_string(store_intent, "",
              "Intent override for mooncake_put/get. Empty uses the scenario "
              "default");
DEFINE_uint64(deadline_us, 0, "Per-request absolute deadline window, 0 disables");
DEFINE_bool(split_store_engine, false,
            "Use separate per-TP TE instances for PD and Store lanes");
DEFINE_int32(pd_traffic_class, -1,
             "RDMA traffic_class override for PD engines, -1 keeps config");
DEFINE_int32(pd_service_level, -1,
             "RDMA service_level override for PD engines, -1 keeps config");
DEFINE_int32(store_traffic_class, -1,
             "RDMA traffic_class override for Store engines, -1 keeps config");
DEFINE_int32(store_service_level, -1,
             "RDMA service_level override for Store engines, -1 keeps config");

namespace mooncake {
namespace tent {
namespace {

using Clock = std::chrono::steady_clock;

enum class Lane { kPd, kStore };
enum class MemKind { kDram, kGpu };
enum class EngineRole { kUnified, kPd, kStore };
enum class Scenario { kNonIntent, kIntent, kQpool };

struct TraceEvent {
    uint64_t index = 0;
    uint64_t release_ns = 0;
    std::string source;
    uint64_t length = 0;
    uint64_t request_id = 0;
    uint64_t new_seq = 0;
    uint64_t new_token = 0;
    uint64_t cached_token = 0;
    std::string prefill_time;
    int tp = 0;
    Lane lane = Lane::kPd;
    Request::OpCode opcode = Request::WRITE;
    MemKind local_mem = MemKind::kGpu;
    MemKind remote_mem = MemKind::kGpu;
};

struct LocalBuffers {
    void* dram = nullptr;
    std::vector<void*> gpu;
    std::vector<std::string> gpu_locations;
};

struct RemoteBuffer {
    uint64_t base = 0;
    uint64_t length = 0;
    std::string location;
};

struct RemoteBuffers {
    RemoteBuffer dram;
    std::vector<RemoteBuffer> gpu;
};

struct TpEngineContext {
    int tp = 0;
    std::unique_ptr<TransferEngine> engine;
    LocalBuffers local_buffers;
    SegmentID target = 0;
    RemoteBuffers remote_buffers;
    std::string target_segment_name;
};

struct Sample {
    uint64_t length = 0;
    uint64_t release_ns = 0;
    std::string group;
    bool foreground_pd = false;
    bool overlapped_store_put = false;
    double latency_us = 0.0;
    double response_us = 0.0;
    double instant_gbps = 0.0;
};

std::string endpointForTp(const std::string& base_endpoint, int tp);

Scenario parseScenarioOrDie() {
    if (FLAGS_scenario == "nonintent") return Scenario::kNonIntent;
    if (FLAGS_scenario == "intent") return Scenario::kIntent;
    if (FLAGS_scenario == "qpool") return Scenario::kQpool;
    LOG(FATAL) << "--scenario must be nonintent, intent, or qpool";
    return Scenario::kNonIntent;
}

std::string scenarioName(Scenario scenario) {
    switch (scenario) {
        case Scenario::kNonIntent:
            return "nonintent";
        case Scenario::kIntent:
            return "intent";
        case Scenario::kQpool:
            return "qpool";
    }
    return "nonintent";
}

template <typename T>
class BlockingQueue {
   public:
    void push(T value) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            queue_.push_back(std::move(value));
        }
        cv_.notify_one();
    }

    T pop() {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] { return !queue_.empty(); });
        T value = std::move(queue_.front());
        queue_.pop_front();
        return value;
    }

   private:
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<T> queue_;
};

class SampleSink {
   public:
    void add(Sample sample) {
        std::lock_guard<std::mutex> lock(mu_);
        samples_.push_back(std::move(sample));
    }

    std::vector<Sample> snapshot() const {
        std::lock_guard<std::mutex> lock(mu_);
        return samples_;
    }

   private:
    mutable std::mutex mu_;
    std::vector<Sample> samples_;
};

std::string roleName(EngineRole role) {
    switch (role) {
        case EngineRole::kPd:
            return "pd";
        case EngineRole::kStore:
            return "store";
        case EngineRole::kUnified:
            return "unified";
    }
    return "unified";
}

std::string roleLabel(EngineRole role) {
    return role == EngineRole::kUnified ? "unified" : roleName(role);
}

int defaultRpcBase(Scenario scenario, const std::string& mode) {
    const bool replay = mode == "replay";
    switch (scenario) {
        case Scenario::kNonIntent:
            return replay ? 19941 : 19901;
        case Scenario::kIntent:
            return replay ? 20041 : 20001;
        case Scenario::kQpool:
            return replay ? 20441 : 20401;
    }
    return replay ? 19941 : 19901;
}

std::string defaultTargetSegmentBase(Scenario scenario) {
    switch (scenario) {
        case Scenario::kNonIntent:
            return "sglang-nonintent-target";
        case Scenario::kIntent:
            return "sglang-intent-target";
        case Scenario::kQpool:
            return "sglang-qpool-target";
    }
    return "sglang-nonintent-target";
}

std::string defaultReplaySegmentBase(Scenario scenario) {
    switch (scenario) {
        case Scenario::kNonIntent:
            return "sglang-nonintent-replay-5m";
        case Scenario::kIntent:
            return "sglang-intent-replay-5m";
        case Scenario::kQpool:
            return "sglang-qpool-replay-5m";
    }
    return "sglang-nonintent-replay-5m";
}

std::string defaultTentConfFile(Scenario scenario) {
    switch (scenario) {
        case Scenario::kNonIntent:
            return "benchmarks/sglang_rdma_blacklist.json";
        case Scenario::kIntent:
            return "benchmarks/sglang_intent_baseline.json";
        case Scenario::kQpool:
            return "benchmarks/sglang_qpool_enhanced.json";
    }
    return "benchmarks/sglang_rdma_blacklist.json";
}

std::string effectiveTentConfFile() {
    if (!FLAGS_tent_conf_file.empty()) return FLAGS_tent_conf_file;
    return defaultTentConfFile(parseScenarioOrDie());
}

std::string effectiveLocalSegmentBase() {
    if (!FLAGS_local_segment_name.empty()) return FLAGS_local_segment_name;
    const auto scenario = parseScenarioOrDie();
    if (FLAGS_mode == "target") return defaultTargetSegmentBase(scenario);
    return defaultReplaySegmentBase(scenario);
}

std::string effectiveTargetSegmentBase() {
    if (!FLAGS_target_segment_name.empty()) return FLAGS_target_segment_name;
    return defaultTargetSegmentBase(parseScenarioOrDie());
}

int effectiveRpcServerPortBase() {
    if (FLAGS_rpc_server_port != 0) return FLAGS_rpc_server_port;
    return defaultRpcBase(parseScenarioOrDie(), FLAGS_mode);
}

std::string effectivePdIntentName() {
    if (!FLAGS_pd_intent.empty()) return FLAGS_pd_intent;
    const auto scenario = parseScenarioOrDie();
    if (scenario == Scenario::kIntent || scenario == Scenario::kQpool)
        return "foreground_get";
    return "";
}

std::string effectiveStoreIntentName() {
    if (!FLAGS_store_intent.empty()) return FLAGS_store_intent;
    const auto scenario = parseScenarioOrDie();
    if (scenario == Scenario::kIntent || scenario == Scenario::kQpool)
        return "background_prefetch";
    return "";
}

std::string segmentNameForTp(const std::string& base_name, int tp,
                             EngineRole role) {
    const std::string base =
        base_name.empty()
            ? "sglang-trace-replay-" + std::to_string(getpid())
            : base_name;
    std::string name = endpointForTp(base, tp);
    if (role != EngineRole::kUnified) {
        const std::string kRoleToken = "{role}";
        const auto token = name.find(kRoleToken);
        if (token != std::string::npos) {
            name.replace(token, kRoleToken.size(), roleName(role));
        } else {
            name += "-" + roleName(role);
        }
    }
    return name;
}

std::string localSegmentNameForTp(int tp, EngineRole role) {
    return segmentNameForTp(effectiveLocalSegmentBase(), tp, role);
}

int rpcPortForTp(int tp, EngineRole role) {
    const int base_port = effectiveRpcServerPortBase();
    if (role == EngineRole::kStore)
        return base_port + FLAGS_tp_size + tp;
    return base_port + tp;
}

void applyRdmaQosOverrides(Config& config, EngineRole role) {
    int traffic_class = -1;
    int service_level = -1;
    if (role == EngineRole::kStore) {
        traffic_class = FLAGS_store_traffic_class;
        service_level = FLAGS_store_service_level;
    } else {
        traffic_class = FLAGS_pd_traffic_class;
        service_level = FLAGS_pd_service_level;
    }
    if (traffic_class >= 0) {
        CHECK_LE(traffic_class, 255);
        config.set("transports/rdma/endpoint/traffic_class", traffic_class);
    }
    if (service_level >= 0) {
        CHECK_LE(service_level, 15);
        config.set("transports/rdma/endpoint/service_level", service_level);
    }
}

std::shared_ptr<Config> makeConfigForTp(int tp, EngineRole role) {
    auto config = std::make_shared<Config>();
    const auto conf_file = effectiveTentConfFile();
    if (!conf_file.empty()) {
        auto status = config->loadFile(conf_file);
        CHECK(status.ok()) << "Failed to load --tent_conf_file="
                           << conf_file << ": " << status.ToString();
    }
    config->set("local_segment_name", localSegmentNameForTp(tp, role));
    config->set("metadata_type", FLAGS_metadata_type);
    config->set("metadata_servers", FLAGS_metadata_servers);
    config->set("rpc_server_port", rpcPortForTp(tp, role));
    applyRdmaQosOverrides(*config, role);

    const auto transport = parseTransportType(FLAGS_transport);
    CHECK(transport != UNSPEC || FLAGS_transport == "unspec")
        << "Unknown --transport=" << FLAGS_transport;
    for (const char* name :
         {"rdma", "tcp", "shm", "nvlink", "gds", "io_uring", "mnnvl",
          "sunrise_link"}) {
        config->set(std::string("transports/") + name + "/enable",
                    FLAGS_transport == "unspec");
    }
    if (transport != UNSPEC) {
        config->set(std::string("transports/") + transportTypeName(transport) +
                        "/enable",
                    true);
    }
    return config;
}

std::string endpointForTp(const std::string& base_endpoint, int tp) {
    const std::string kTpToken = "{tp}";
    const auto token = base_endpoint.find(kTpToken);
    if (token != std::string::npos) {
        std::string endpoint = base_endpoint;
        endpoint.replace(token, kTpToken.size(), std::to_string(tp));
        return endpoint;
    }

    const auto colon = base_endpoint.rfind(':');
    if (colon == std::string::npos) {
        return base_endpoint + "-tp" + std::to_string(tp);
    }
    const auto host = base_endpoint.substr(0, colon);
    const int base_port = std::stoi(base_endpoint.substr(colon + 1));
    return host + ":" + std::to_string(base_port + tp);
}

bool parseIntent(const std::string& name, IntentType* out) {
    if (!out) return false;
    if (name.empty() || name == "unspec") {
        *out = IntentType::INTENT_UNSPEC;
        return true;
    }
    if (name == "foreground_get") {
        *out = IntentType::FOREGROUND_GET;
        return true;
    }
    if (name == "background_prefetch") {
        *out = IntentType::BACKGROUND_PREFETCH;
        return true;
    }
    if (name == "migration") {
        *out = IntentType::MIGRATION;
        return true;
    }
    if (name == "checkpoint") {
        *out = IntentType::CHECKPOINT;
        return true;
    }
    if (name == "weight_loading") {
        *out = IntentType::WEIGHT_LOADING;
        return true;
    }
    if (name == "staging_internal") {
        *out = IntentType::STAGING_INTERNAL;
        return true;
    }
    return false;
}

IntentType intentFor(const TraceEvent& ev, IntentType default_intent,
                     IntentType pd_intent, IntentType store_intent) {
    if (ev.lane == Lane::kPd && pd_intent != IntentType::INTENT_UNSPEC) {
        return pd_intent;
    }
    if (ev.lane == Lane::kStore &&
        store_intent != IntentType::INTENT_UNSPEC) {
        return store_intent;
    }
    return default_intent;
}

std::vector<std::string> splitCsvLine(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    std::stringstream ss(line);
    while (std::getline(ss, field, ',')) fields.push_back(field);
    if (!line.empty() && line.back() == ',') fields.emplace_back();
    return fields;
}

int64_t parseTimestampNs(const std::string& value) {
    std::tm tm = {};
    std::istringstream ss(value.substr(0, 19));
    ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
    CHECK(!ss.fail()) << "Invalid timestamp: " << value;
    uint64_t micros = 0;
    if (value.size() > 19 && value[19] == '.') {
        std::string frac = value.substr(20);
        if (frac.size() > 6) frac.resize(6);
        while (frac.size() < 6) frac.push_back('0');
        micros = std::stoull(frac);
    }
    tm.tm_isdst = -1;
    time_t seconds = std::mktime(&tm);
    CHECK(seconds >= 0) << "Invalid timestamp: " << value;
    return static_cast<int64_t>(seconds) * 1000000000ll +
           static_cast<int64_t>(micros) * 1000ll;
}

int columnIndex(const std::unordered_map<std::string, int>& columns,
                const std::string& name) {
    auto it = columns.find(name);
    return it == columns.end() ? -1 : it->second;
}

std::string fieldAt(const std::vector<std::string>& fields, int index) {
    if (index < 0 || static_cast<size_t>(index) >= fields.size()) return "";
    return fields[index];
}

uint64_t parseU64OrZero(const std::string& value) {
    return value.empty() ? 0 : std::stoull(value);
}

std::string eventKey(const TraceEvent& ev) {
    std::ostringstream os;
    os << ev.source << '|' << ev.length << '|' << ev.request_id << '|'
       << ev.new_seq << '|' << ev.new_token << '|' << ev.cached_token << '|'
       << ev.prefill_time;
    return os.str();
}

std::vector<TraceEvent> loadTrace(uint64_t* foreground_max_bytes) {
    CHECK(foreground_max_bytes != nullptr);
    std::ifstream input(FLAGS_trace_file);
    CHECK(input.is_open()) << "Unable to open trace file: " << FLAGS_trace_file;

    std::string header;
    CHECK(std::getline(input, header)) << "Empty trace file: "
                                       << FLAGS_trace_file;
    auto header_fields = splitCsvLine(header);
    std::unordered_map<std::string, int> columns;
    for (size_t i = 0; i < header_fields.size(); ++i) {
        columns[header_fields[i]] = static_cast<int>(i);
    }
    const int timestamp_col = columnIndex(columns, "timestamp");
    const int source_col = columnIndex(columns, "source");
    const int length_col = columnIndex(columns, "length");
    CHECK(timestamp_col >= 0 && source_col >= 0 && length_col >= 0)
        << "Trace CSV must contain timestamp, source, and length";

    std::vector<TraceEvent> events;
    std::unordered_map<std::string, uint64_t> occurrence;
    int64_t first_ts = -1;
    uint64_t inferred_foreground = std::numeric_limits<uint64_t>::max();
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        auto fields = splitCsvLine(line);
        TraceEvent ev;
        const auto ts = parseTimestampNs(fieldAt(fields, timestamp_col));
        if (first_ts < 0) first_ts = ts;
        ev.release_ns = static_cast<uint64_t>(std::max<int64_t>(0, ts - first_ts));
        ev.source = fieldAt(fields, source_col);
        ev.length = parseU64OrZero(fieldAt(fields, length_col));
        ev.request_id =
            parseU64OrZero(fieldAt(fields, columnIndex(columns, "request_id")));
        ev.new_seq =
            parseU64OrZero(fieldAt(fields, columnIndex(columns, "new_seq")));
        ev.new_token =
            parseU64OrZero(fieldAt(fields, columnIndex(columns, "new_token")));
        ev.cached_token = parseU64OrZero(
            fieldAt(fields, columnIndex(columns, "cached_token")));
        ev.prefill_time = fieldAt(fields, columnIndex(columns, "prefill_time"));
        ev.index = events.size();

        if (ev.length == 0) continue;
        if (FLAGS_skip_control && ev.length <= FLAGS_control_max_bytes) {
            continue;
        }

        if (ev.source == "batch_transfer_sync") {
            ev.lane = Lane::kPd;
            ev.opcode = Request::WRITE;
            ev.local_mem = MemKind::kGpu;
            ev.remote_mem = MemKind::kGpu;
            if (ev.length > FLAGS_control_max_bytes) {
                inferred_foreground = std::min(inferred_foreground, ev.length);
            }
        } else if (ev.source == "mooncake_put") {
            ev.lane = Lane::kStore;
            ev.opcode = Request::WRITE;
            ev.local_mem = MemKind::kGpu;
            ev.remote_mem = MemKind::kDram;
        } else if (ev.source == "mooncake_get") {
            ev.lane = Lane::kStore;
            ev.opcode = Request::READ;
            ev.local_mem = MemKind::kGpu;
            ev.remote_mem = MemKind::kDram;
        } else {
            continue;
        }

        const std::string key = eventKey(ev);
        ev.tp = static_cast<int>(occurrence[key]++ %
                                 static_cast<uint64_t>(std::max(1, FLAGS_tp_size)));
        events.push_back(std::move(ev));
        if (FLAGS_max_events != 0 && events.size() >= FLAGS_max_events) break;
    }

    if (*foreground_max_bytes == 0) {
        *foreground_max_bytes =
            inferred_foreground == std::numeric_limits<uint64_t>::max()
                ? FLAGS_control_max_bytes
                : inferred_foreground;
    }
    std::sort(events.begin(), events.end(), [](const auto& a, const auto& b) {
        if (a.release_ns != b.release_ns) return a.release_ns < b.release_ns;
        return a.index < b.index;
    });
    return events;
}

std::vector<TraceEvent> expandEventsForDuration(
    const std::vector<TraceEvent>& events) {
    if (FLAGS_duration_s <= 0.0) return events;
    CHECK(!events.empty());

    uint64_t max_release_ns = 0;
    for (const auto& ev : events) {
        max_release_ns = std::max(max_release_ns, ev.release_ns);
    }
    const uint64_t period_ns = std::max<uint64_t>(1, max_release_ns + 1000000ull);
    const auto target_ns =
        static_cast<uint64_t>(FLAGS_duration_s * 1000000000.0);
    std::vector<TraceEvent> expanded;
    expanded.reserve(events.size() *
                     std::max<uint64_t>(1, target_ns / period_ns + 1));
    for (uint64_t loop = 0;; ++loop) {
        const uint64_t base_ns = loop * period_ns;
        bool added = false;
        for (const auto& ev : events) {
            const uint64_t release_ns = base_ns + ev.release_ns;
            if (release_ns >= target_ns) continue;
            auto copy = ev;
            copy.release_ns = release_ns;
            copy.index = expanded.size();
            expanded.push_back(std::move(copy));
            added = true;
        }
        if (!added || base_ns + period_ns >= target_ns) break;
    }
    return expanded;
}

bool isDramLocation(const std::string& location) {
    return location == kWildcardLocation || location.rfind("cpu", 0) == 0;
}

std::string replaceAll(std::string value, const std::string& needle,
                       const std::string& replacement) {
    size_t pos = 0;
    while ((pos = value.find(needle, pos)) != std::string::npos) {
        value.replace(pos, needle.size(), replacement);
        pos += replacement.size();
    }
    return value;
}

std::string gpuLocationForTp(int tp) {
    const auto tp_value = std::to_string(tp);
    if (FLAGS_gpu_location.find("{tp}") != std::string::npos) {
        return replaceAll(FLAGS_gpu_location, "{tp}", tp_value);
    }
    if (FLAGS_gpu_location.find("{}") != std::string::npos) {
        return replaceAll(FLAGS_gpu_location, "{}", tp_value);
    }
    const auto colon = FLAGS_gpu_location.rfind(':');
    if (colon != std::string::npos) {
        return FLAGS_gpu_location.substr(0, colon + 1) + tp_value;
    }
    return FLAGS_gpu_location + ":" + tp_value;
}

std::string joinLocations(const std::vector<std::string>& locations) {
    std::ostringstream os;
    for (size_t i = 0; i < locations.size(); ++i) {
        if (i != 0) os << ',';
        os << locations[i];
    }
    return os.str();
}

RemoteBuffer selectRemoteDramBuffer(const SegmentInfo& info) {
    RemoteBuffer fallback;
    bool has_fallback = false;
    for (const auto& buffer : info.buffers) {
        if (!isDramLocation(buffer.location)) continue;
        if (buffer.location == FLAGS_cpu_location) {
            return RemoteBuffer{buffer.base, buffer.length, buffer.location};
        }
        if (!has_fallback) {
            fallback = RemoteBuffer{buffer.base, buffer.length, buffer.location};
            has_fallback = true;
        }
    }
    if (has_fallback) return fallback;
    LOG(FATAL) << "Target segment does not expose a DRAM buffer";
    return {};
}

RemoteBuffer selectRemoteGpuBuffer(const SegmentInfo& info,
                                   const std::string& location) {
    for (const auto& buffer : info.buffers) {
        if (!isDramLocation(buffer.location) && buffer.location == location) {
            return RemoteBuffer{buffer.base, buffer.length, buffer.location};
        }
    }
    LOG(FATAL) << "Target segment does not expose GPU buffer at " << location;
    return {};
}

Status allocateAndRegister(TransferEngine& engine, void** ptr,
                           const std::string& location, size_t size,
                           TransportType transport) {
    MemoryOptions options;
    options.location = location;
    options.perm = kGlobalReadWrite;
    options.type = transport;
    CHECK_STATUS(engine.allocateLocalMemory(ptr, size, options));
    CHECK_STATUS(engine.registerLocalMemory(*ptr, size, options));
    return Status::OK();
}

Status unregisterAndFree(TransferEngine& engine, void* ptr, size_t size) {
    if (!ptr) return Status::OK();
    CHECK_STATUS(engine.unregisterLocalMemory(ptr, size));
    CHECK_STATUS(engine.freeLocalMemory(ptr));
    return Status::OK();
}

void unregisterAndFreeBuffers(TransferEngine& engine, LocalBuffers* buffers,
                              size_t size) {
    if (!buffers) return;
    for (void* ptr : buffers->gpu) {
        (void)unregisterAndFree(engine, ptr, size);
    }
    (void)unregisterAndFree(engine, buffers->dram, size);
}

LocalBuffers allocateBuffersForTp(TransferEngine& engine, int tp, size_t size,
                                  TransportType transport) {
    LocalBuffers buffers;
    auto status = allocateAndRegister(engine, &buffers.dram, FLAGS_cpu_location,
                                      size, transport);
    CHECK(status.ok()) << "Failed to allocate/register DRAM buffer: "
                       << status.ToString();
    buffers.gpu.resize(FLAGS_tp_size, nullptr);
    buffers.gpu_locations.resize(FLAGS_tp_size);
    const auto location = gpuLocationForTp(tp);
    buffers.gpu_locations[tp] = location;
    status = allocateAndRegister(engine, &buffers.gpu[tp], location, size,
                                     transport);
    CHECK(status.ok()) << "Failed to allocate/register GPU buffer for TP "
                       << tp << " at " << location << ": "
                       << status.ToString();
    return buffers;
}

bool createContext(int tp, EngineRole role, TransportType transport,
                   TpEngineContext* context) {
    CHECK(context != nullptr);
    context->tp = tp;
    context->engine = std::make_unique<TransferEngine>(
        makeConfigForTp(tp, FLAGS_split_store_engine ? role : EngineRole::kUnified));
    if (!context->engine->available()) {
        LOG(ERROR) << "TENT TransferEngine is not available for TP " << tp
                   << " role=" << roleLabel(role);
        return false;
    }
    context->local_buffers =
        allocateBuffersForTp(*context->engine, tp, FLAGS_buffer_size, transport);
    return true;
}

bool openTargetForContext(TpEngineContext* context, EngineRole role) {
    CHECK(context != nullptr);
    const auto effective_role =
        FLAGS_split_store_engine ? role : EngineRole::kUnified;
    context->target_segment_name =
        segmentNameForTp(effectiveTargetSegmentBase(), context->tp,
                         effective_role);
    auto status =
        context->engine->openSegment(context->target,
                                     context->target_segment_name);
    if (!status.ok()) {
        LOG(ERROR) << "openSegment failed for TP " << context->tp
                   << " role=" << roleLabel(role) << " target "
                   << context->target_segment_name << ": "
                   << status.ToString();
        return false;
    }

    SegmentInfo info;
    status = context->engine->getSegmentInfo(context->target, info);
    if (!status.ok()) {
        LOG(ERROR) << "getSegmentInfo failed for TP " << context->tp
                   << " role=" << roleLabel(role) << ": "
                   << status.ToString();
        return false;
    }
    const auto location = gpuLocationForTp(context->tp);
    context->remote_buffers.dram = selectRemoteDramBuffer(info);
    context->remote_buffers.gpu.resize(FLAGS_tp_size);
    context->remote_buffers.gpu[context->tp] =
        selectRemoteGpuBuffer(info, location);
    return true;
}

uint64_t slotOffset(int tp, uint64_t event_index, uint64_t buffer_size,
                    uint64_t length, MemKind kind) {
    (void)tp;
    (void)kind;
    CHECK_GT(FLAGS_tp_size, 0);
    CHECK_GE(buffer_size, length)
        << "buffer_size is smaller than trace transfer length";
    const uint64_t span = buffer_size - length + 1;
    return ((event_index * 4096ull) % span) & ~63ull;
}

void* localAddress(const LocalBuffers& buffers, const TraceEvent& ev) {
    char* base = nullptr;
    if (ev.local_mem == MemKind::kGpu) {
        CHECK_GE(ev.tp, 0);
        CHECK_LT(static_cast<size_t>(ev.tp), buffers.gpu.size());
        base = static_cast<char*>(buffers.gpu[ev.tp]);
    } else {
        base = static_cast<char*>(buffers.dram);
    }
    return base + slotOffset(ev.tp, ev.index, FLAGS_buffer_size, ev.length,
                             ev.local_mem);
}

uint64_t remoteAddress(const RemoteBuffer& buffer, const TraceEvent& ev,
                       MemKind kind) {
    return buffer.base + slotOffset(ev.tp, ev.index, buffer.length, ev.length,
                                    kind);
}

uint64_t steadyClockNs() {
    auto now = Clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
}

uint64_t deadlineNs() {
    if (FLAGS_deadline_us == 0) return 0;
    return steadyClockNs() + FLAGS_deadline_us * 1000ull;
}

double gbps(uint64_t bytes, double latency_us) {
    if (latency_us <= 0.0) return 0.0;
    return static_cast<double>(bytes) / (1000.0 * latency_us);
}

Clock::time_point releaseTime(Clock::time_point start, uint64_t release_ns) {
    const double scale = FLAGS_replay_scale <= 0.0 ? 1.0 : FLAGS_replay_scale;
    return start + std::chrono::nanoseconds(
                       static_cast<int64_t>(release_ns / scale));
}

std::string groupFor(const TraceEvent& ev, bool overlapped_store_put) {
    if (ev.source == "batch_transfer_sync") {
        if (ev.length <= FLAGS_control_max_bytes) {
            return "pd.metadata";
        }
        return overlapped_store_put ? "pd.data.overlap_store_put"
                                    : "pd.data.no_store_put";
    }
    if (ev.source == "mooncake_put") return "store.put";
    if (ev.source == "mooncake_get") return "store.get";
    return ev.source;
}

bool waitForCompletion(TransferEngine& engine, BatchID batch_id) {
    for (;;) {
        TransferStatus status;
        auto result = engine.getTransferStatus(batch_id, status);
        if (!result.ok()) {
            LOG(ERROR) << "getTransferStatus failed: " << result.ToString();
            return false;
        }
        if (status.s == COMPLETED) return true;
        if (status.s == FAILED || status.s == TIMEOUT ||
            status.s == CANCELED || status.s == INVALID) {
            LOG(ERROR) << "Transfer ended with status=" << status.s;
            return false;
        }
        std::this_thread::yield();
    }
}

bool submitOne(TransferEngine& engine, SegmentID target,
               const LocalBuffers& local_buffers, const RemoteBuffers& remote,
               const TraceEvent& ev,
               IntentType intent, TransportType transport,
               double* latency_us) {
    auto batch = engine.allocateBatch(1);
    Request request;
    request.opcode = ev.opcode;
    request.source = localAddress(local_buffers, ev);
    request.target_id = target;
    if (ev.remote_mem == MemKind::kGpu) {
        CHECK_GE(ev.tp, 0);
        CHECK_LT(static_cast<size_t>(ev.tp), remote.gpu.size());
        request.target_offset = remoteAddress(remote.gpu[ev.tp], ev, ev.remote_mem);
    } else {
        request.target_offset = remoteAddress(remote.dram, ev, ev.remote_mem);
    }
    request.length = ev.length;
    request.transport_hint = transport;
    request.intent_type = intent;
    request.deadline_ns = deadlineNs();

    const uint64_t start_ns = steadyClockNs();
    auto status = engine.submitTransfer(batch, {request});
    if (!status.ok()) {
        LOG(ERROR) << "submitTransfer failed for event " << ev.index << ": "
                   << status.ToString();
        (void)engine.freeBatch(batch);
        return false;
    }
    const bool ok = waitForCompletion(engine, batch);
    const uint64_t finish_ns = steadyClockNs();
    status = engine.freeBatch(batch);
    if (!status.ok()) {
        LOG(ERROR) << "freeBatch failed: " << status.ToString();
    }
    if (latency_us) {
        *latency_us = static_cast<double>(finish_ns - start_ns) / 1000.0;
    }
    return ok;
}

void decodeScheduler(int tp, const std::vector<TraceEvent>& events,
                     BlockingQueue<const TraceEvent*>* queue,
                     Clock::time_point replay_start) {
    for (const auto& ev : events) {
        if (ev.tp != tp || ev.lane != Lane::kPd) continue;
        std::this_thread::sleep_until(releaseTime(replay_start, ev.release_ns));
        queue->push(&ev);
    }
    queue->push(nullptr);
}

void pdWorker(TpEngineContext* context, BlockingQueue<const TraceEvent*>* queue,
              Clock::time_point replay_start, IntentType default_intent,
              IntentType pd_intent, IntentType store_intent,
              TransportType transport, uint64_t foreground_max_bytes,
              std::atomic<uint64_t>* put_epoch, std::atomic<int>* active_puts,
              SampleSink* sink, std::atomic<bool>* ok) {
    while (ok->load(std::memory_order_acquire)) {
        const TraceEvent* ev = queue->pop();
        if (!ev) return;
        const auto scheduled = releaseTime(replay_start, ev->release_ns);
        const auto before_epoch = put_epoch->load(std::memory_order_acquire);
        const auto before_active = active_puts->load(std::memory_order_acquire);
        double latency_us = 0.0;
        const auto issue = Clock::now();
        const bool success = submitOne(
            *context->engine, context->target, context->local_buffers,
            context->remote_buffers, *ev,
            intentFor(*ev, default_intent, pd_intent, store_intent), transport,
            &latency_us);
        const auto done = Clock::now();
        const auto after_epoch = put_epoch->load(std::memory_order_acquire);
        const auto after_active = active_puts->load(std::memory_order_acquire);
        if (!success) {
            ok->store(false, std::memory_order_release);
            return;
        }
        const bool collect =
            static_cast<double>(ev->release_ns) / 1e9 >= FLAGS_stats_skip_s;
        if (collect) {
            const bool overlap = before_active > 0 || after_active > 0 ||
                                 before_epoch != after_epoch;
            Sample sample;
            sample.length = ev->length;
            sample.release_ns = ev->release_ns;
            sample.foreground_pd = ev->length > FLAGS_control_max_bytes &&
                                   ev->length <= foreground_max_bytes;
            sample.overlapped_store_put = overlap;
            sample.group = groupFor(*ev, overlap);
            sample.latency_us = latency_us;
            sample.response_us =
                static_cast<double>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(done -
                                                                         scheduled)
                        .count()) /
                1000.0;
            sample.instant_gbps = gbps(ev->length, latency_us);
            sink->add(std::move(sample));
        }
        (void)issue;
    }
}

void storeWorker(int tp, const std::vector<TraceEvent>& events,
                 TpEngineContext* context,
                 Clock::time_point replay_start,
                 IntentType default_intent, IntentType pd_intent,
                 IntentType store_intent, TransportType transport,
                 std::atomic<uint64_t>* put_epoch, std::atomic<int>* active_puts,
                 SampleSink* sink, std::atomic<bool>* ok) {
    for (const auto& ev : events) {
        if (!ok->load(std::memory_order_acquire)) return;
        if (ev.tp != tp || ev.lane != Lane::kStore) continue;
        std::this_thread::sleep_until(releaseTime(replay_start, ev.release_ns));
        if (ev.source == "mooncake_put") {
            active_puts->fetch_add(1, std::memory_order_acq_rel);
            put_epoch->fetch_add(1, std::memory_order_acq_rel);
        }
        double latency_us = 0.0;
        const bool success = submitOne(
            *context->engine, context->target, context->local_buffers,
            context->remote_buffers, ev,
            intentFor(ev, default_intent, pd_intent, store_intent), transport,
            &latency_us);
        if (ev.source == "mooncake_put") {
            put_epoch->fetch_add(1, std::memory_order_acq_rel);
            active_puts->fetch_sub(1, std::memory_order_acq_rel);
        }
        if (!success) {
            ok->store(false, std::memory_order_release);
            return;
        }
        const bool collect =
            static_cast<double>(ev.release_ns) / 1e9 >= FLAGS_stats_skip_s;
        if (collect) {
            Sample sample;
            sample.length = ev.length;
            sample.release_ns = ev.release_ns;
            sample.group = groupFor(ev, false);
            sample.latency_us = latency_us;
            sample.response_us = latency_us;
            sample.instant_gbps = gbps(ev.length, latency_us);
            sink->add(std::move(sample));
        }
    }
}

double percentile(std::vector<double> values, double q) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    size_t index = static_cast<size_t>(std::ceil(q * values.size()));
    index = std::min(values.size() - 1, index == 0 ? 0 : index - 1);
    return values[index];
}

double average(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) /
           static_cast<double>(values.size());
}

void printGroup(const std::string& name, const std::vector<Sample>& samples) {
    std::vector<double> latency;
    std::vector<double> bandwidth;
    uint64_t bytes = 0;
    for (const auto& sample : samples) {
        latency.push_back(sample.latency_us);
        bandwidth.push_back(sample.instant_gbps);
        bytes += sample.length;
    }
    std::cout << std::fixed << std::setprecision(6) << name
              << " events=" << samples.size() << " bytes=" << bytes
              << " latency_us(p50/p95/p99)=" << percentile(latency, 0.50)
              << '/' << percentile(latency, 0.95) << '/'
              << percentile(latency, 0.99)
              << " avg_inst_GB/s=" << average(bandwidth)
              << " inst_GB/s(p50/p95/p99)=" << percentile(bandwidth, 0.50)
              << '/' << percentile(bandwidth, 0.95) << '/'
              << percentile(bandwidth, 0.99) << std::endl;
}

void printSummary(const std::vector<TraceEvent>& events,
                  const SampleSink& sample_sink,
                  uint64_t foreground_max_bytes) {
    uint64_t pd_events = 0, store_events = 0, pd_bytes = 0, store_bytes = 0;
    for (const auto& ev : events) {
        if (ev.lane == Lane::kPd) {
            ++pd_events;
            pd_bytes += ev.length;
        } else {
            ++store_events;
            store_bytes += ev.length;
        }
    }
    std::cout << "Trace loaded: pd_events=" << pd_events
              << " pd_bytes=" << pd_bytes << " store_events=" << store_events
              << " store_bytes=" << store_bytes
              << " foreground_max_bytes=" << foreground_max_bytes << std::endl;

    const auto samples = sample_sink.snapshot();
    printGroup("[summary][overall]", samples);

    std::unordered_map<std::string, std::vector<Sample>> groups;
    std::vector<Sample> foreground;
    std::vector<Sample> foreground_overlap;
    std::vector<Sample> foreground_no_overlap;
    for (const auto& sample : samples) {
        groups[sample.group].push_back(sample);
        if (sample.foreground_pd) {
            foreground.push_back(sample);
            if (sample.overlapped_store_put) {
                foreground_overlap.push_back(sample);
            } else {
                foreground_no_overlap.push_back(sample);
            }
        }
    }
    std::vector<std::string> order;
    order.reserve(groups.size());
    for (const auto& kv : groups) order.push_back(kv.first);
    std::sort(order.begin(), order.end());
    for (const auto& name : order) {
        printGroup("[summary][" + name + "]", groups[name]);
    }
    printGroup("[summary][foreground_pd]", foreground);
    printGroup("[summary][foreground_pd.overlap_store_put]", foreground_overlap);
    printGroup("[summary][foreground_pd.no_store_put]", foreground_no_overlap);
}

int runTarget() {
    CHECK_GT(FLAGS_tp_size, 0);
    const auto transport = parseTransportType(FLAGS_transport);

    std::vector<TpEngineContext> contexts;
    contexts.reserve(FLAGS_tp_size * (FLAGS_split_store_engine ? 2 : 1));
    for (int tp = 0; tp < FLAGS_tp_size; ++tp) {
        const auto roles = FLAGS_split_store_engine
                               ? std::vector<EngineRole>{EngineRole::kPd,
                                                         EngineRole::kStore}
                               : std::vector<EngineRole>{EngineRole::kUnified};
        for (const auto role : roles) {
            TpEngineContext context;
            if (!createContext(tp, role, transport, &context)) {
                return EXIT_FAILURE;
            }
            std::cout << "Target ready: tp=" << tp
                      << " role=" << roleLabel(role)
                      << " segment_name=" << context.engine->getSegmentName()
                      << " rpc_port=" << rpcPortForTp(
                             tp, FLAGS_split_store_engine ? role
                                                          : EngineRole::kUnified)
                      << " dram=" << FLAGS_cpu_location
                      << " gpu=" << context.local_buffers.gpu_locations[tp]
                      << " bytes_per_buffer=" << FLAGS_buffer_size
                      << std::endl;
            contexts.push_back(std::move(context));
        }
    }

    for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    for (auto& context : contexts) {
        unregisterAndFreeBuffers(*context.engine, &context.local_buffers,
                                 FLAGS_buffer_size);
    }
    return EXIT_SUCCESS;
}

int runReplay() {
    CHECK_GT(FLAGS_tp_size, 0);
    IntentType default_intent, pd_intent, store_intent;
    if (!parseIntent(FLAGS_default_intent, &default_intent) ||
        !parseIntent(effectivePdIntentName(), &pd_intent) ||
        !parseIntent(effectiveStoreIntentName(), &store_intent)) {
        LOG(ERROR) << "Invalid intent flag";
        return EXIT_FAILURE;
    }

    uint64_t foreground_max_bytes = FLAGS_foreground_max_bytes;
    auto events = expandEventsForDuration(loadTrace(&foreground_max_bytes));
    if (events.empty()) {
        LOG(ERROR) << "No replayable SGLang transfer events found";
        return EXIT_FAILURE;
    }

    const auto transport = parseTransportType(FLAGS_transport);

    std::vector<TpEngineContext> pd_contexts;
    std::vector<TpEngineContext> store_contexts;
    pd_contexts.reserve(FLAGS_tp_size);
    store_contexts.reserve(FLAGS_split_store_engine ? FLAGS_tp_size : 0);
    std::vector<std::string> local_gpu_locations;
    std::vector<std::string> remote_gpu_locations;
    local_gpu_locations.reserve(FLAGS_tp_size);
    remote_gpu_locations.reserve(FLAGS_tp_size);
    for (int tp = 0; tp < FLAGS_tp_size; ++tp) {
        TpEngineContext pd_context;
        if (!createContext(tp, EngineRole::kPd, transport, &pd_context)) {
            return EXIT_FAILURE;
        }
        if (!openTargetForContext(&pd_context, EngineRole::kPd)) {
            return EXIT_FAILURE;
        }

        TpEngineContext* store_context = &pd_context;
        if (FLAGS_split_store_engine) {
            TpEngineContext separate_store_context;
            if (!createContext(tp, EngineRole::kStore, transport,
                               &separate_store_context)) {
                return EXIT_FAILURE;
            }
            if (!openTargetForContext(&separate_store_context,
                                      EngineRole::kStore)) {
                return EXIT_FAILURE;
            }
            store_contexts.push_back(std::move(separate_store_context));
            store_context = &store_contexts.back();
        }

        local_gpu_locations.push_back(pd_context.local_buffers.gpu_locations[tp]);
        remote_gpu_locations.push_back(pd_context.remote_buffers.gpu[tp].location);
        std::cout << "Replay TP ready: tp=" << tp
                  << " pd_segment=" << pd_context.engine->getSegmentName()
                  << " pd_target=" << pd_context.target_segment_name
                  << " store_segment=" << store_context->engine->getSegmentName()
                  << " store_target=" << store_context->target_segment_name
                  << " local_gpu=" << pd_context.local_buffers.gpu_locations[tp]
                  << " remote_gpu=" << pd_context.remote_buffers.gpu[tp].location
                  << " local_dram=" << FLAGS_cpu_location
                  << " remote_dram=" << store_context->remote_buffers.dram.location
                  << std::endl;
        pd_contexts.push_back(std::move(pd_context));
    }

    std::cout << "Replay start: scenario="
              << scenarioName(parseScenarioOrDie())
              << " trace=" << FLAGS_trace_file
              << " target=" << effectiveTargetSegmentBase()
              << " tp_size=" << FLAGS_tp_size
              << " split_store_engine=" << FLAGS_split_store_engine
              << " pd_route=GPU->GPU"
              << " store_put_route=GPU->DRAM"
              << " store_get_route=DRAM->GPU"
              << " local_gpu_per_tp=" << joinLocations(local_gpu_locations)
              << " local_dram=" << FLAGS_cpu_location
              << " remote_gpu_per_tp=" << joinLocations(remote_gpu_locations)
              << " remote_dram="
              << (FLAGS_split_store_engine ? store_contexts.front()
                                                  .remote_buffers.dram.location
                                           : pd_contexts.front()
                                                  .remote_buffers.dram.location)
              << " transport=" << transportTypeName(transport)
              << " replay_scale=" << FLAGS_replay_scale
              << " duration_s=" << FLAGS_duration_s
              << " tent_conf_file=" << effectiveTentConfFile() << std::endl;

    std::vector<BlockingQueue<const TraceEvent*>> pd_queues(FLAGS_tp_size);
    std::vector<std::thread> threads;
    SampleSink samples;
    std::atomic<uint64_t> put_epoch{0};
    std::atomic<int> active_puts{0};
    std::atomic<bool> ok{true};
    const auto replay_start = Clock::now() + std::chrono::milliseconds(200);

    for (int tp = 0; tp < FLAGS_tp_size; ++tp) {
        threads.emplace_back(decodeScheduler, tp, std::cref(events),
                             &pd_queues[tp], replay_start);
    }
    for (int tp = 0; tp < FLAGS_tp_size; ++tp) {
        threads.emplace_back(pdWorker, &pd_contexts[tp], &pd_queues[tp],
                             replay_start, default_intent, pd_intent,
                             store_intent, transport, foreground_max_bytes,
                             &put_epoch, &active_puts, &samples, &ok);
    }
    for (int tp = 0; tp < FLAGS_tp_size; ++tp) {
        TpEngineContext* store_context =
            FLAGS_split_store_engine ? &store_contexts[tp] : &pd_contexts[tp];
        threads.emplace_back(storeWorker, tp, std::cref(events), store_context,
                             replay_start, default_intent, pd_intent,
                             store_intent, transport, &put_epoch, &active_puts,
                             &samples, &ok);
    }
    for (auto& thread : threads) thread.join();

    auto close_context = [](TpEngineContext& context) {
        auto status = context.engine->closeSegment(context.target);
        if (!status.ok()) {
            LOG(WARNING) << "closeSegment failed for TP " << context.tp << ": "
                         << status.ToString();
        }
        unregisterAndFreeBuffers(*context.engine, &context.local_buffers,
                                 FLAGS_buffer_size);
    };
    for (auto& context : pd_contexts) {
        close_context(context);
    }
    for (auto& context : store_contexts) {
        close_context(context);
    }

    printSummary(events, samples, foreground_max_bytes);
    return ok.load(std::memory_order_acquire) ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace
}  // namespace tent
}  // namespace mooncake

int main(int argc, char** argv) {
    gflags::SetUsageMessage("SGLang 4P+4D TENT trace replay benchmark");
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    if (FLAGS_mode == "target") return mooncake::tent::runTarget();
    if (FLAGS_mode == "replay") return mooncake::tent::runReplay();
    LOG(ERROR) << "--mode must be target or replay";
    return EXIT_FAILURE;
}
