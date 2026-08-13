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

#include "trace_replay.h"

#include "bench_runner.h"
#include "utils.h"
#include "workload_config.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>

namespace mooncake {
namespace tent {

namespace {

constexpr uint64_t kTraceReportIntervalNs = 10000000000ull;
constexpr uint64_t kTraceStatsSkipNs = 10000000000ull;
constexpr size_t kTraceMetadataThreshold = 1ull << 20;

struct TraceRecord {
    uint64_t offset_ns = 0;
    size_t length = 0;
    std::string source;
};

struct TraceStatsGroup {
    XferBenchStats stats;
    size_t total_events = 0;
    uint64_t total_bytes = 0;
    size_t completed_events = 0;
    uint64_t completed_bytes = 0;
};

uint64_t steadyClockNs() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
}

double gbPerSecond(uint64_t bytes, double duration_us) {
    if (duration_us <= 0.0) return 0.0;
    return static_cast<double>(bytes) / (1000.0 * duration_us);
}

std::vector<std::string> splitCsvLine(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    std::stringstream ss(line);
    while (std::getline(ss, field, ',')) fields.push_back(field);
    if (!line.empty() && line.back() == ',') fields.emplace_back();
    return fields;
}

bool parseUint64(const std::string& value, uint64_t* out) {
    if (!out || value.empty()) return false;
    char* end = nullptr;
    errno = 0;
    unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0') return false;
    *out = static_cast<uint64_t>(parsed);
    return true;
}

bool parseTimestampNs(const std::string& value, uint64_t* out) {
    if (!out || value.size() < 19) return false;
    std::tm tm = {};
    std::istringstream ss(value.substr(0, 19));
    ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
    if (ss.fail()) return false;

    uint64_t micros = 0;
    if (value.size() > 19 && value[19] == '.') {
        std::string frac = value.substr(20);
        if (frac.size() > 6) frac.resize(6);
        while (frac.size() < 6) frac.push_back('0');
        if (!parseUint64(frac, &micros)) return false;
    }

    tm.tm_isdst = -1;
    time_t seconds = std::mktime(&tm);
    if (seconds < 0) return false;
    *out = static_cast<uint64_t>(seconds) * 1000000000ull + micros * 1000ull;
    return true;
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

bool loadTraceCsv(const std::string& path, std::vector<TraceRecord>* records,
                  uint64_t* trace_span_ns, std::string* error) {
    if (!records || !trace_span_ns) return false;
    std::ifstream input(path);
    if (!input.is_open()) {
        if (error) *error = "unable to open trace file: " + path;
        return false;
    }

    std::string header;
    if (!std::getline(input, header)) {
        if (error) *error = "trace file is empty: " + path;
        return false;
    }
    const auto header_fields = splitCsvLine(header);
    std::unordered_map<std::string, int> columns;
    for (size_t i = 0; i < header_fields.size(); ++i) {
        columns[header_fields[i]] = static_cast<int>(i);
    }
    const int timestamp_col = columnIndex(columns, "timestamp");
    const int length_col = columnIndex(columns, "length");
    if (timestamp_col < 0 || length_col < 0) {
        if (error) *error = "trace CSV must contain timestamp and length columns";
        return false;
    }
    const int source_col = columnIndex(columns, "source");
    if ((!XferBenchConfig::trace_source_filter.empty() ||
         !XferBenchConfig::trace_source_intents.empty()) &&
        source_col < 0) {
        if (error)
            *error = "trace source filtering/intent mapping requires a source "
                     "column in trace CSV";
        return false;
    }

    records->clear();
    uint64_t first_ts = 0;
    uint64_t last_ts = 0;
    std::string line;
    size_t line_no = 1;
    while (std::getline(input, line)) {
        ++line_no;
        if (line.empty()) continue;
        const auto fields = splitCsvLine(line);
        const std::string source = fieldAt(fields, source_col);
        if (!XferBenchConfig::trace_source_filter.empty() &&
            source != XferBenchConfig::trace_source_filter)
            continue;

        uint64_t ts = 0;
        if (!parseTimestampNs(fieldAt(fields, timestamp_col), &ts)) {
            if (error)
                *error = "invalid timestamp at line " + std::to_string(line_no);
            return false;
        }
        uint64_t length = 0;
        if (!parseUint64(fieldAt(fields, length_col), &length) || length == 0) {
            if (error)
                *error = "invalid length at line " + std::to_string(line_no);
            return false;
        }
        if (records->empty()) first_ts = ts;
        if (ts < first_ts) {
            if (error)
                *error = "trace timestamps must be nondecreasing at line " +
                         std::to_string(line_no);
            return false;
        }

        records->push_back(
            TraceRecord{ts - first_ts, static_cast<size_t>(length), source});
        last_ts = ts;
    }

    if (records->empty()) {
        if (error) {
            *error = "trace file contains no records";
            if (!XferBenchConfig::trace_source_filter.empty()) {
                *error += " matching source filter '" +
                          XferBenchConfig::trace_source_filter + "'";
            }
            *error += ": " + path;
        }
        return false;
    }
    *trace_span_ns = last_ts - first_ts;
    return true;
}

OpCode traceOpcode() {
    if (XferBenchConfig::op_type == "write") return WRITE;
    return READ;
}

uint64_t traceDeadlineNs() {
    if (XferBenchConfig::deadline_us == 0) return 0;
    return steadyClockNs() + XferBenchConfig::deadline_us * 1000ull;
}

bool parseTraceSourceIntents(std::unordered_map<std::string, IntentType>* intents,
                             std::string* error) {
    if (!intents) return false;
    intents->clear();
    if (XferBenchConfig::trace_source_intents.empty()) return true;

    std::stringstream ss(XferBenchConfig::trace_source_intents);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        const auto sep = item.find(':');
        if (sep == std::string::npos || sep == 0 || sep + 1 >= item.size()) {
            if (error)
                *error = "invalid --trace_source_intents entry: " + item;
            return false;
        }
        const std::string source = item.substr(0, sep);
        const std::string intent_name = item.substr(sep + 1);
        IntentType intent = IntentType::INTENT_UNSPEC;
        if (!parseBenchIntentType(intent_name, &intent)) {
            if (error)
                *error = "invalid intent '" + intent_name +
                         "' in --trace_source_intents";
            return false;
        }
        (*intents)[source] = intent;
    }
    return true;
}

IntentType intentForTraceRecord(
    const TraceRecord& record,
    const std::unordered_map<std::string, IntentType>& source_intents,
    IntentType default_intent) {
    auto it = source_intents.find(record.source);
    return it == source_intents.end() ? default_intent : it->second;
}

const char* traceIntentTypeName(IntentType intent) {
    switch (intent) {
        case IntentType::INTENT_UNSPEC:
            return "unspec";
        case IntentType::FOREGROUND_GET:
            return "foreground_get";
        case IntentType::BACKGROUND_PREFETCH:
            return "background_prefetch";
        case IntentType::MIGRATION:
            return "migration";
        case IntentType::CHECKPOINT:
            return "checkpoint";
        case IntentType::WEIGHT_LOADING:
            return "weight_loading";
        case IntentType::STAGING_INTERNAL:
            return "staging_internal";
    }
    return "unknown";
}

std::string describeTraceSourceIntents(
    const std::unordered_map<std::string, IntentType>& source_intents) {
    if (source_intents.empty()) return "<none>";
    std::vector<std::string> entries;
    entries.reserve(source_intents.size());
    for (const auto& kv : source_intents) {
        entries.push_back(kv.first + ":" + traceIntentTypeName(kv.second));
    }
    std::sort(entries.begin(), entries.end());
    std::ostringstream os;
    for (size_t i = 0; i < entries.size(); ++i) {
        if (i > 0) os << ',';
        os << entries[i];
    }
    return os.str();
}

bool shouldCollectTraceStats(const TraceRecord& record) {
    return record.offset_ns >= kTraceStatsSkipNs;
}

void printTraceStats(const char* label, size_t completed, size_t total_events,
                     uint64_t completed_bytes, XferBenchStats& stats) {
    std::cout << std::fixed << std::setprecision(6) << label
              << " events=" << completed << '/' << total_events
              << " bytes=" << completed_bytes
              << " wall_time_us(p50/p95/p99)="
              << stats.transfer_duration.p50() << '/'
              << stats.transfer_duration.p95() << '/'
              << stats.transfer_duration.p99()
              << " instant_GB/s(p50/p95/p99)="
              << stats.instant_bandwidth.p50() << '/'
              << stats.instant_bandwidth.p95() << '/'
              << stats.instant_bandwidth.p99() << std::endl;
}

std::string groupForTraceRecord(const TraceRecord& record) {
    if (record.source != "batch_transfer_sync") return record.source;
    return record.length < kTraceMetadataThreshold
               ? "batch_transfer_sync.metadata"
               : "batch_transfer_sync.data";
}

void addTraceSample(TraceStatsGroup& group, size_t bytes, double latency_us,
                    double instant_gbps) {
    group.completed_events++;
    group.completed_bytes += bytes;
    group.stats.transfer_duration.add(latency_us);
    group.stats.instant_bandwidth.add(instant_gbps);
}

void addTraceExpected(TraceStatsGroup& group, size_t bytes) {
    group.total_events++;
    group.total_bytes += bytes;
}

void printTraceStatsGroups(
    const char* label, const std::vector<std::string>& group_order,
    std::unordered_map<std::string, TraceStatsGroup>& groups) {
    auto overall = groups.find("overall");
    if (overall != groups.end()) {
        printTraceStats(label, overall->second.completed_events,
                        overall->second.total_events,
                        overall->second.completed_bytes, overall->second.stats);
    }
    for (const auto& name : group_order) {
        auto it = groups.find(name);
        if (it == groups.end() || it->second.completed_events == 0) continue;
        std::string group_label = std::string(label) + "[" + name + "]";
        printTraceStats(group_label.c_str(), it->second.completed_events,
                        it->second.total_events, it->second.completed_bytes,
                        it->second.stats);
    }
}

}  // namespace

int processTraceReplay(BenchRunner& runner, int num_threads) {
    std::vector<TraceRecord> records;
    uint64_t trace_span_ns = 0;
    std::string error;
    if (!loadTraceCsv(XferBenchConfig::trace_file, &records, &trace_span_ns,
                      &error)) {
        LOG(ERROR) << error;
        return -1;
    }

    size_t max_length = 0;
    for (const auto& record : records) {
        max_length = std::max(max_length, record.length);
    }
    if (num_threads != 1) {
        LOG(ERROR) << "trace replay uses a single submit/poll thread";
        return -1;
    }
    const size_t slot_count = 1;
    if (max_length > XferBenchConfig::total_buffer_size) {
        LOG(ERROR) << "trace replay requires total_buffer_size >= "
                   << max_length << " bytes for max trace length "
                   << max_length;
        return -1;
    }

    IntentType default_intent = IntentType::INTENT_UNSPEC;
    if (!parseBenchIntentType(XferBenchConfig::tent_intent_type,
                              &default_intent)) {
        LOG(ERROR) << "Invalid --tent_intent_type="
                   << XferBenchConfig::tent_intent_type;
        return -1;
    }
    std::unordered_map<std::string, IntentType> source_intents;
    if (!parseTraceSourceIntents(&source_intents, &error)) {
        LOG(ERROR) << error;
        return -1;
    }

    std::unordered_map<std::string, TraceStatsGroup> stats_groups;
    stats_groups.emplace("overall", TraceStatsGroup{});
    std::vector<std::string> group_order;
    std::unordered_map<std::string, bool> seen_groups;
    size_t stats_window_skipped_events = 0;
    uint64_t stats_window_skipped_bytes = 0;
    for (const auto& record : records) {
        if (!shouldCollectTraceStats(record)) {
            stats_window_skipped_events++;
            stats_window_skipped_bytes += record.length;
            continue;
        }
        const std::string group = groupForTraceRecord(record);
        if (!seen_groups[group]) {
            seen_groups[group] = true;
            group_order.push_back(group);
            stats_groups.emplace(group, TraceStatsGroup{});
        }
    }

    const OpCode opcode = traceOpcode();
    std::cout << "Trace replay: file=" << XferBenchConfig::trace_file
              << " source_filter="
              << (XferBenchConfig::trace_source_filter.empty()
                      ? "<none>"
                      : XferBenchConfig::trace_source_filter)
              << " default_intent=" << traceIntentTypeName(default_intent)
              << " source_intents=" << describeTraceSourceIntents(source_intents)
              << " records=" << records.size()
              << " submit_threads=" << num_threads
              << " buffer_slots=" << slot_count << " max_length=" << max_length
              << " trace_span_s=" << trace_span_ns / 1e9
              << " stats_skip_s=" << kTraceStatsSkipNs / 1e9
              << " stats_window_records="
              << records.size() - stats_window_skipped_events
              << " stats_window_skipped_records="
              << stats_window_skipped_events
              << " stats_window_skipped_bytes=" << stats_window_skipped_bytes
              << " max_outstanding_by_buffer=" << slot_count
              << " delayed_when_inflight=1"
              << std::endl;

    size_t delayed_events = 0;
    uint64_t delayed_bytes = 0;

    const int rc = runner.runInitiatorTasks([&](int thread_id) -> int {
        runner.pinThread(thread_id);
        const uint64_t local_base =
            runner.getLocalBufferBase(thread_id, max_length, slot_count);
        const uint64_t target_base =
            runner.getTargetBufferBase(thread_id, max_length, slot_count);

        const uint64_t start_ns = steadyClockNs();
        uint64_t next_report_ns = start_ns + kTraceReportIntervalNs;
        auto maybeReportProgress = [&](uint64_t now_ns) {
            if (now_ns < next_report_ns) return;
            printTraceStatsGroups("[trace-progress]", group_order,
                                  stats_groups);
            do {
                next_report_ns += kTraceReportIntervalNs;
            } while (now_ns >= next_report_ns);
        };

        for (size_t event = 0; event < records.size(); ++event) {
            const auto& record = records[event];
            const uint64_t scheduled_ns = start_ns + record.offset_ns;
            uint64_t now_ns = steadyClockNs();
            const bool delayed_by_backpressure = now_ns > scheduled_ns;
            while (now_ns < scheduled_ns) {
                maybeReportProgress(now_ns);
                std::this_thread::yield();
                now_ns = steadyClockNs();
            }

            if (delayed_by_backpressure) {
                delayed_events++;
                delayed_bytes += record.length;
            }

            const uint64_t local_addr = local_base;
            const uint64_t target_addr = target_base;
            uint64_t batch_id = 0;
            const uint64_t issue_ns = steadyClockNs();
            const IntentType intent =
                intentForTraceRecord(record, source_intents, default_intent);
            if (!runner.submitTraceTransfer(local_addr, target_addr,
                                            record.length, opcode,
                                            traceDeadlineNs(), intent,
                                            &batch_id)) {
                return -1;
            }

            const bool collect_stats = shouldCollectTraceStats(record);
            const std::string group = groupForTraceRecord(record);
            if (collect_stats) {
                addTraceExpected(stats_groups["overall"], record.length);
                addTraceExpected(stats_groups[group], record.length);
            }

            while (true) {
                bool completed = false;
                if (!runner.pollTraceTransfer(batch_id, &completed)) {
                    return -1;
                }
                if (completed) break;
                maybeReportProgress(steadyClockNs());
            }

            const uint64_t complete_ns = steadyClockNs();
            const double latency_us =
                static_cast<double>(complete_ns - issue_ns) / 1000.0;
            const double instant_gbps =
                gbPerSecond(record.length, latency_us);
            runner.freeTraceTransfer(batch_id);
            if (collect_stats) {
                addTraceSample(stats_groups["overall"], record.length,
                               latency_us, instant_gbps);
                addTraceSample(stats_groups[group], record.length, latency_us,
                               instant_gbps);
            }
            maybeReportProgress(complete_ns);
        }
        return 0;
    });
    if (rc != 0) return -1;

    printTraceStatsGroups("[trace-summary]", group_order, stats_groups);
    std::cout << "Trace replay delayed: events=" << delayed_events
              << " bytes=" << delayed_bytes << std::endl;

    return 0;
}

}  // namespace tent
}  // namespace mooncake
