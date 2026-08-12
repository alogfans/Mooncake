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
constexpr size_t kTraceMetadataThreshold = 1ull << 20;

struct TraceRecord {
    uint64_t offset_ns = 0;
    size_t length = 0;
    std::string source;
};

struct InflightTraceTransfer {
    size_t slot = 0;
    uint64_t batch_id = 0;
    size_t length = 0;
    uint64_t issue_ns = 0;
    std::string group;
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

void printTraceStats(const char* label, size_t completed, size_t total_events,
                     uint64_t completed_bytes, XferBenchStats& stats) {
    std::cout << std::fixed << std::setprecision(6) << label
              << " events=" << completed << '/' << total_events
              << " bytes=" << completed_bytes
              << " wall_time_us(avg/p99/p999)="
              << stats.transfer_duration.avg() << '/'
              << stats.transfer_duration.p99() << '/'
              << stats.transfer_duration.p999()
              << " instant_GB/s(avg/p99/p999)="
              << stats.instant_bandwidth.avg() << '/'
              << stats.instant_bandwidth.p99() << '/'
              << stats.instant_bandwidth.p999() << std::endl;
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
    const size_t slot_count = std::min<size_t>(
        records.size(), XferBenchConfig::total_buffer_size / max_length);
    if (slot_count == 0) {
        LOG(ERROR) << "trace replay requires total_buffer_size >= "
                   << max_length << " bytes for max trace length " << max_length;
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

    const OpCode opcode = traceOpcode();
    std::cout << "Trace replay: file=" << XferBenchConfig::trace_file
              << " source_filter="
              << (XferBenchConfig::trace_source_filter.empty()
                      ? "<none>"
                      : XferBenchConfig::trace_source_filter)
              << " source_intents="
              << (XferBenchConfig::trace_source_intents.empty()
                      ? "<none>"
                      : XferBenchConfig::trace_source_intents)
              << " records=" << records.size() << " submit_threads=1"
              << " buffer_slots=" << slot_count << " max_length=" << max_length
              << " trace_span_s=" << trace_span_ns / 1e9
              << " max_outstanding_by_buffer=" << slot_count << std::endl;

    std::unordered_map<std::string, TraceStatsGroup> stats_groups;
    stats_groups.emplace("overall", TraceStatsGroup{});
    std::vector<std::string> group_order;
    std::unordered_map<std::string, bool> seen_groups;
    for (const auto& record : records) {
        stats_groups["overall"].total_events++;
        stats_groups["overall"].total_bytes += record.length;
        const std::string group = groupForTraceRecord(record);
        if (!seen_groups[group]) {
            seen_groups[group] = true;
            group_order.push_back(group);
            stats_groups.emplace(group, TraceStatsGroup{});
        }
        stats_groups[group].total_events++;
        stats_groups[group].total_bytes += record.length;
    }

    const int rc = runner.runInitiatorTasks([&](int thread_id) -> int {
        runner.pinThread(thread_id);
        const uint64_t local_base =
            runner.getLocalBufferBase(thread_id, max_length, slot_count);
        const uint64_t target_base =
            runner.getTargetBufferBase(thread_id, max_length, slot_count);
        const uint64_t start_ns = steadyClockNs();
        uint64_t next_report_ns = start_ns + kTraceReportIntervalNs;
        size_t next_event = 0;
        std::vector<size_t> free_slots;
        free_slots.reserve(slot_count);
        for (size_t slot = 0; slot < slot_count; ++slot) {
            free_slots.push_back(slot_count - 1 - slot);
        }
        std::vector<InflightTraceTransfer> inflight;
        inflight.reserve(slot_count);

        while (true) {
            bool made_progress = false;
            uint64_t now_ns = steadyClockNs();

            while (next_event < records.size() && !free_slots.empty()) {
                const auto& record = records[next_event];
                const uint64_t scheduled_ns = start_ns + record.offset_ns;
                if (now_ns < scheduled_ns) break;

                const size_t slot = free_slots.back();
                free_slots.pop_back();
                const uint64_t local_addr = local_base + slot * max_length;
                const uint64_t target_addr = target_base + slot * max_length;
                uint64_t batch_id = 0;
                const uint64_t issue_ns = steadyClockNs();
                if (!runner.submitTraceTransfer(local_addr, target_addr,
                                                record.length, opcode,
                                                traceDeadlineNs(),
                                                intentForTraceRecord(
                                                    record, source_intents,
                                                    default_intent),
                                                &batch_id)) {
                    return -1;
                }
                inflight.push_back(InflightTraceTransfer{
                    slot, batch_id, record.length, issue_ns,
                    groupForTraceRecord(record)});
                ++next_event;
                made_progress = true;
                now_ns = steadyClockNs();
            }

            for (size_t i = 0; i < inflight.size();) {
                bool completed = false;
                if (!runner.pollTraceTransfer(inflight[i].batch_id, &completed))
                    return -1;
                if (!completed) {
                    ++i;
                    continue;
                }
                const uint64_t complete_ns = steadyClockNs();
                const double latency_us =
                    static_cast<double>(complete_ns - inflight[i].issue_ns) /
                    1000.0;
                const double instant_gbps =
                    gbPerSecond(inflight[i].length, latency_us);
                runner.freeTraceTransfer(inflight[i].batch_id);
                addTraceSample(stats_groups["overall"], inflight[i].length,
                               latency_us, instant_gbps);
                addTraceSample(stats_groups[inflight[i].group],
                               inflight[i].length, latency_us, instant_gbps);
                free_slots.push_back(inflight[i].slot);
                inflight[i] = inflight.back();
                inflight.pop_back();
                made_progress = true;
            }

            now_ns = steadyClockNs();
            if (now_ns >= next_report_ns) {
                printTraceStatsGroups("[trace-progress]", group_order,
                                      stats_groups);
                do {
                    next_report_ns += kTraceReportIntervalNs;
                } while (now_ns >= next_report_ns);
            }

            if (next_event >= records.size() && inflight.empty()) break;
            if (!made_progress) {
                std::this_thread::yield();
            }
        }

        printTraceStatsGroups("[trace-summary]", group_order, stats_groups);
        return 0;
    });
    if (rc != 0) return -1;

    return 0;
}

}  // namespace tent
}  // namespace mooncake
