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

#include "utils.h"

#include "bench_runner.h"
#include "te_backend.h"
#include "tent_backend.h"

#include <atomic>
#include <thread>
#include <chrono>
#include <fstream>
#include <iomanip>

using namespace mooncake::tent;

// ============================================================================
// High-Precision Bandwidth Monitor
// ============================================================================
class BandwidthMonitor {
public:
    BandwidthMonitor() : running_(false), total_bytes_(0), start_time_ns_(0),
                        last_bytes_(0), last_time_ns_(0) {}

    void setOutputFile(const std::string& file) { output_file_ = file; }
    void setIntervalMs(uint64_t ms) { interval_ms_ = ms; }

    void start() {
        if (!output_file_.empty()) {
            ofs_.open(output_file_);
            if (ofs_.is_open()) {
                ofs_ << "timestamp_ms,bandwidth_gbps" << std::endl;
            }
        }
        running_ = true;
        start_time_ns_ = getCurrentTimeNs();
        last_time_ns_ = start_time_ns_;
        last_bytes_ = 0;
        monitor_thread_ = std::thread(&BandwidthMonitor::monitorLoop, this);
    }

    void stop() {
        running_ = false;
        if (monitor_thread_.joinable()) monitor_thread_.join();
        if (ofs_.is_open()) ofs_.close();
    }

    void addBytes(uint64_t bytes) { total_bytes_ += bytes; }

private:
    void monitorLoop() {
        while (running_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms_));

            uint64_t now_ns = getCurrentTimeNs();
            uint64_t bytes_delta = total_bytes_ - last_bytes_;
            uint64_t time_delta_ns = now_ns - last_time_ns_;

            double bw_gbps = 0;
            if (time_delta_ns > 0) {
                bw_gbps = (bytes_delta * 8.0) / (time_delta_ns * 1e-9 * 1e9);
            }

            uint64_t timestamp_ms = (now_ns - start_time_ns_) / 1000000;

            if (ofs_.is_open()) {
                ofs_ << timestamp_ms << "," << std::fixed << std::setprecision(3)
                     << bw_gbps << std::endl;
            }

            last_bytes_ = total_bytes_;
            last_time_ns_ = now_ns;
        }
    }

    static uint64_t getCurrentTimeNs() {
        auto ret = std::chrono::steady_clock::now().time_since_epoch();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(ret).count();
    }

    bool running_;
    std::atomic<uint64_t> total_bytes_;
    uint64_t start_time_ns_, last_bytes_, last_time_ns_;
    uint64_t interval_ms_ = 10;
    std::string output_file_;
    std::thread monitor_thread_;
    std::ofstream ofs_;
};

static std::unique_ptr<BandwidthMonitor> g_bw_monitor;

// ============================================================================
// Fault Injection (for testing quota scheduling)
// Hardcoded: fault at 1s, recover at 3s
// ============================================================================
extern "C" {
    // Per-device power state: 0=disabled, 50=half, 100=full
    static std::atomic<int> g_device_power[64] = {
        100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100, 100
    };

    void setDevicePower(int device_id, int percent) {
        if (device_id >= 0 && device_id < 64) {
            g_device_power[device_id].store(percent, std::memory_order_relaxed);
            LOG(INFO) << "[FaultInject] Dev" << device_id << " power=" << percent << "%";
        }
    }

    bool isDeviceDisabled(int device_id) {
        if (device_id < 0 || device_id >= 64) return false;
        return g_device_power[device_id].load(std::memory_order_relaxed) == 0;
    }
}

int processBatchSizes(BenchRunner& runner, size_t block_size, size_t batch_size,
                      int num_threads) {
    bool mixed_opcode = false;
    OpCode opcode = READ;
    if (XferBenchConfig::check_consistency || XferBenchConfig::op_type == "mix")
        mixed_opcode = true;
    else if (XferBenchConfig::op_type == "read")
        opcode = READ;
    else if (XferBenchConfig::op_type == "write")
        opcode = WRITE;
    else {
        LOG(ERROR) << "Invalid args: workload only support read|write|mix";
        exit(EXIT_FAILURE);
    }

    XferBenchStats stats;
    std::mutex mutex;

    // Start bandwidth monitor
    std::unique_ptr<BandwidthMonitor> monitor = std::make_unique<BandwidthMonitor>();
    monitor->setOutputFile(XferBenchConfig::bw_csv_output);
    monitor->setIntervalMs(XferBenchConfig::bw_interval_ms);
    monitor->start();
    LOG(INFO) << "Bandwidth monitoring: " << XferBenchConfig::bw_csv_output
              << " (interval=" << XferBenchConfig::bw_interval_ms << "ms)";

    int rc = runner.runInitiatorTasks([&](int thread_id) -> int {
        runner.pinThread(thread_id);
        auto max_block_size = XferBenchConfig::max_block_size;
        auto max_batch_size = XferBenchConfig::max_batch_size;
        auto local_gpu_offset = std::max(0, XferBenchConfig::local_gpu_id);
        auto target_gpu_offset = std::max(0, XferBenchConfig::target_gpu_id);
        uint64_t local_addr = runner.getLocalBufferBase(
            local_gpu_offset + thread_id, max_block_size, max_batch_size);
        uint64_t target_addr = runner.getTargetBufferBase(
            target_gpu_offset + thread_id, max_block_size, max_batch_size);

        XferBenchTimer timer;
        while (timer.lap_us(false) < 1000000ull) {
            runner.runSingleTransfer(local_addr, target_addr, block_size,
                                     batch_size, opcode);
        }
        timer.reset();
        std::vector<double> transfer_duration;
        if (mixed_opcode) {
            while (timer.lap_us(false) <
                   XferBenchConfig::duration * 1000000ull) {
                uint8_t pattern = 0;
                if (XferBenchConfig::check_consistency)
                    pattern =
                        fillData((void*)local_addr, block_size * batch_size);
                auto val = runner.runSingleTransfer(
                    local_addr, target_addr, block_size, batch_size, WRITE);
                transfer_duration.push_back(val);
                monitor->addBytes(block_size * batch_size);
                fillData((void*)local_addr, block_size * batch_size);
                val = runner.runSingleTransfer(local_addr, target_addr,
                                               block_size, batch_size, READ);
                if (XferBenchConfig::check_consistency)
                    verifyData((void*)local_addr, block_size * batch_size,
                               pattern);
                transfer_duration.push_back(val);
                monitor->addBytes(block_size * batch_size);
            }
        } else {
            while (timer.lap_us(false) <
                   XferBenchConfig::duration * 1000000ull) {
                auto val = runner.runSingleTransfer(
                    local_addr, target_addr, block_size, batch_size, opcode);
                transfer_duration.push_back(val);
                monitor->addBytes(block_size * batch_size);
            }
        }
        auto total_duration = timer.lap_us();
        mutex.lock();
        stats.total_duration.add(total_duration);
        for (auto val : transfer_duration) stats.transfer_duration.add(val);
        mutex.unlock();
        return 0;
    });

    monitor->stop();

    if (rc != 0) return -1;
    printStats(block_size, batch_size, stats, num_threads);
    return 0;
}

int main(int argc, char* argv[]) {
    gflags::SetUsageMessage(
        "Mooncake Transfer Engine Benchmarking Tool\n"
        "Usage: ./tebench [options]");
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    XferBenchConfig::loadFromFlags();
    std::unique_ptr<BenchRunner> runner;
    if (XferBenchConfig::backend == "classic")
        runner = std::make_unique<TEBenchRunner>();
    else
        runner = std::make_unique<TENTBenchRunner>();
    if (XferBenchConfig::target_seg_name.empty()) {
        std::cout << "\033[33mTo start initiators, run " << std::endl
                  << "  ./tebench --target_seg_name="
                  << runner->getSegmentName()
                  << " --seg_type=" << XferBenchConfig::seg_type
                  << " --backend=" << XferBenchConfig::backend << std::endl
                  << "Press Ctrl-C to terminate\033[0m" << std::endl;
        return runner->runTarget();
    }
    printStatsHeader();
    bool interrupted = false;

    // Fault injection: hardcode t=1s fault, t=3s recover
    // Note: power=0 disables device at quota layer (filtered)
    //       power=50..99 has no effect without workers.cpp integration
    std::thread fault_thread;
    if (XferBenchConfig::fault_power_percent < 100) {
        int dev_id = XferBenchConfig::fault_device_id;
        int power_percent = XferBenchConfig::fault_power_percent;
        fault_thread = std::thread([dev_id, power_percent]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            // Use power=0 for actual device disable (only mode that works via quota.cpp)
            setDevicePower(dev_id, 0);
            LOG(INFO) << "=== FAULT at t=1s: Dev" << dev_id << " DISABLED ===";
            std::this_thread::sleep_for(std::chrono::milliseconds(2000));
            setDevicePower(dev_id, 100);
            LOG(INFO) << "=== RECOVER at t=3s: Dev" << dev_id << " -> 100% ===";
        });
        LOG(INFO) << "Fault injection: Dev" << dev_id << " (1s->DISABLED, 3s->100%)";
    }
    for (int num_threads = XferBenchConfig::start_num_threads;
         !interrupted && num_threads <= XferBenchConfig::max_num_threads;
         num_threads *= 2) {
        runner->startInitiator(num_threads);
        for (size_t block_size = XferBenchConfig::start_block_size;
             !interrupted && block_size <= XferBenchConfig::max_block_size;
             block_size *= 2) {
            for (size_t batch_size = XferBenchConfig::start_batch_size;
                 !interrupted && batch_size <= XferBenchConfig::max_batch_size;
                 batch_size *= 2) {
                if (block_size * batch_size * num_threads >
                    XferBenchConfig::total_buffer_size) {
                    LOG(INFO) << "Skipped for block_size " << block_size
                              << " batch_size " << batch_size;
                } else {
                    if (processBatchSizes(*runner, block_size, batch_size,
                                          num_threads) != 0)
                        interrupted = true;
                }
            }
        }
        runner->stopInitiator();
    }

    if (fault_thread.joinable()) fault_thread.join();
    return 0;
}
