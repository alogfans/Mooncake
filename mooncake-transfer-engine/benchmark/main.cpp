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

#include <fstream>
#include <atomic>
#include <thread>
#include <chrono>
#include <iomanip>

using namespace mooncake::tent;

// Global fault injection state (accessed by TENT RDMA transport)
extern "C" {
    // Bitmask of disabled devices (bit N = 1 means device N is disabled)
    std::atomic<uint64_t> g_fault_device_mask{0};
}

// ============================================================================
// High-Precision Bandwidth Monitor
// ============================================================================

class BandwidthMonitor {
public:
    struct Sample {
        uint64_t timestamp_us;      // Microseconds since start
        uint64_t elapsed_ms;        // Elapsed time in milliseconds
        uint64_t total_bytes;       // Total bytes transferred
        double instantaneous_bw_gbps;  // Instantaneous bandwidth (GB/s)
        bool fault_active;          // Device fault state
        int active_devices;         // Number of active devices
    };

    BandwidthMonitor()
        : running_(false),
          total_bytes_(0),
          start_time_ns_(0),
          last_bytes_(0),
          last_sample_time_ns_(0),
          fault_active_(false),
          active_devices_(2),
          fault_time_ms_(1000),      // Default: fault at 1 second
          recovery_time_ms_(3000),   // Default: recover at 3 seconds
          sampling_interval_ms_(10)  // Default: 10ms sampling
    {}

    void setFaultTiming(uint64_t fault_ms, uint64_t recovery_ms) {
        fault_time_ms_ = fault_ms;
        recovery_time_ms_ = recovery_ms;
    }

    void setSamplingInterval(uint64_t interval_ms) {
        sampling_interval_ms_ = interval_ms;
    }

    void setActiveDevices(int num_devices) {
        active_devices_ = num_devices;
    }

    void setOutputFile(const std::string& filename) {
        output_file_ = filename;
    }

    void setFaultMode(const std::string& mode) {
        fault_mode_ = mode;
    }

    void start() {
        if (!output_file_.empty()) {
            // Open CSV file and write header
            ofs_.open(output_file_);
            if (ofs_.is_open()) {
                ofs_ << "timestamp_us,elapsed_ms,total_bytes,instantaneous_bw_gbps,"
                     << "fault_active,active_devices" << std::endl;
            }
        }

        running_ = true;
        start_time_ns_ = getCurrentTimeNs();
        last_sample_time_ns_ = start_time_ns_;
        last_bytes_ = 0;

        monitor_thread_ = std::thread(&BandwidthMonitor::monitorLoop, this);
    }

    void stop() {
        running_ = false;
        if (monitor_thread_.joinable()) {
            monitor_thread_.join();
        }
        if (ofs_.is_open()) {
            ofs_.close();
        }
    }

    void addBytes(uint64_t bytes) {
        total_bytes_ += bytes;
    }

    uint64_t getTotalBytes() const {
        return total_bytes_;
    }

    const std::vector<Sample>& getSamples() const {
        return samples_;
    }

    bool isFaultActive() const {
        return fault_active_;
    }

    int getActiveDevices() const {
        return active_devices_;
    }

    void printSummary() const {
        if (samples_.empty()) {
            std::cout << "[Monitor] No samples collected" << std::endl;
            return;
        }

        double total_gb = total_bytes_ / (1024.0 * 1024.0 * 1024.0);
        double total_sec = samples_.back().elapsed_ms / 1000.0;
        double avg_bw = 0;
        double max_bw = 0;
        double min_bw = std::numeric_limits<double>::max();

        size_t fault_count = 0;
        double fault_bw_sum = 0;
        double normal_bw_sum = 0;
        size_t normal_count = 0;

        for (const auto& s : samples_) {
            avg_bw += s.instantaneous_bw_gbps;
            max_bw = std::max(max_bw, s.instantaneous_bw_gbps);
            min_bw = std::min(min_bw, s.instantaneous_bw_gbps);

            if (s.fault_active) {
                fault_count++;
                fault_bw_sum += s.instantaneous_bw_gbps;
            } else {
                normal_count++;
                normal_bw_sum += s.instantaneous_bw_gbps;
            }
        }

        avg_bw /= samples_.size();
        double fault_avg = fault_count > 0 ? fault_bw_sum / fault_count : 0;
        double normal_avg = normal_count > 0 ? normal_bw_sum / normal_count : 0;

        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "HIGH-PRECISION BANDWIDTH MONITOR SUMMARY" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        std::cout << "Total Bytes:      " << std::fixed << std::setprecision(3)
                  << total_gb << " GB" << std::endl;
        std::cout << "Total Duration:   " << std::setprecision(2)
                  << total_sec << " seconds" << std::endl;
        std::cout << "Samples Collected: " << samples_.size() << std::endl;
        std::cout << "Sampling Interval: " << sampling_interval_ms_ << " ms" << std::endl;
        std::cout << std::endl;
        std::cout << "Average Bandwidth: " << std::setprecision(3)
                  << avg_bw << " GB/s" << std::endl;
        std::cout << "Peak Bandwidth:    " << max_bw << " GB/s" << std::endl;
        std::cout << "Minimum Bandwidth: " << min_bw << " GB/s" << std::endl;
        std::cout << std::endl;
        std::cout << "Normal Avg BW:    " << normal_avg << " GB/s ("
                  << normal_count << " samples)" << std::endl;
        std::cout << "Fault Avg BW:     " << fault_avg << " GB/s ("
                  << fault_count << " samples)" << std::endl;
        if (normal_avg > 0) {
            std::cout << "Degradation:      "
                      << std::setprecision(1)
                      << ((normal_avg - fault_avg) / normal_avg * 100.0)
                      << "%" << std::endl;
        }
        std::cout << std::string(70, '=') << std::endl;
    }

private:
    void monitorLoop() {
        while (running_) {
            auto current_time_ns = getCurrentTimeNs();
            auto elapsed_ns = current_time_ns - start_time_ns_;
            auto elapsed_ms = elapsed_ns / 1000000ULL;

            // Check for fault trigger
            if (elapsed_ms >= fault_time_ms_ && !fault_active_) {
                fault_active_ = true;
                LOG(WARNING) << "=== DEVICE FAULT SIMULATED: " << fault_mode_
                             << " at t=" << (elapsed_ms / 1000.0) << "s ===";
                if (fault_mode_ == "half_speed") {
                    active_devices_ = std::max(1, active_devices_ / 2);
                    // Disable device 0 for half bandwidth (only device 1 active)
                    g_fault_device_mask.store(0x1, std::memory_order_relaxed);  // Bit 0 = device 0
                } else if (fault_mode_ == "disconnect") {
                    active_devices_ = 1;
                    // Disable device 0 completely
                    g_fault_device_mask.store(0x1, std::memory_order_relaxed);  // Bit 0 = device 0
                }
            }

            // Check for recovery
            if (elapsed_ms >= recovery_time_ms_ && fault_active_) {
                fault_active_ = false;
                active_devices_ = 2;  // Restore to 2 devices
                LOG(INFO) << "=== DEVICE RECOVERY at t=" << (elapsed_ms / 1000.0) << "s ===";
                // Re-enable device 0
                g_fault_device_mask.store(0, std::memory_order_relaxed);  // Clear all faults
            }

            // Calculate instantaneous bandwidth
            uint64_t bytes_since_last = total_bytes_ - last_bytes_;
            uint64_t time_since_last_ns = current_time_ns - last_sample_time_ns_;

            double inst_bw_gbps = 0;
            if (time_since_last_ns > 0) {
                inst_bw_gbps = (bytes_since_last * 8.0) / (time_since_last_ns * 1e-9 * 1e9);
            }

            // Record sample
            Sample sample{
                elapsed_ns / 1000ULL,  // timestamp_us
                elapsed_ms,             // elapsed_ms
                total_bytes_,           // total_bytes
                inst_bw_gbps,           // instantaneous_bw_gbps
                fault_active_,          // fault_active
                active_devices_         // active_devices
            };
            samples_.push_back(sample);

            // Write to CSV
            if (ofs_.is_open()) {
                ofs_ << sample.timestamp_us << ","
                     << sample.elapsed_ms << ","
                     << sample.total_bytes << ","
                     << std::fixed << std::setprecision(6)
                     << sample.instantaneous_bw_gbps << ","
                     << (sample.fault_active ? 1 : 0) << ","
                     << sample.active_devices
                     << std::endl;
            }

            // Update trackers
            last_bytes_ = total_bytes_;
            last_sample_time_ns_ = current_time_ns;

            // Sleep for sampling interval
            std::this_thread::sleep_for(std::chrono::milliseconds(sampling_interval_ms_));
        }
    }

    static uint64_t getCurrentTimeNs() {
        auto ret = std::chrono::steady_clock::now().time_since_epoch();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(ret).count();
    }

    bool running_;
    std::atomic<uint64_t> total_bytes_;
    uint64_t start_time_ns_;
    uint64_t last_bytes_;
    uint64_t last_sample_time_ns_;
    bool fault_active_;
    int active_devices_;
    uint64_t fault_time_ms_;
    uint64_t recovery_time_ms_;
    uint64_t sampling_interval_ms_;
    std::string fault_mode_;
    std::string output_file_;
    std::vector<Sample> samples_;
    std::thread monitor_thread_;
    std::ofstream ofs_;
};

// Global bandwidth monitor instance
static std::unique_ptr<BandwidthMonitor> g_bandwidth_monitor;

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

    // Start bandwidth monitor if enabled
    if (g_bandwidth_monitor) {
        g_bandwidth_monitor->start();
    }

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

                // Update bandwidth monitor
                if (g_bandwidth_monitor) {
                    g_bandwidth_monitor->addBytes(block_size * batch_size);
                }

                fillData((void*)local_addr, block_size * batch_size);
                val = runner.runSingleTransfer(local_addr, target_addr,
                                               block_size, batch_size, READ);
                if (XferBenchConfig::check_consistency)
                    verifyData((void*)local_addr, block_size * batch_size,
                               pattern);
                transfer_duration.push_back(val);

                // Update bandwidth monitor
                if (g_bandwidth_monitor) {
                    g_bandwidth_monitor->addBytes(block_size * batch_size);
                }
            }
        } else {
            while (timer.lap_us(false) <
                   XferBenchConfig::duration * 1000000ull) {
                auto val = runner.runSingleTransfer(
                    local_addr, target_addr, block_size, batch_size, opcode);
                transfer_duration.push_back(val);

                // Update bandwidth monitor
                if (g_bandwidth_monitor) {
                    g_bandwidth_monitor->addBytes(block_size * batch_size);
                }
            }
        }
        auto total_duration = timer.lap_us();
        mutex.lock();
        stats.total_duration.add(total_duration);
        for (auto val : transfer_duration) stats.transfer_duration.add(val);
        mutex.unlock();
        return 0;
    });

    // Stop bandwidth monitor and print summary
    if (g_bandwidth_monitor) {
        g_bandwidth_monitor->stop();
        g_bandwidth_monitor->printSummary();
    }

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

    // Initialize high-precision bandwidth monitor if enabled
    if (!XferBenchConfig::bw_monitor_output.empty()) {
        g_bandwidth_monitor = std::make_unique<BandwidthMonitor>();
        g_bandwidth_monitor->setOutputFile(XferBenchConfig::bw_monitor_output);
        g_bandwidth_monitor->setSamplingInterval(XferBenchConfig::bw_monitor_interval_ms);
        g_bandwidth_monitor->setFaultTiming(XferBenchConfig::bw_monitor_fault_time_ms,
                                             XferBenchConfig::bw_monitor_recovery_time_ms);
        g_bandwidth_monitor->setFaultMode(XferBenchConfig::bw_monitor_fault_mode);

        LOG(INFO) << "High-precision bandwidth monitoring enabled:";
        LOG(INFO) << "  Output file: " << XferBenchConfig::bw_monitor_output;
        LOG(INFO) << "  Sampling interval: " << XferBenchConfig::bw_monitor_interval_ms << " ms";
        LOG(INFO) << "  Fault time: " << XferBenchConfig::bw_monitor_fault_time_ms << " ms";
        LOG(INFO) << "  Recovery time: " << XferBenchConfig::bw_monitor_recovery_time_ms << " ms";
        LOG(INFO) << "  Fault mode: " << XferBenchConfig::bw_monitor_fault_mode;
    }

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
    return 0;
}
