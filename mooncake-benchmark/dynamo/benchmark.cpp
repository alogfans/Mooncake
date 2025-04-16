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

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <signal.h>
#include <sys/time.h>
#include <boost/asio.hpp>

#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>

#include "nixl.h"

using namespace boost::asio;
using ip::tcp;

DEFINE_string(metadata_server, "192.168.3.77:2379", "etcd server host address");
DEFINE_string(mode, "initiator",
              "Running mode: initiator or target. Initiator node read/write "
              "data blocks from target node");
DEFINE_string(operation, "read", "Operation type: read or write");

DEFINE_string(segment_id, "192.168.3.76", "Segment ID to access data");
DEFINE_uint64(buffer_size, 1ull << 30, "total size of data buffer");
DEFINE_int32(batch_size, 128, "Batch size");
DEFINE_uint64(block_size, 4096, "Block size for each transfer request");
DEFINE_int32(duration, 10, "Test duration in seconds");
DEFINE_int32(threads, 4, "Task submission threads");
DEFINE_string(report_unit, "GB", "Report unit: GB|GiB|Gb|MB|MiB|Mb|KB|KiB|Kb");
DEFINE_uint32(report_precision, 2, "Report precision");

const static std::unordered_map<std::string, uint64_t> RATE_UNIT_MP = {
    {"GB", 1000ull * 1000ull * 1000ull},
    {"GiB", 1ull << 30},
    {"Gb", 1000ull * 1000ull * 1000ull / 8},
    {"MB", 1000ull * 1000ull},
    {"MiB", 1ull << 20},
    {"Mb", 1000ull * 1000ull / 8},
    {"KB", 1000ull},
    {"KiB", 1ull << 10},
    {"Kb", 1000ull / 8}};

static inline std::string calculateRate(uint64_t data_bytes, double duration) {
    if (std::fabs(duration) < 1e-10) {
        LOG(ERROR) << "Invalid args: duration shouldn't be 0";
        return "";
    }
    if (!RATE_UNIT_MP.count(FLAGS_report_unit)) {
        LOG(WARNING) << "Invalid flag: report_unit only support "
                        "GB|GiB|Gb|MB|MiB|Mb|KB|KiB|Kb, not support "
                     << FLAGS_report_unit
                     << " . Now use GB(default) as report_unit";
        FLAGS_report_unit = "GB";
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(FLAGS_report_precision)
        << 1.0 * data_bytes / duration / RATE_UNIT_MP.at(FLAGS_report_unit)
        << " " << FLAGS_report_unit << "/s";
    return oss.str();
}

volatile bool running = true;
std::atomic<size_t> total_batch_count(0);

std::string agent1("Agent001");
std::string agent2("Agent002");

bool equal_buf(void* buf1, void* buf2, size_t len) {
    // Do some checks on the data.
    for (size_t i = 0; i < len; i++)
        if (((uint8_t*)buf1)[i] != ((uint8_t*)buf2)[i]) return false;
    return true;
}

nixlAgentConfig cfg(true);
nixlAgent agent_initiator("initiator", cfg);
nixlAgent agent_target("target", cfg);
nixl_opt_args_t extra_params;

void initiatorWorker(int thread_id, void* local_addr, uint64_t remote_base) {
    auto &agent = agent_initiator;
    nixl_xfer_op_t opcode;
    if (FLAGS_operation == "read")
        opcode = NIXL_READ;
    else if (FLAGS_operation == "write")
        opcode = NIXL_WRITE;
    else {
        LOG(ERROR) << "Unsupported operation: must be 'read' or 'write'";
        exit(EXIT_FAILURE);
    }

    size_t batch_count = 0;
    while (running) {
        nixl_xfer_dlist_t req_src_descs(DRAM_SEG);
        nixl_xfer_dlist_t req_dst_descs(DRAM_SEG);

        for (int i = 0; i < FLAGS_batch_size; ++i) {
            nixlBasicDesc req_src;
            req_src.addr = (uintptr_t)(local_addr) +
                           FLAGS_block_size * (i * FLAGS_threads + thread_id);
            req_src.len = FLAGS_block_size;
            req_src.devId = 0;
            req_src_descs.addDesc(req_src);

            nixlBasicDesc req_dst;
            req_dst.addr = remote_base +
                           FLAGS_block_size * (i * FLAGS_threads + thread_id);
            req_dst.len = FLAGS_block_size;
            req_dst.devId = 0;
            req_dst_descs.addDesc(req_dst);
        }

        nixlXferReqH* req_handle;
        auto ret = agent.createXferReq(opcode, req_src_descs, req_dst_descs,
                                       "target", req_handle, &extra_params);

        assert(ret == NIXL_SUCCESS);

        nixl_status_t status = agent.postXferReq(req_handle);
        while (status != NIXL_SUCCESS) {
            status = agent.getXferStatus(req_handle);
            assert(status >= 0);
        }

        agent.releaseXferReq(req_handle);
        batch_count++;
    }

    LOG(INFO) << "Worker " << thread_id << " stopped!";
    total_batch_count.fetch_add(batch_count);
}

int initiator() {
#if 0
    nixl_status_t ret1, ret2;
    std::string ret_s1, ret_s2;

    // Example: assuming two agents running on the same machine,
    // with separate memory regions in DRAM

    nixlAgentConfig cfg(true);
    nixl_b_params_t init1, init2;
    nixl_mem_list_t mems1, mems2;

    // populate required/desired inits
    nixlAgent A1(agent1, cfg);
    nixlAgent A2(agent2, cfg);

    std::vector<nixl_backend_t> plugins;

    ret1 = A1.getAvailPlugins(plugins);
    assert(ret1 == NIXL_SUCCESS);

    std::cout << "Available plugins:\n";

    for (nixl_backend_t b : plugins) std::cout << b << "\n";

    nixlBackendH *ucx1, *ucx2;
    ret1 = A1.createBackend("UCX", init1, ucx1);
    ret2 = A2.createBackend("UCX", init2, ucx2);

    nixl_opt_args_t extra_params1, extra_params2;
    extra_params1.backends.push_back(ucx1);
    extra_params2.backends.push_back(ucx2);

    assert(ret1 == NIXL_SUCCESS);
    assert(ret2 == NIXL_SUCCESS);

    nixlBlobDesc buff1, buff2, buff3;
    nixl_reg_dlist_t dlist1(DRAM_SEG), dlist2(DRAM_SEG);
    size_t len = 256;
    void* addr1 = calloc(1, len);
    void* addr2 = calloc(1, len);

    memset(addr1, 0xbb, len);
    memset(addr2, 0, len);

    buff1.addr = (uintptr_t)addr1;
    buff1.len = len;
    buff1.devId = 0;
    dlist1.addDesc(buff1);

    buff2.addr = (uintptr_t)addr2;
    buff2.len = len;
    buff2.devId = 0;
    dlist2.addDesc(buff2);

    // sets the metadata field to a pointer to an object inside the ucx_class
    ret1 = A1.registerMem(dlist1, &extra_params1);
    ret2 = A2.registerMem(dlist2, &extra_params2);

    assert(ret1 == NIXL_SUCCESS);
    assert(ret2 == NIXL_SUCCESS);

    std::string meta1;
    ret1 = A1.getLocalMD(meta1);
    std::string meta2;
    ret2 = A2.getLocalMD(meta2);

    assert(ret1 == NIXL_SUCCESS);
    assert(ret2 == NIXL_SUCCESS);

    ret1 = A1.loadRemoteMD(meta2, ret_s1);

    assert(ret1 == NIXL_SUCCESS);
    assert(ret2 == NIXL_SUCCESS);

    size_t req_size = 8;
    size_t dst_offset = 8;

    nixl_xfer_dlist_t req_src_descs(DRAM_SEG);
    nixlBasicDesc req_src;
    req_src.addr = (uintptr_t)(((char*)addr1) + 16);  // random offset
    req_src.len = req_size;
    req_src.devId = 0;
    req_src_descs.addDesc(req_src);

    nixl_xfer_dlist_t req_dst_descs(DRAM_SEG);
    nixlBasicDesc req_dst;
    req_dst.addr = (uintptr_t)((char*)addr2) + dst_offset;  // random offset
    req_dst.len = req_size;
    req_dst.devId = 0;
    req_dst_descs.addDesc(req_dst);

    std::cout << "Transfer request from " << addr1 << " to " << addr2 << "\n";
    nixlXferReqH* req_handle;

    ret1 = A1.createXferReq(NIXL_WRITE, req_src_descs, req_dst_descs, agent2,
                            req_handle, &extra_params1);

    assert(ret1 == NIXL_SUCCESS);

    nixl_status_t status = A1.postXferReq(req_handle);

    std::cout << "Transfer was posted\n";
    while (status != NIXL_SUCCESS) {
        status = A1.getXferStatus(req_handle);
        assert(status >= 0);
    }

    std::cout << equal_buf(addr1, addr2, len) << "\n";

    ret1 = A1.releaseXferReq(req_handle);
    assert(ret1 == NIXL_SUCCESS);

    ret1 = A1.deregisterMem(dlist1, &extra_params1);
    ret2 = A2.deregisterMem(dlist2, &extra_params2);
    assert(ret1 == NIXL_SUCCESS);
    assert(ret2 == NIXL_SUCCESS);

    // only initiator should call invalidate
    ret1 = A1.invalidateRemoteMD(agent2);
    assert(ret1 == NIXL_SUCCESS);

    free(addr1);
    free(addr2);

    std::cout << "Test done\n";
#endif
    auto &agent = agent_initiator;
    nixl_b_params_t init;
    nixl_mem_list_t mems;
    std::vector<nixl_backend_t> plugins;
    auto ret = agent.getAvailPlugins(plugins);
    assert(ret == NIXL_SUCCESS);
    nixlBackendH* backend;
    ret = agent.createBackend("UCX", init, backend);
    assert(ret == NIXL_SUCCESS);
    extra_params.backends.push_back(backend);

    nixl_reg_dlist_t dlist(DRAM_SEG);
    void* addr = malloc(FLAGS_buffer_size);
    nixlBlobDesc buff;
    buff.addr = (uintptr_t)addr;
    buff.len = FLAGS_buffer_size;
    buff.devId = 0;
    dlist.addDesc(buff);

    ret = agent.registerMem(dlist, &extra_params);
    assert(ret == NIXL_SUCCESS);

    std::string meta;
    ret = agent.getLocalMD(meta);
    assert(ret == NIXL_SUCCESS);

    const short port = 12345;
    std::vector<char> buf(4096);
    try {
        io_context io;
        tcp::socket socket(io);
        socket.connect(tcp::endpoint(ip::make_address(FLAGS_segment_id), port));
        read(socket, buffer(buf), transfer_exactly(4096));
        
        socket.shutdown(tcp::socket::shutdown_both);
        socket.close();
    } catch (std::exception& e) {
        std::cerr << "Initiator error: " << e.what() << std::endl;
    }

    std::string target_meta, ret_string;
    uint64_t remote_base;

    remote_base = *(uint64_t *) ((char *) buf.data());
    auto target_meta_length = *(uint64_t *) ((char *) buf.data() + sizeof(uint64_t));
    LOG(INFO) << (void *) remote_base << " " << target_meta_length;
    target_meta = std::string((char *) buf.data() + 2 * sizeof(uint64_t), target_meta_length);

    ret = agent.loadRemoteMD(target_meta, ret_string);
    assert(ret == NIXL_SUCCESS);

    std::thread workers[FLAGS_threads];

    struct timeval start_tv, stop_tv;
    gettimeofday(&start_tv, nullptr);

    for (int i = 0; i < FLAGS_threads; ++i)
        workers[i] = std::thread(initiatorWorker, i, addr, remote_base);

    sleep(FLAGS_duration);
    running = false;

    for (int i = 0; i < FLAGS_threads; ++i) workers[i].join();

    gettimeofday(&stop_tv, nullptr);
    auto duration = (stop_tv.tv_sec - start_tv.tv_sec) +
                    (stop_tv.tv_usec - start_tv.tv_usec) / 1000000.0;
    auto batch_count = total_batch_count.load();

    LOG(INFO) << "Test completed: duration " << std::fixed
              << std::setprecision(2) << duration << ", batch count "
              << batch_count << ", throughput "
              << calculateRate(
                     batch_count * FLAGS_batch_size * FLAGS_block_size,
                     duration);

    free(addr);
    return 0;
}

volatile bool target_running = true;

int target() {
    auto &agent = agent_target;
    nixl_b_params_t init;
    nixl_mem_list_t mems;
    std::vector<nixl_backend_t> plugins;
    auto ret = agent.getAvailPlugins(plugins);
    assert(ret == NIXL_SUCCESS);
    nixlBackendH* backend;
    ret = agent.createBackend("UCX", init, backend);
    assert(ret == NIXL_SUCCESS);
    nixl_opt_args_t extra_params;
    extra_params.backends.push_back(backend);

    nixl_reg_dlist_t dlist(DRAM_SEG);
    void* addr = malloc(FLAGS_buffer_size);
    nixlBlobDesc buff;
    buff.addr = (uintptr_t)addr;
    buff.len = FLAGS_buffer_size;
    buff.devId = 0;
    dlist.addDesc(buff);

    ret = agent.registerMem(dlist, &extra_params);
    assert(ret == NIXL_SUCCESS);

    std::string meta;
    ret = agent.getLocalMD(meta);
    assert(ret == NIXL_SUCCESS);

    const short port = 12345;
    try {
        io_context io;
        tcp::acceptor acceptor(io, tcp::endpoint(tcp::v4(), port));
        
        LOG(INFO) << "Start listening";

        tcp::socket socket(io);
        acceptor.accept(socket);

        std::vector<char> data(4096, 0);
        size_t meta_size = meta.size();
        memcpy(data.data(), &addr, sizeof(uint64_t));
        memcpy((char *) data.data() + sizeof(uint64_t), &meta_size, sizeof(uint64_t));
        memcpy((char *) data.data() + 2 * sizeof(uint64_t), meta.data(), meta_size);
        LOG(INFO) << addr << " " << meta_size;
        write(socket, buffer(data));
        
        socket.shutdown(tcp::socket::shutdown_both);
        socket.close();
    } catch (std::exception& e) {
        std::cerr << "Target error: " << e.what() << std::endl;
    }

    while (target_running) sleep(1);

    return 0;
}

void check_total_buffer_size() {
    uint64_t require_size = FLAGS_block_size * FLAGS_batch_size * FLAGS_threads;
    if (FLAGS_buffer_size < require_size) {
        FLAGS_buffer_size = require_size;
        LOG(WARNING) << "Invalid flag: buffer size is smaller than "
                        "require_size, adjust to "
                     << require_size;
    }
}

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    check_total_buffer_size();

    if (FLAGS_mode == "initiator")
        return initiator();
    else if (FLAGS_mode == "target")
        return target();

    LOG(ERROR) << "Unsupported mode: must be 'initiator' or 'target'";
    exit(EXIT_FAILURE);
}
