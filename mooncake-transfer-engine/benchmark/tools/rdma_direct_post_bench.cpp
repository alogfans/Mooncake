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

#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Options {
    bool server{false};
    std::string target;
    std::string ib_dev;
    int tcp_port{15172};
    int ib_port{1};
    int gid_index{0};
    size_t block_size{1048576};
    int duration_s{30};
    uint64_t request_interval_us{0};
    std::string op{"read"};
};

struct PeerInfo {
    uint32_t qp_num{0};
    uint16_t lid{0};
    uint32_t rkey{0};
    uint64_t addr{0};
    uint8_t gid[16]{};
};

uint64_t nowNs() {
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
}

void die(const std::string& msg) {
    std::cerr << msg << std::endl;
    std::exit(EXIT_FAILURE);
}

bool startsWith(const std::string& s, const std::string& prefix) {
    return s.rfind(prefix, 0) == 0;
}

std::string argValue(const std::string& arg, const std::string& key) {
    return arg.substr(key.size());
}

Options parseArgs(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--server") {
            opt.server = true;
        } else if (startsWith(arg, "--target_seg_name=")) {
            opt.target = argValue(arg, "--target_seg_name=");
        } else if (startsWith(arg, "--target=")) {
            opt.target = argValue(arg, "--target=");
        } else if (startsWith(arg, "--listen_port=")) {
            opt.tcp_port = std::stoi(argValue(arg, "--listen_port="));
        } else if (startsWith(arg, "--ib_dev=")) {
            opt.ib_dev = argValue(arg, "--ib_dev=");
        } else if (startsWith(arg, "--ib_port=")) {
            opt.ib_port = std::stoi(argValue(arg, "--ib_port="));
        } else if (startsWith(arg, "--gid_index=")) {
            opt.gid_index = std::stoi(argValue(arg, "--gid_index="));
        } else if (startsWith(arg, "--block_size=") ||
                   startsWith(arg, "--start_block_size=") ||
                   startsWith(arg, "--max_block_size=")) {
            const auto pos = arg.find('=');
            opt.block_size = std::stoull(arg.substr(pos + 1));
        } else if (startsWith(arg, "--duration=")) {
            opt.duration_s = std::stoi(argValue(arg, "--duration="));
        } else if (startsWith(arg, "--request_interval_us=")) {
            opt.request_interval_us =
                std::stoull(argValue(arg, "--request_interval_us="));
        } else if (startsWith(arg, "--op=")) {
            opt.op = argValue(arg, "--op=");
        } else {
            die("unknown argument: " + arg);
        }
    }
    if (!opt.server && opt.target.empty())
        die("client mode requires --target_seg_name=<host:port>");
    return opt;
}

std::pair<std::string, int> parseHostPort(const std::string& value) {
    const auto pos = value.rfind(':');
    if (pos == std::string::npos) die("target must be host:port");
    return {value.substr(0, pos), std::stoi(value.substr(pos + 1))};
}

void writeFull(int fd, const void* data, size_t len) {
    const char* p = static_cast<const char*>(data);
    while (len > 0) {
        ssize_t n = ::write(fd, p, len);
        if (n <= 0) die("tcp write failed: " + std::string(strerror(errno)));
        p += n;
        len -= static_cast<size_t>(n);
    }
}

void readFull(int fd, void* data, size_t len) {
    char* p = static_cast<char*>(data);
    while (len > 0) {
        ssize_t n = ::read(fd, p, len);
        if (n <= 0) die("tcp read failed: " + std::string(strerror(errno)));
        p += n;
        len -= static_cast<size_t>(n);
    }
}

int listenTcp(int port) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) die("socket failed");
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0)
        die("bind failed: " + std::string(strerror(errno)));
    if (::listen(fd, 1) != 0) die("listen failed");
    int conn = ::accept(fd, nullptr, nullptr);
    if (conn < 0) die("accept failed");
    ::close(fd);
    return conn;
}

int connectTcp(const std::string& host, int port) {
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* result = nullptr;
    const std::string port_str = std::to_string(port);
    if (getaddrinfo(host.c_str(), port_str.c_str(), &hints, &result) != 0)
        die("getaddrinfo failed");
    int fd = -1;
    for (auto* rp = result; rp; rp = rp->ai_next) {
        fd = ::socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (fd < 0) continue;
        if (::connect(fd, rp->ai_addr, rp->ai_addrlen) == 0) break;
        ::close(fd);
        fd = -1;
    }
    freeaddrinfo(result);
    if (fd < 0) die("connect failed");
    return fd;
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double rank = p / 100.0 * (values.size() - 1);
    const size_t idx = static_cast<size_t>(rank);
    const double frac = rank - idx;
    if (idx + 1 >= values.size()) return values[idx];
    return values[idx] * (1.0 - frac) + values[idx + 1] * frac;
}

class RdmaDirectBench {
   public:
    explicit RdmaDirectBench(const Options& opt) : opt_(opt) {}

    void init() {
        int num_devs = 0;
        ibv_device** devs = ibv_get_device_list(&num_devs);
        if (!devs || num_devs == 0) die("no RDMA devices found");
        ibv_device* selected = nullptr;
        for (int i = 0; i < num_devs; ++i) {
            const char* name = ibv_get_device_name(devs[i]);
            if (opt_.ib_dev.empty() || opt_.ib_dev == name) {
                selected = devs[i];
                break;
            }
        }
        if (!selected) die("requested RDMA device not found");
        ctx_ = ibv_open_device(selected);
        dev_name_ = ibv_get_device_name(selected);
        ibv_free_device_list(devs);
        if (!ctx_) die("ibv_open_device failed");

        if (ibv_query_port(ctx_, opt_.ib_port, &port_attr_) != 0)
            die("ibv_query_port failed");
        if (ibv_query_gid(ctx_, opt_.ib_port, opt_.gid_index, &gid_) != 0)
            die("ibv_query_gid failed");

        pd_ = ibv_alloc_pd(ctx_);
        if (!pd_) die("ibv_alloc_pd failed");
        cq_ = ibv_create_cq(ctx_, 1, nullptr, nullptr, 0);
        if (!cq_) die("ibv_create_cq failed");

        ibv_qp_init_attr qp_attr{};
        qp_attr.send_cq = cq_;
        qp_attr.recv_cq = cq_;
        qp_attr.qp_type = IBV_QPT_RC;
        qp_attr.sq_sig_all = false;
        qp_attr.cap.max_send_wr = 1;
        qp_attr.cap.max_recv_wr = 1;
        qp_attr.cap.max_send_sge = 1;
        qp_attr.cap.max_recv_sge = 1;
        qp_ = ibv_create_qp(pd_, &qp_attr);
        if (!qp_) die("ibv_create_qp failed");

        buffer_.resize(opt_.block_size);
        std::fill(buffer_.begin(), buffer_.end(), 0x5a);
        mr_ = ibv_reg_mr(pd_, buffer_.data(), buffer_.size(),
                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                             IBV_ACCESS_REMOTE_WRITE);
        if (!mr_) die("ibv_reg_mr failed");
    }

    PeerInfo localInfo() const {
        PeerInfo info{};
        info.qp_num = qp_->qp_num;
        info.lid = port_attr_.lid;
        info.rkey = mr_->rkey;
        info.addr = reinterpret_cast<uint64_t>(buffer_.data());
        std::memcpy(info.gid, gid_.raw, sizeof(info.gid));
        return info;
    }

    void connectQp(const PeerInfo& peer) {
        ibv_qp_attr attr{};
        attr.qp_state = IBV_QPS_INIT;
        attr.port_num = opt_.ib_port;
        attr.pkey_index = 0;
        attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE |
                               IBV_ACCESS_REMOTE_READ |
                               IBV_ACCESS_REMOTE_WRITE;
        if (ibv_modify_qp(qp_, &attr,
                          IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                              IBV_QP_ACCESS_FLAGS) != 0)
            die("modify QP INIT failed: " + std::string(strerror(errno)));

        std::memset(&attr, 0, sizeof(attr));
        attr.qp_state = IBV_QPS_RTR;
        attr.path_mtu = port_attr_.active_mtu;
        attr.dest_qp_num = peer.qp_num;
        attr.rq_psn = 0;
        attr.max_dest_rd_atomic = 1;
        attr.min_rnr_timer = 0x12;
        attr.ah_attr.is_global = 1;
        attr.ah_attr.dlid = peer.lid;
        attr.ah_attr.sl = 0;
        attr.ah_attr.src_path_bits = 0;
        attr.ah_attr.port_num = opt_.ib_port;
        std::memcpy(&attr.ah_attr.grh.dgid, peer.gid, sizeof(peer.gid));
        attr.ah_attr.grh.flow_label = 0;
        attr.ah_attr.grh.sgid_index = opt_.gid_index;
        attr.ah_attr.grh.hop_limit = 255;
        attr.ah_attr.grh.traffic_class = 0;
        if (ibv_modify_qp(qp_, &attr,
                          IBV_QP_STATE | IBV_QP_PATH_MTU |
                              IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                              IBV_QP_MAX_DEST_RD_ATOMIC |
                              IBV_QP_MIN_RNR_TIMER | IBV_QP_AV) != 0)
            die("modify QP RTR failed: " + std::string(strerror(errno)));

        std::memset(&attr, 0, sizeof(attr));
        attr.qp_state = IBV_QPS_RTS;
        attr.timeout = 0x12;
        attr.retry_cnt = 7;
        attr.rnr_retry = 7;
        attr.sq_psn = 0;
        attr.max_rd_atomic = 1;
        if (ibv_modify_qp(qp_, &attr,
                          IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                              IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                              IBV_QP_MAX_QP_RD_ATOMIC) != 0)
            die("modify QP RTS failed: " + std::string(strerror(errno)));
    }

    void runClient(const PeerInfo& peer) {
        const bool is_read = opt_.op != "write";
        std::vector<double> samples;
        const uint64_t end_ns = nowNs() + opt_.duration_s * 1000000000ull;
        uint64_t bytes = 0;
        while (nowNs() < end_ns) {
            if (opt_.request_interval_us != 0) {
                const uint64_t target =
                    nowNs() + opt_.request_interval_us * 1000ull;
                while (nowNs() < target) {}
            }
            const uint64_t begin = nowNs();
            postOne(peer, is_read);
            pollOne();
            const uint64_t done = nowNs();
            samples.push_back(static_cast<double>(done - begin) / 1000.0);
            bytes += opt_.block_size;
        }
        const double total_us =
            std::accumulate(samples.begin(), samples.end(), 0.0);
        const double avg_us = samples.empty() ? 0.0 : total_us / samples.size();
        const double wall_s = opt_.duration_s;
        const double gbps = static_cast<double>(bytes) / 1e9 / wall_s;
        std::cout << std::fixed << std::setprecision(6)
                  << "rdma_direct_post op=" << (is_read ? "read" : "write")
                  << " device=" << dev_name_
                  << " block_size=" << opt_.block_size
                  << " interval_us=" << opt_.request_interval_us
                  << " operations=" << samples.size()
                  << " BW(GB/s)=" << gbps
                  << " latency_us(avg/p50/p95/p99)=" << avg_us << '/'
                  << percentile(samples, 50) << '/' << percentile(samples, 95)
                  << '/' << percentile(samples, 99) << std::endl;
    }

    void runServer() {
        std::cout << "rdma_direct_post server ready on device " << dev_name_
                  << ", buffer bytes=" << buffer_.size() << std::endl;
        while (true) std::this_thread::sleep_for(std::chrono::seconds(60));
    }

   private:
    void postOne(const PeerInfo& peer, bool is_read) {
        ibv_sge sge{};
        sge.addr = reinterpret_cast<uint64_t>(buffer_.data());
        sge.length = static_cast<uint32_t>(opt_.block_size);
        sge.lkey = mr_->lkey;

        ibv_send_wr wr{};
        wr.wr_id = 1;
        wr.opcode = is_read ? IBV_WR_RDMA_READ : IBV_WR_RDMA_WRITE;
        wr.num_sge = 1;
        wr.sg_list = &sge;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.wr.rdma.remote_addr = peer.addr;
        wr.wr.rdma.rkey = peer.rkey;
        ibv_send_wr* bad = nullptr;
        if (ibv_post_send(qp_, &wr, &bad) != 0)
            die("ibv_post_send failed: " + std::string(strerror(errno)));
    }

    void pollOne() {
        ibv_wc wc{};
        while (true) {
            int n = ibv_poll_cq(cq_, 1, &wc);
            if (n < 0) die("ibv_poll_cq failed");
            if (n == 0) continue;
            if (wc.status != IBV_WC_SUCCESS)
                die("WC failed: " + std::to_string(wc.status));
            return;
        }
    }

    Options opt_;
    std::string dev_name_;
    ibv_context* ctx_{nullptr};
    ibv_pd* pd_{nullptr};
    ibv_cq* cq_{nullptr};
    ibv_qp* qp_{nullptr};
    ibv_mr* mr_{nullptr};
    ibv_port_attr port_attr_{};
    ibv_gid gid_{};
    std::vector<char> buffer_;
};

}  // namespace

int main(int argc, char** argv) {
    Options opt = parseArgs(argc, argv);
    if (!opt.server) {
        const auto [host, port] = parseHostPort(opt.target);
        opt.tcp_port = port;
        RdmaDirectBench bench(opt);
        bench.init();
        int fd = connectTcp(host, opt.tcp_port);
        PeerInfo local = bench.localInfo();
        PeerInfo peer{};
        writeFull(fd, &local, sizeof(local));
        readFull(fd, &peer, sizeof(peer));
        bench.connectQp(peer);
        writeFull(fd, "R", 1);
        char ready = 0;
        readFull(fd, &ready, 1);
        bench.runClient(peer);
        return 0;
    }

    RdmaDirectBench bench(opt);
    bench.init();
    int fd = listenTcp(opt.tcp_port);
    PeerInfo local = bench.localInfo();
    PeerInfo peer{};
    readFull(fd, &peer, sizeof(peer));
    writeFull(fd, &local, sizeof(local));
    bench.connectQp(peer);
    char ready = 0;
    readFull(fd, &ready, 1);
    writeFull(fd, "R", 1);
    bench.runServer();
    return 0;
}
