# Mooncake TE+ 架构规划

## 1. 现状分析

### 1.1 已实现的核心功能

| 功能模块 | 实现位置 | 状态 |
|---------|---------|------|
| 多传输后端支持 | `transport_*` | ✅ RDMA, NVLink, TCP, SHM, GDS, MNNVL, AscendDirect |
| Slice 切分 | `rdma_transport.cpp:submitTransferTasks` | ✅ 块大小自适应，最多 64 个 slice |
| Worker 模型 | `workers.cpp` | ✅ 多 worker 并发处理 |
| 轨道监控 | `rail_monitor.cpp` | ✅ 错误计数、冷却机制 |
| 设备配额 | `quota.cpp` | ✅ beta0/beta1 自适应负载均衡 |
| 共享配额 | `shared_quota.cpp` | ✅ 跨进程 PID 级别配额同步 |
| QoS 优先级 | `quota.h` | ✅ 3 级优先级 (HIGH/MEDIUM/LOW) |
| 故障重试 | `workers.cpp` | ✅ retry_count + fallback device |
| Metrics 系统 | `tent_metrics.cpp` | ✅ 读写延迟、失败率统计 |

### 1.2 现状 vs 设计需求

#### 1.2.1 硬件异构性屏蔽

**现状:**
- ✅ 支持 NVIDIA (CUDA/NCCL), 华为 (Ascend), CPU 内存
- ✅ 自动拓扑发现 (`Topology::discover()`)
- ⚠️ **问题**: Transport 选择在 `submitTransfer` 时完成，属于**早期绑定**
- ⚠️ **问题**: 每个任务一旦选择 transport 后，无法动态切换

**代码位置:**
```cpp
// transfer_engine_impl.cpp:962
task.type = resolveTransport(merged_request, 0);  // 早期绑定
```

#### 1.2.2 灰故障处理

**现状:**
- ✅ `RailMonitor::markFailed()` 区分拥塞错误和致命错误
- ✅ `RailMonitor::available()` 支持暂停/恢复机制
- ✅ `DeviceQuota` 通过 beta0/beta1 跟踪每个 NIC 的性能
- ⚠️ **问题**: Slice 提交时只选择一条 rail，没有**同时喷洒到多条 rail**
- ⚠️ **问题**: 没有基于实时遥测的动态路由

**代码位置:**
```cpp
// workers.cpp:506-574
Status Workers::selectOptimalDevice(...) {
    // 只选择一对 source_dev_id -> target_dev_id
    // 没有多路径并发传输
}
```

#### 1.2.3 快速自愈

**现状:**
- ✅ Slice 级别的重试 (`retry_count`)
- ✅ Fallback device 选择 (`selectFallbackDevice`)
- ⚠️ **问题**: `slice_timeout_ns_` 默认 10 秒，不是 50ms 级别
- ⚠️ **问题**: 重试次数超过 `max_retry_count` 后才宣告失败

**代码位置:**
```cpp
// workers.cpp:65-66
slice_timeout_ns_ = transport_->conf_->get(
    "transports/rdma/max_timeout_ns", kDefaultMaxTimeoutNs);  // 10秒
```

#### 1.2.4 QoS 支持

**现状:**
- ✅ `DeviceQuota::priority_` 支持 3 级优先级
- ✅ `SharedQuotaManager` 跨进程同步优先级负载
- ✅ QoS penalty 在 quota allocation 中
- ⚠️ **问题**: 优先级未暴露给上层 API

**代码位置:**
```cpp
// quota.cpp:147-160
if (shared_quota_ && priority_ > PRIO_HIGH) {
    uint64_t high_load = shared_quota_->getHighPrioLoad(dev_id);
    // ... QoS penalty
}
```

---

## 2. TE+ 架构演进路线

### 阶段一：声明式 API + Late Binding (短期)

#### 2.1.1 目标
- 应用只需声明传输意图，不绑定具体 transport
- 运行时根据实时状况动态选择最优路径

#### 2.1.2 设计方案

**新增 Intent 结构:**
```cpp
// intent.h
struct TransferIntent {
    void* source;
    uint64_t target_offset;
    size_t length;

    // 声明式约束
    uint8_t priority = 0;        // 0=high, 1=medium, 2=low
    uint64_t deadline_ns = 0;    // 绝对截止时间
    bool allow_multi_rail = true;  // 允许多路径喷洒
    bool require_ordered = false;  // 是否保序

    // 性能约束
    double min_bandwidth_gbps = 0;
    double max_latency_us = 0;
};
```

**Late Binding 时机:**
```cpp
// 在 slice 级别延迟绑定，而非 task 级别
// workers.cpp:generatePostPath() 中每次重试都重新评估
Status Workers::generatePostPath(RdmaSlice* slice) {
    // 当前实现：retry_count == 0 时选最优，否则选 fallback
    // 改进：每次都基于实时遥评估

    if (slice->retry_count == 0)
        CHECK_STATUS(selectOptimalDevice(source, target, slice));
    else
        CHECK_STATUS(selectFallbackDevice(source, target, slice));

    // 改进为：
    CHECK_STATUS(selectDeviceByTelemetry(source, target, slice));
}
```

#### 2.1.3 实施步骤
1. 扩展 `Request` 结构支持 Intent 字段
2. 修改 `generatePostPath()` 实现真正的 late binding
3. 添加 transport 动态切换逻辑

---

### 阶段二：灰故障感知与 Slice Spraying (中期)

#### 2.2.1 目标
- 将大象流拆分为细粒度 slice
- 基于实时遥测将 slice "喷洒"到最健康的轨道
- 主动绕过亚健康链路

#### 2.2.2 设计方案

**Rail Health Score:**
```cpp
// rail_monitor.h 增强
struct RailHealth {
    int error_count;
    double avg_latency_us;
    double bandwidth_gbps;
    double last_update_ts;

    // 综合健康评分 (0-100)
    double health_score() const {
        double latency_score = exp(-avg_latency_us / 1000.0);
        double error_score = exp(-error_count / 10.0);
        return 100 * latency_score * error_score;
    }
};
```

**Slice Spraying 策略:**
```cpp
// workers.cpp 新增
class SliceSprayer {
    // 加权随机选择，基于 health_score
    std::vector<int> selectRails(
        const std::vector<RailHealth>& healths,
        size_t num_slices);

    // 最小延迟路由
    std::vector<int> minLatencyRouting(
        const std::vector<RailHealth>& healths,
        size_t num_slices);
};
```

**实时遥测更新:**
```cpp
// quota.cpp 增强
Status DeviceQuota::release(int dev_id, uint64_t length, double latency) {
    // 现有 beta0/beta1 更新

    // 新增：更新 RailHealth
    auto& rail = rails_[dev_id];
    rail.avg_latency_us = latency * 1e6;
    rail.bandwidth_gbps = length / latency / 1e9;
    rail.last_update_ts = getCurrentTimeInNano();

    // 新增：触发 health_score 重算
    updateRailHealthScores();
}
```

#### 2.2.3 实施步骤
1. 扩展 `RailMonitor` 添加 `RailHealth` 结构
2. 在 `asyncPollCq()` 中更新遥测数据
3. 修改 `selectOptimalDevice()` 使用 health_score
4. 实现 `SliceSprayer` 多路径喷洒

---

### 阶段三：50ms 级快速自愈 (中期)

#### 2.3.1 目标
- Slice 级别的幂等重试
- 50ms 内自动屏蔽故障路径
- 无需应用感知

#### 2.3.2 设计方案

**快速故障检测:**
```cpp
// workers.cpp 修改
void Workers::asyncPollCq() {
    // 现有：slice_timeout_ns_ = 10秒
    // 改进：多级超时

    for (auto& slice : worker.inflight_slice_set) {
        uint64_t elapsed = current_ts - slice->submit_ts;

        // Level 1: 50ms 快速超时 - 立即重试
        if (elapsed > 50_000_000 && slice->retry_count == 0) {
            slice->retry_count++;
            submit(slice);  // 重新路由
            continue;
        }

        // Level 2: 500ms - 标记 rail 亚健康
        if (elapsed > 500_000_000 && slice->retry_count == 1) {
            markRailDegraded(slice->source_dev_id, slice->target_dev_id);
            slice->retry_count++;
            submit(slice);  // 选择替代 rail
            continue;
        }

        // Level 3: 10秒 - 最终超时
        if (elapsed > slice_timeout_ns_) {
            updateSliceStatus(slice, TIMEOUT);
        }
    }
}
```

**Rail 状态快速切换:**
```cpp
// rail_monitor.h 增强
enum RailState {
    RAIL_HEALTHY,
    RAIL_DEGRADED,    // 亚健康，降低权重
    RAIL_UNAVAILABLE  // 不可用，完全避开
};
```

#### 2.3.3 实施步骤
1. 实现多级超时机制
2. 添加 RAIL_DEGRADED 状态
3. 修改 `selectOptimalDevice()` 降低 degraded rail 权重
4. 优化重试路径选择

---

### 阶段四：完整的 QoS 支持 (长期)

#### 2.4.1 目标
- 基于优先级的资源预留
- 低优先级流被高优先级流抢占
- 满足 SLA 延迟要求

#### 2.4.2 设计方案

**优先级队列:**
```cpp
// workers.cpp 新增
struct PrioritySliceQueue {
    BoundedMPSCQueue<RdmaSliceList, 1024> high;
    BoundedMPSCQueue<RdmaSliceList, 1024> medium;
    BoundedMPSCQueue<RdmaSliceList, 1024> low;

    // 严格优先级调度
    RdmaSliceList* pop() {
        if (!high.empty()) return high.pop();
        if (!medium.empty()) return medium.pop();
        return low.pop();
    }
};
```

**资源预留:**
```cpp
// quota.cpp 增强
class DeviceQuota {
    // 按优先级预留带宽
    struct Reservation {
        uint64_t high_reserved;   // 保留给高优先级
        uint64_t medium_reserved;
    };

    bool canAllocate(uint64_t length, uint8_t priority) {
        // 高优先级可以抢占低优先级
        if (priority == PRIO_HIGH) {
            return true;
        }
        // 中/低优先级检查预留
        return (active_bytes + length) < (total_bytes - high_reserved);
    }
};
```

#### 2.4.3 实施步骤
1. 扩展 API 支持 priority 参数
2. 实现优先级队列调度
3. 添加带宽预留机制
4. 实现抢占逻辑

---

## 3. 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        Application                              │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Transfer Engine (TE+)                        │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │  Intent API │  │ Late Binding │  │  QoS Scheduler        │  │
│  └──────┬──────┘  └──────┬───────┘  └───────┬───────────────┘  │
│         │                │                  │                   │
│         └────────────────┴──────────────────┘                   │
│                            │                                     │
│         ┌──────────────────┴───────────────────┐                │
│         │         Task Dispatcher               │                │
│         └──────────────────┬───────────────────┘                │
└────────────────────────────┼─────────────────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
┌───────────────────────────┐    ┌───────────────────────────┐
│      Transport Layer      │    │     Telemetry Service     │
│  ┌──────┐  ┌──────┐      │    │  ┌─────┐  ┌──────┐       │
│  │ RDMA │  │ NVLink│ ... │    │  │ Rail│  │ Device│      │
│  └──┬───┘  └──┬───┘      │    │  │Health│ │Quota │       │
│     │         │           │    │  └──┬──┘  └──┬───┘       │
│     └────┬────┘           │    │     │        │            │
│          │                │    │     └───┬────┘            │
│          ▼                │    │         │                 │
│  ┌───────────────────┐    │    │         ▼                 │
│  │  Slice Sprayer    │    │    │  ┌───────────┐           │
│  │  - Multi-Rail     │    │    │  │ Real-time │           │
│  │  - Health-Based   │◄───┴────┘  │ Scoring   │           │
│  └─────────┬─────────┘    │       └───────────┘           │
│            │              │                                  │
│            ▼              │                                  │
│  ┌───────────────────┐    │                                  │
│  │   Worker Pool     │    │                                  │
│  │  - Post Send      │    │                                  │
│  │  - Poll CQ        │    │                                  │
│  │  - Fast Retry     │    │                                  │
│  └───────────────────┘    │                                  │
└──────────────────────────┴──────────────────────────────────┘
```

---

## 4. 关键性能指标

| 指标 | 当前 | 目标 | 实现阶段 |
|-----|------|------|---------|
| 故障检测时间 | 10秒 | 50ms | 阶段三 |
| 故障恢复时间 | 分钟级 | 50ms | 阶段三 |
| 多路径并发 | 无 | 支持 | 阶段二 |
| QoS 级别 | 3 | 3 + 抢占 | 阶段四 |
| 声明式 API | 无 | 有 | 阶段一 |
| Late Binding | 部分 | 完整 | 阶段一 |

---

## 5. 文件修改清单

### 阶段一
| 文件 | 修改内容 |
|-----|---------|
| `tent/common/types.h` | 添加 `TransferIntent` 结构 |
| `tent/runtime/transfer_engine_impl.cpp` | 修改 `submitTransfer` 支持 Intent |
| `tent/transport/rdma/workers.cpp` | 修改 `generatePostPath` 实现真正的 late binding |

### 阶段二
| 文件 | 修改内容 |
|-----|---------|
| `tent/transport/rdma/rail_monitor.h` | 添加 `RailHealth` 结构 |
| `tent/transport/rdma/rail_monitor.cpp` | 实现 health score 计算 |
| `tent/transport/rdma/quota.cpp` | 在 `release()` 中更新遥测 |
| `tent/transport/rdma/workers.cpp` | 实现 `SliceSprayer` |
| `tent/transport/rdma/workers.h` | 添加 `SliceSprayer` 类 |

### 阶段三
| 文件 | 修改内容 |
|-----|---------|
| `tent/transport/rdma/workers.cpp` | 实现多级超时机制 |
| `tent/transport/rdma/rail_monitor.h` | 添加 `RAIL_DEGRADED` 状态 |
| `tent/transport/rdma/params.h` | 添加超时参数 |

### 阶段四
| 文件 | 修改内容 |
|-----|---------|
| `tent/transfer_engine.h` | 添加 priority 参数 |
| `tent/transport/rdma/workers.h` | 添加 `PrioritySliceQueue` |
| `tent/transport/rdma/quota.cpp` | 实现带宽预留和抢占 |

---

## 6. 总结

Mooncake TE 已经具备良好的基础架构，主要差距在于：

1. **Late Binding**: 当前是 task 级别的早期绑定，需要演进到 slice 级别的延迟绑定
2. **Slice Spraying**: 当前每个 slice 只走一条 rail，需要支持多路径并发
3. **快速自愈**: 当前超时 10 秒，需要缩短到 50ms
4. **QoS 暴露**: 当前优先级是内部实现，需要暴露给 API 层

以上架构演进计划将分四个阶段逐步实现 TE+ 的设计目标。
