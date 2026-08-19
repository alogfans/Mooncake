# SGLang RDMA Trace Replay Benchmark

This document describes how to reproduce the 4P + 4D SGLang/Mooncake RDMA
trace replay experiment and records the three-mode comparison result.

The benchmark replays `build/rdma_traffic.csv` as real TENT
`TransferEngine::submitTransfer` operations:

- `batch_transfer_sync`: P/D KV transfer, GPU-to-GPU `WRITE`.
- `mooncake_put`: Mooncake Store put, GPU-to-DRAM `WRITE`.
- `mooncake_get`: Mooncake Store get, DRAM-to-GPU `READ`.

Each TP uses its own GPU device `cuda:{tp}`. DRAM buffers use NUMA node 0
(`cpu:0`). The replay reports P50/P95/P99 latency and average instantaneous
bandwidth. Foreground PD traffic is also split by whether a Store put was
active at the same time.

## Test Environment

- Date: 2026-08-19
- Replay node: local machine under `/mnt/qjh000/rf/Mooncake`
- Target node: `qjh001`
- Metadata server: `http://qjh001:8080/metadata`
- TP size: 4
- Duration: 300 s
- Warmup skipped from stats: 10 s
- Trace: `build/rdma_traffic.csv`
- DRAM location: `cpu:0`
- GPU location: `cuda:{tp}`
- Modes 1/2 RDMA blacklist: `mlx5_0`
- Mode 3 RDMA whitelist: `mlx5_1,mlx5_2,mlx5_3,mlx5_4`

Unset these environment variables for every run. In particular,
`MC_TE_FILTERS` and `MC_TE_FILTERS_EXCLUDE` can override the config-file
NIC filters.

```bash
unset MC_TENT_CONF
unset MC_TE_FILTERS
unset MC_TE_FILTERS_EXCLUDE
```

## Build

```bash
cmake -S . -B build -DUSE_TENT=ON -DUSE_CUDA=ON -DBUILD_BENCHMARK=ON
cmake --build build --target sglang_trace_replay_bench -j16
```

Binary:

```bash
build/mooncake-transfer-engine/benchmark/sglang_trace_replay_bench
```

## Built-In Defaults

The benchmark is specialized for this SGLang 4P + 4D replay. These parameters
are built into `sglang_trace_replay_bench` and normally do not need to be
passed on the command line:

| Parameter | Default |
|---|---|
| `--trace_file` | `build/rdma_traffic.csv` |
| `--metadata_type` | `http` |
| `--metadata_servers` | `http://qjh001:8080/metadata` |
| `--transport` | `rdma` |
| `--tp_size` | `4` |
| `--cpu_location` | `cpu:0` |
| `--gpu_location` | `cuda:{tp}` |
| `--buffer_size` | `1073741824` |
| `--duration_s` | `300` |
| `--stats_skip_s` | `10` |
| `--skip_control` | `false` |

The remaining scenario-specific defaults are selected by
`--scenario=nonintent|intent|qpool`:

| Scenario | Config | Target base port | Replay base port | Target segment | Replay segment | PD intent | Store intent |
|---|---|---:|---:|---|---|---|---|
| `nonintent` | `benchmarks/sglang_rdma_blacklist.json` | 19901 | 19941 | `sglang-nonintent-target` | `sglang-nonintent-replay-5m` | `unspec` | `unspec` |
| `intent` | `benchmarks/sglang_intent_baseline.json` | 20001 | 20041 | `sglang-intent-target` | `sglang-intent-replay-5m` | `foreground_get` | `background_prefetch` |
| `qpool` | `benchmarks/sglang_qpool_enhanced.json` | 20401 | 20441 | `sglang-qpool-target` | `sglang-qpool-replay-5m` | `foreground_get` | `background_prefetch` |

All defaults can still be overridden explicitly when running a different host,
trace, port range, or buffer layout.

## Config Files

`benchmarks/sglang_rdma_blacklist.json` is used by the non-intent mode:

```json
{
  "topology": {
    "rdma_blacklist": ["mlx5_0"]
  }
}
```

`benchmarks/sglang_intent_baseline.json` is used by the single-TE intent
baseline. It intentionally does not configure per-intent QP pools.

```json
{
  "topology": {
    "rdma_blacklist": ["mlx5_0"]
  },
  "policy": [
    {
      "name": "sglang-foreground",
      "segment_type": "memory",
      "intent_type": "foreground_get",
      "transports": ["rdma"]
    },
    {
      "name": "sglang-background",
      "segment_type": "memory",
      "intent_type": "background_prefetch",
      "transports": ["rdma"]
    },
    {
      "name": "sglang-default",
      "segment_type": "memory",
      "transports": ["rdma"]
    }
  ]
}
```

`benchmarks/sglang_qpool_enhanced.json` is used by the enhanced single-TE mode.
It keeps one TE instance per TP, but maps foreground PD and background Store
traffic to different QP pools and traffic classes.

```json
{
  "topology": {
    "rdma_whitelist": ["mlx5_1", "mlx5_2", "mlx5_3", "mlx5_4"]
  },
  "policy": [
    {
      "name": "sglang-foreground",
      "segment_type": "memory",
      "intent_type": "foreground_get",
      "transports": ["rdma"],
      "qp_pool": "fg"
    },
    {
      "name": "sglang-background",
      "segment_type": "memory",
      "intent_type": "background_prefetch",
      "transports": ["rdma"],
      "qp_pool": "bg"
    },
    {
      "name": "sglang-default",
      "segment_type": "memory",
      "transports": ["rdma"],
      "qp_pool": "fg"
    }
  ],
  "transports": {
    "rdma": {
      "endpoint": {
        "traffic_class": 96,
        "service_level": 3,
        "qp_pools": [
          {
            "name": "fg",
            "num_qp": 4,
            "traffic_class": 96,
            "service_level": 3
          },
          {
            "name": "bg",
            "num_qp": 2,
            "traffic_class": 8,
            "service_level": 0
          }
        ]
      }
    }
  }
}
```

## Mode 1: Non-Intent Baseline

This mode uses one TE instance per TP. PD and Store traffic share the same TE.
All requests use the default `unspec` intent.

Start the target on `qjh001`:

```bash
ssh qjh001 '
cd /mnt/qjh000/rf/Mooncake &&
env -u MC_TENT_CONF -u MC_TE_FILTERS -u MC_TE_FILTERS_EXCLUDE \
build/mooncake-transfer-engine/benchmark/sglang_trace_replay_bench \
  --mode=target \
  --scenario=nonintent
'
```

Run replay locally:

```bash
env -u MC_TENT_CONF -u MC_TE_FILTERS -u MC_TE_FILTERS_EXCLUDE \
build/mooncake-transfer-engine/benchmark/sglang_trace_replay_bench \
  --mode=replay \
  --scenario=nonintent
```

## Mode 2: Intent Baseline

This mode still uses one TE instance per TP. PD and Store traffic still share
the same TE, but replay attaches intents:

- PD: `foreground_get`
- Store: `background_prefetch`

Start the target on `qjh001`:

```bash
ssh qjh001 '
cd /mnt/qjh000/rf/Mooncake &&
env -u MC_TENT_CONF -u MC_TE_FILTERS -u MC_TE_FILTERS_EXCLUDE \
build/mooncake-transfer-engine/benchmark/sglang_trace_replay_bench \
  --mode=target \
  --scenario=intent
'
```

Run replay locally:

```bash
env -u MC_TENT_CONF -u MC_TE_FILTERS -u MC_TE_FILTERS_EXCLUDE \
build/mooncake-transfer-engine/benchmark/sglang_trace_replay_bench \
  --mode=replay \
  --scenario=intent
```

## Mode 3: Enhanced Single TE + Per-Intent QP Pool

This mode uses one TE instance per TP. PD and Store share that TE, but the
intent policy routes them to different QP pools:

- Foreground PD pool `fg`: 4 QPs, `traffic_class=96`, `service_level=3`
- Background Store pool `bg`: 2 QPs, `traffic_class=8`, `service_level=0`

This path required a worker/QP ownership fix: slices routed to a per-intent QP
pool must be submitted by the worker that polls that QP's CQ, and mixed-pool
worker batches must be split by QP pool before posting.

Start the target on `qjh001`:

```bash
ssh qjh001 '
cd /mnt/qjh000/rf/Mooncake &&
env -u MC_TENT_CONF -u MC_TE_FILTERS -u MC_TE_FILTERS_EXCLUDE \
build/mooncake-transfer-engine/benchmark/sglang_trace_replay_bench \
  --mode=target \
  --scenario=qpool
'
```

Run replay locally:

```bash
env -u MC_TENT_CONF -u MC_TE_FILTERS -u MC_TE_FILTERS_EXCLUDE \
build/mooncake-transfer-engine/benchmark/sglang_trace_replay_bench \
  --mode=replay \
  --scenario=qpool
```

## Cleanup

After each run, stop the remote target and verify that the ports are free.

```bash
ssh qjh001 "pgrep -af 'sglang_trace_replay_bench.*--mode=target' || true"
ssh qjh001 "pkill -f 'sglang_trace_replay_bench.*--mode=target' || true"
ssh qjh001 "ss -ltnp '( sport >= :19901 and sport <= :20550 )' 2>/dev/null || true"
```

## Results

All reported runs use the same 5-minute replay window and skip the first 10
seconds from summary statistics.

### Three-Mode Comparison

The first experiment compares the three benchmark modes on the original trace.
Mode 3 is the most complete intent-aware configuration.

Raw logs are under:

```text
build/sglang_three_mode_rerun/
```

| Mode | Description | Overall P50/P95/P99 us | Overall Avg Inst GB/s | Foreground overlap P50/P95/P99 us | Foreground no-overlap P50/P95/P99 us | Store put P50/P95/P99 us |
|---|---|---:|---:|---:|---:|---:|
| 1 | Non-intent, single TE | 531.562 / 9357.936 / 21372.567 | 11.128 | 519.318 / 9325.965 / 10710.299 | 365.062 / 543.358 / 1184.003 | 1637.175 / 14182.239 / 28191.208 |
| 2 | Intent baseline, single TE | 368.446 / 9219.833 / 20640.438 | 11.860 | 1131.722 / 11969.799 / 21194.583 | 357.531 / 547.632 / 1260.666 | 1525.836 / 15723.282 / 29872.811 |
| 3 | Single TE + per-intent QP pool | 450.828 / 11344.172 / 22489.473 | 9.506 | 605.326 / 1546.242 / 2903.871 | 402.235 / 756.143 / 966.026 | 2289.001 / 19300.610 / 28558.473 |

The key metric is foreground PD latency while Store puts are active:

```text
Mode 1 foreground overlap P99: 10710.299 us
Mode 2 foreground overlap P99: 21194.583 us
Mode 3 foreground overlap P99:  2903.871 us
```

Mode 2 improves metadata latency significantly, but because PD and Store still
share the same TE/RDMA resources, foreground overlap tail latency is not
isolated. Mode 3 keeps one TE per TP, but separates foreground and background at
the QP-pool/traffic-class level. In this run it reduces foreground-overlap P99
by about 72.9% versus Mode 1 and 86.3% versus Mode 2. The tradeoff is lower
overall average instantaneous bandwidth and higher Store put P50/P95 latency,
because background Store traffic is explicitly pushed into the lower-TC, smaller
QP pool. Store put P99 remains comparable across the three modes.

### Mode 3 Background-Load Sweep

The second experiment keeps Mode 3 fixed and changes only the amount of
background Store traffic. Foreground PD rows from `build/rdma_traffic.csv` are
kept unchanged. Store rows are edited as follows:

| BG ratio | Trace mutation |
|---:|---|
| `0x` | drop all `mooncake_put` / `mooncake_get` rows |
| `0.5x` | keep every other Store row |
| `1x` | original trace |
| `2x` | duplicate each Store row once |

The generated traces and raw logs are under:

```text
build/sglang_qpool_bg_sweep/
```

The target was started once with Mode 3:

```bash
ssh qjh001 '
cd /mnt/qjh000/rf/Mooncake &&
env -u MC_TENT_CONF -u MC_TE_FILTERS -u MC_TE_FILTERS_EXCLUDE \
build/mooncake-transfer-engine/benchmark/sglang_trace_replay_bench \
  --mode=target \
  --scenario=qpool
'
```

Each replay run used a generated trace and a unique replay segment name:

```bash
env -u MC_TENT_CONF -u MC_TE_FILTERS -u MC_TE_FILTERS_EXCLUDE \
build/mooncake-transfer-engine/benchmark/sglang_trace_replay_bench \
  --mode=replay \
  --scenario=qpool \
  --trace_file=build/sglang_qpool_bg_sweep/rdma_traffic_bg_${ratio}x.csv \
  --local_segment_name=sglang-qpool-replay-bg-${ratio}x \
  --duration_s=300 \
  --stats_skip_s=10
```

#### Sweep Summary

| BG ratio | Overall P50/P95/P99 us | Overall Avg Inst GB/s | Foreground P50/P95/P99 us | Foreground overlap P50/P95/P99 us | Store put P50/P95/P99 us |
|---:|---:|---:|---:|---:|---:|
| `0x` | 164.799 / 5694.080 / 11369.625 | 8.913 | 376.588 / 671.148 / 1848.576 | n/a | n/a |
| `0.5x` | 370.445 / 7555.016 / 19180.804 | 8.902 | 436.017 / 911.903 / 2169.208 | 423.470 / 1112.425 / 2744.306 | 2327.048 / 13996.307 / 23462.938 |
| `1x` | 385.139 / 9472.040 / 21491.452 | 10.977 | 380.381 / 775.577 / 1142.370 | 454.457 / 1479.991 / 2969.827 | 1755.582 / 16229.208 / 27729.423 |
| `2x` | 624.798 / 10190.685 / 23118.628 | 11.567 | 395.648 / 1083.424 / 5927.868 | 522.480 / 1636.366 / 2504.047 | 1798.106 / 14631.067 / 27896.365 |

#### Sweep Details

| BG ratio | Group | Events | Bytes | Latency P50/P95/P99 us | Avg Inst GB/s |
|---:|---|---:|---:|---:|---:|
| `0x` | overall | 3296 | 54917610496 | 164.799 / 5694.080 / 11369.625 | 8.913 |
| `0x` | foreground_pd | 692 | 4172283904 | 376.588 / 671.148 / 1848.576 | 17.310 |
| `0x` | foreground_pd.no_store_put | 692 | 4172283904 | 376.588 / 671.148 / 1848.576 | 17.310 |
| `0.5x` | overall | 3774 | 77479296000 | 370.445 / 7555.016 / 19180.804 | 8.902 |
| `0.5x` | foreground_pd | 692 | 4172283904 | 436.017 / 911.903 / 2169.208 | 15.035 |
| `0.5x` | foreground_pd.overlap_store_put | 87 | 524550144 | 423.470 / 1112.425 / 2744.306 | 14.673 |
| `0.5x` | foreground_pd.no_store_put | 605 | 3647733760 | 436.286 / 768.435 / 1820.072 | 15.088 |
| `0.5x` | store.put | 478 | 22561685504 | 2327.048 / 13996.307 / 23462.938 | 16.296 |
| `1x` | overall | 4252 | 100040981504 | 385.139 / 9472.040 / 21491.452 | 10.977 |
| `1x` | foreground_pd | 692 | 4172283904 | 380.381 / 775.577 / 1142.370 | 16.188 |
| `1x` | foreground_pd.overlap_store_put | 91 | 548667392 | 454.457 / 1479.991 / 2969.827 | 12.576 |
| `1x` | foreground_pd.no_store_put | 601 | 3623616512 | 374.106 / 664.809 / 866.969 | 16.736 |
| `1x` | store.put | 956 | 45123371008 | 1755.582 / 16229.208 / 27729.423 | 17.483 |
| `2x` | overall | 5208 | 145164352512 | 624.798 / 10190.685 / 23118.628 | 11.567 |
| `2x` | foreground_pd | 692 | 4172283904 | 395.648 / 1083.424 / 5927.868 | 15.041 |
| `2x` | foreground_pd.overlap_store_put | 104 | 627048448 | 522.480 / 1636.366 / 2504.047 | 11.830 |
| `2x` | foreground_pd.no_store_put | 588 | 3545235456 | 386.435 / 773.390 / 6113.051 | 15.609 |
| `2x` | store.put | 1912 | 90246742016 | 1798.106 / 14631.067 / 27896.365 | 17.090 |

### Mode 3 External-Traffic Fault Injection

The third experiment keeps the main Mode 3 replay unchanged and injects
additional external RDMA traffic from a separate `sglang_trace_replay_bench`
process. This models reduced available network bandwidth without mutating the
main business trace.

The external traces are Store-only traces generated from `build/rdma_traffic.csv`:

| External load | Trace mutation |
|---:|---|
| `none` | no external injector; reuse the Mode 3 three-mode rerun |
| `1x` | keep only `mooncake_put` / `mooncake_get` rows |
| `2x` | keep only Store rows and duplicate each Store row once |

The generated traces and raw logs are under:

```text
build/sglang_fault_injection/
```

The external target uses an independent segment and RPC port range:

```bash
ssh qjh001 '
cd /mnt/qjh000/rf/Mooncake &&
env -u MC_TENT_CONF -u MC_TE_FILTERS -u MC_TE_FILTERS_EXCLUDE \
build/mooncake-transfer-engine/benchmark/sglang_trace_replay_bench \
  --mode=target \
  --scenario=qpool \
  --local_segment_name=external-store-target \
  --rpc_server_port=21201
'
```

For each fault-injection run, the external Store-only replay starts 15 seconds
before the main replay and runs for 330 seconds. The main replay remains the
original 300-second Mode 3 replay over `build/rdma_traffic.csv`.

#### Fault-Injection Summary

| External load | Injector Store put Avg Inst GB/s | Main overall P50/P95/P99 us | Main overall Avg Inst GB/s | Main foreground overlap P50/P95/P99 us | Main foreground P50/P95/P99 us | Main Store put P50/P95/P99 us |
|---:|---:|---:|---:|---:|---:|---:|
| `none` | n/a | 450.828 / 11344.172 / 22489.473 | 9.506 | 605.326 / 1546.242 / 2903.871 | 421.850 / 881.835 / 1406.842 | 2289.001 / 19300.610 / 28558.473 |
| `1x` | 15.080 | 411.022 / 9778.554 / 20922.863 | 9.721 | 464.017 / 1044.085 / 2330.303 | 391.748 / 839.921 / 1430.001 | 1974.028 / 16772.057 / 33566.936 |
| `2x` | 14.048 | 516.453 / 9161.568 / 23771.400 | 8.374 | 513.373 / 1003.326 / 2654.430 | 498.388 / 820.145 / 1539.336 | 2623.945 / 17126.192 / 35384.478 |

#### Fault-Injection Details

| External load | Group | Events | Bytes | Latency P50/P95/P99 us | Avg Inst GB/s |
|---:|---|---:|---:|---:|---:|
| `1x` | injector.store.put | 1056 | 50983862272 | 2280.465 / 10779.944 / 17630.387 | 15.080 |
| `1x` | main.overall | 4252 | 100040981504 | 411.022 / 9778.554 / 20922.863 | 9.721 |
| `1x` | main.foreground_pd | 692 | 4172283904 | 391.748 / 839.921 / 1430.001 | 15.288 |
| `1x` | main.foreground_pd.overlap_store_put | 92 | 554696704 | 464.017 / 1044.085 / 2330.303 | 13.040 |
| `1x` | main.foreground_pd.no_store_put | 600 | 3617587200 | 386.482 / 792.983 / 1113.030 | 15.632 |
| `1x` | main.store.put | 956 | 45123371008 | 1974.028 / 16772.057 / 33566.936 | 15.046 |
| `2x` | injector.store.put | 2112 | 101967724544 | 2503.754 / 13338.609 / 24343.803 | 14.048 |
| `2x` | main.overall | 4252 | 100040981504 | 516.453 / 9161.568 / 23771.400 | 8.374 |
| `2x` | main.foreground_pd | 692 | 4172283904 | 498.388 / 820.145 / 1539.336 | 13.243 |
| `2x` | main.foreground_pd.overlap_store_put | 98 | 590872576 | 513.373 / 1003.326 / 2654.430 | 11.713 |
| `2x` | main.foreground_pd.no_store_put | 594 | 3581411328 | 490.756 / 803.266 / 1421.604 | 13.495 |
| `2x` | main.store.put | 956 | 45123371008 | 2623.945 / 17126.192 / 35384.478 | 12.655 |

### Interpretation

The three-mode comparison shows why intent alone is not enough: Mode 2 attaches
semantic intent, but foreground and background traffic still share the same RDMA
resources. Mode 3 uses the intent to place foreground PD traffic in the `fg`
QP pool and Store traffic in the `bg` QP pool, giving foreground traffic a
separate RDMA/QoS lane.

The 2026-08-19 background-load sweep confirms the same story under changing
background pressure. As Store traffic scales from `0.5x` to `2x`,
foreground-overlap P99 stays in the `2.5-3.0 ms` range:

```text
0.5x Store foreground overlap P99: 2744.306 us
1.0x Store foreground overlap P99: 2969.827 us
2.0x Store foreground overlap P99: 2504.047 us
```

This is still far below the non-isolated Mode 1/Mode 2 foreground-overlap P99
from the three-mode comparison:

```text
Mode 1 foreground overlap P99: 10710.299 us
Mode 2 foreground overlap P99: 21194.583 us
```

In short, Mode 3 keeps user-facing foreground transfers in the low-millisecond
tail-latency range even as background Store traffic increases. The expected
tradeoff remains visible: Store put tail latency stays high because the
background class is deliberately assigned to the smaller, lower-TC QP pool.

The external-traffic fault-injection run strengthens the same claim under a
different pressure model. Even with an independent Store-only injector sending
about `14-15 GB/s`, the main Mode 3 foreground-overlap P99 stays between
`2.3 ms` and `2.7 ms`. The cost again falls mostly on aggregate bandwidth and
Store traffic: under `2x` external load, main overall average instantaneous
bandwidth drops to `8.374 GB/s`, and main Store put P99 rises to `35.384 ms`.
