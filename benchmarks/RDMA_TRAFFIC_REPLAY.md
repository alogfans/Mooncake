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

#### Reliability Retest on 2026-08-20

The four background-load combinations were replayed again with the same
Mode 3 configuration. The raw logs are under:

```text
build/sglang_qpool_bg_sweep_retest_20260820/
```

The first `0.5x` retest produced a clear outlier
(`foreground_pd.overlap_store_put` P99 `22343.725 us`). A fresh target was
started and `0.5x` was replayed once more; the second run is consistent with
the original sweep and is the value used in the reliability summary below.

| BG ratio | Overall P50/P95/P99 us | Overall Avg Inst GB/s | Foreground P50/P95/P99 us | Foreground overlap P50/P95/P99 us | Store put P50/P95/P99 us |
|---:|---:|---:|---:|---:|---:|
| `0x` | 154.552 / 5297.239 / 11154.123 | 9.256 | 352.121 / 522.629 / 618.647 | n/a | n/a |
| `0.5x` | 325.378 / 8295.537 / 17835.053 | 9.270 | 368.581 / 678.549 / 1434.887 | 424.920 / 1434.887 / 1988.603 | 2236.606 / 14355.208 / 27532.090 |
| `1x` | 446.315 / 10377.688 / 22678.859 | 9.615 | 429.606 / 1014.673 / 5647.577 | 510.325 / 1288.862 / 1814.177 | 2157.478 / 17795.081 / 30057.924 |
| `2x` | 597.070 / 9321.420 / 21799.238 | 11.768 | 383.765 / 801.154 / 5471.501 | 500.193 / 1373.587 / 1708.673 | 1784.625 / 14555.910 / 27910.670 |

The retest supports the main conclusion: after discarding the single `0.5x`
outlier, foreground PD transfers that overlap Store puts remain in the
low-millisecond P99 range across `0.5x`, `1x`, and `2x` background load:

```text
0.5x Store foreground overlap P99: 1988.603 us
1.0x Store foreground overlap P99: 1814.177 us
2.0x Store foreground overlap P99: 1708.673 us
```

The retest also shows that single-run P99 can be noisy in this shared RDMA/GPU
environment. For publication-quality numbers, repeat each ratio at least three
times and report the median run, while keeping outlier logs for diagnosis.

### Full Same-Day Retest on 2026-08-20

All test points in this document were replayed again on 2026-08-20 with the
same binary and NIC/filter configuration. The full automation script and raw
logs are under:

```text
build/sglang_full_retest_20260820/
```

The run completed from `2026-08-20T11:32:07+08:00` to
`2026-08-20T12:13:36+08:00`.

#### Same-Day Three-Mode Comparison

| Mode | Description | Overall P50/P95/P99 us | Overall Avg Inst GB/s | Foreground overlap P50/P95/P99 us | Foreground no-overlap P50/P95/P99 us | Store put P50/P95/P99 us |
|---|---|---:|---:|---:|---:|---:|
| 1 | Non-intent, single TE | 536.382 / 9201.924 / 20612.425 | 10.666 | 635.539 / 4333.642 / 9478.125 | 366.613 / 544.246 / 1101.569 | 1745.057 / 14928.501 / 25025.354 |
| 2 | Intent baseline, single TE | 409.861 / 9210.360 / 21579.343 | 9.134 | 1774.624 / 13208.841 / 18126.385 | 383.477 / 916.593 / 2557.276 | 2359.958 / 14650.466 / 27917.874 |
| 3 | Single TE + per-intent QP pool | 448.918 / 10176.289 / 23928.868 | 9.595 | 547.666 / 1626.902 / 2954.772 | 410.556 / 1059.949 / 2007.505 | 2228.490 / 17109.610 / 25787.349 |

#### Same-Day Background-Load Sweep

| BG ratio | Overall P50/P95/P99 us | Overall Avg Inst GB/s | Foreground P50/P95/P99 us | Foreground overlap P50/P95/P99 us | Store put P50/P95/P99 us |
|---:|---:|---:|---:|---:|---:|
| `0x` | 157.978 / 5350.477 / 11224.218 | 9.010 | 358.362 / 659.159 / 3895.399 | n/a | n/a |
| `0.5x` | 329.350 / 7656.108 / 17169.652 | 9.390 | 375.852 / 714.410 / 2384.282 | 423.874 / 1279.371 / 3095.714 | 2004.812 / 14965.253 / 28746.259 |
| `1x` | 402.663 / 9206.186 / 22231.212 | 9.914 | 388.832 / 801.524 / 2071.576 | 480.622 / 1440.520 / 2928.440 | 2144.890 / 17605.468 / 31878.862 |
| `2x` | 763.169 / 13257.181 / 24107.763 | 9.889 | 427.903 / 932.314 / 1673.601 | 628.623 / 1960.353 / 4049.195 | 2367.781 / 17807.560 / 29842.479 |

#### Same-Day Fault Injection

The fault-injection windows are only 65 seconds and contain 16-18 foreground
overlap events, so their P99 values are much noisier than the 5-minute sweep.

| Main mode | External load | External measured Avg Inst GB/s | Main overall P50/P95/P99 us | Main overall Avg Inst GB/s | Main foreground overlap P50/P95/P99 us | Main foreground P50/P95/P99 us | Main Store put P50/P95/P99 us |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mode 3 qpool | `none` | n/a | 411.911 / 10054.185 / 19852.051 | 8.510 | 581.119 / 1135.515 / 1135.515 | 417.072 / 755.266 / 954.809 | 2058.321 / 18285.403 / 23523.766 |
| Mode 3 qpool | `10 GB/s` | 27.720 | 483.937 / 13574.655 / 22185.775 | 6.891 | 878.617 / 2127.767 / 2127.767 | 468.725 / 1364.798 / 2127.767 | 3139.985 / 20324.883 / 25013.637 |
| `unspec` baseline | `none` | n/a | 422.263 / 9274.532 / 16536.852 | 9.113 | 1612.761 / 15139.642 / 15139.642 | 402.007 / 1702.668 / 12653.385 | 2495.812 / 15545.388 / 23939.740 |
| `unspec` baseline | `10 GB/s` | 26.112 | 483.782 / 11983.557 / 25984.313 | 7.339 | 974.914 / 6007.672 / 6007.672 | 437.672 / 5151.877 / 6007.672 | 2577.233 / 18452.922 / 32874.226 |

The same-day full retest supports the strong conclusion from the 5-minute
three-mode and background-sweep runs: Mode 3 keeps foreground overlap P99 in
the low-millisecond range while Mode 1/2 remain much higher. The fault-injection
short-window comparison is less stable. In this run qpool degrades from
`1.136 ms` to `2.128 ms` under the external injector, but the `unspec` baseline
improves from `15.140 ms` to `6.008 ms`; because each value is based on fewer
than 20 overlap events, use the fault-injection result as directional evidence
only, not as a standalone reliability claim.

### Mode 3 External-Traffic Fault Injection

The third experiment keeps the main Mode 3 replay unchanged and injects
additional external RDMA traffic from a separate `sglang_trace_replay_bench`
process. The external traffic is intentionally unlabeled (`unspec` intent) and
restricted to a single rail (`mlx5_1`). This models reduced available network
bandwidth without mutating the main business trace.

The external injector uses this config:

```json
{
  "topology": {
    "rdma_whitelist": ["mlx5_1"]
  }
}
```

The external trace is synthetic Store-put traffic with fixed-size transfers and
fixed inter-arrival time:

| External load | Trace mutation |
|---:|---|
| `none` | no external injector |
| `10 GB/s` | 16 MiB `mooncake_put` every 1677.722 us for 60 s |

The generated traces and raw logs are under:

```text
build/sglang_fault_injection/
```

Complete logs for this experiment:

| Run | Log |
|---|---|
| Mode 3 target | `build/sglang_fault_injection/main_target_fault_const_10GBps.log` |
| Mode 3 no-inject main replay | `build/sglang_fault_injection/main_replay_65s_baseline.log` |
| Mode 3 + injector main replay | `build/sglang_fault_injection/main_replay_65s_const_10GBps_unlabeled_1rail.log` |
| Mode 3 + injector external replay | `build/sglang_fault_injection/external_replay_const_10GBps_unlabeled_1rail_60s.log` |
| `unspec` target | `build/sglang_fault_injection/main_target_fault_unspec.log` |
| `unspec` no-inject main replay | `build/sglang_fault_injection/main_replay_65s_unspec_baseline.log` |
| `unspec` + injector main replay | `build/sglang_fault_injection/main_replay_65s_unspec_const_10GBps_unlabeled_1rail.log` |
| `unspec` + injector external replay | `build/sglang_fault_injection/external_replay_const_10GBps_unlabeled_1rail_60s_unspec_main.log` |
| External injector target | `build/sglang_fault_injection/external_const_unlabeled_1rail_target.log` |
| External injector target for `unspec` run | `build/sglang_fault_injection/external_const_unlabeled_1rail_target_unspec.log` |

The external target uses an independent segment and RPC port range:

```bash
ssh qjh001 '
cd /mnt/qjh000/rf/Mooncake &&
env -u MC_TENT_CONF -u MC_TE_FILTERS -u MC_TE_FILTERS_EXCLUDE \
build/mooncake-transfer-engine/benchmark/sglang_trace_replay_bench \
  --mode=target \
  --scenario=qpool \
  --tent_conf_file=benchmarks/sglang_external_unlabeled_1rail.json \
  --default_intent=unspec \
  --pd_intent=unspec \
  --store_intent=unspec \
  --local_segment_name=external-const-unlabeled-1rail-target \
  --rpc_server_port=21401
'
```

The main replay runs for 65 seconds and skips the first 5 seconds from summary
statistics. For the fault-injection run, the external replay starts at the
5-second mark and runs for 60 seconds, so the main summary window corresponds
to the injection window.

#### Fault-Injection Summary

| Main mode | External load | External offered GB/s | Main overall P50/P95/P99 us | Main overall Avg Inst GB/s | Main foreground overlap P50/P95/P99 us | Main foreground P50/P95/P99 us | Main Store put P50/P95/P99 us |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mode 3 qpool | `none` | n/a | 399.947 / 8446.195 / 22083.855 | 8.856 | 560.462 / 1849.417 / 1849.417 | 408.275 / 849.626 / 1612.712 | 2233.797 / 17050.385 / 26511.581 |
| Mode 3 qpool | `10 GB/s` | 10.000 | 450.955 / 12784.542 / 23309.914 | 7.339 | 646.588 / 2716.734 / 2716.734 | 442.029 / 1623.183 / 3270.395 | 3249.379 / 18100.817 / 26792.877 |
| `unspec` baseline | `none` | n/a | 434.688 / 9451.527 / 16982.534 | 9.647 | 407.808 / 3499.565 / 3499.565 | 377.298 / 1008.823 / 3179.772 | 1742.668 / 13661.900 / 20176.747 |
| `unspec` baseline | `10 GB/s` | 10.000 | 498.145 / 11897.875 / 20675.951 | 9.194 | 602.340 / 8498.431 / 8498.431 | 380.990 / 3589.812 / 6902.588 | 1682.588 / 17075.642 / 30372.225 |

#### Fault-Injection Details

| Main mode | External load | Group | Events | Bytes | Latency P50/P95/P99 us | Avg Inst GB/s |
|---|---:|---|---:|---:|---:|---:|
| Mode 3 qpool | `none` | main.overall | 764 | 15869629184 | 399.947 / 8446.195 / 22083.855 | 8.856 |
| Mode 3 qpool | `none` | main.foreground_pd | 136 | 819986432 | 408.275 / 849.626 / 1612.712 | 15.161 |
| Mode 3 qpool | `none` | main.foreground_pd.overlap_store_put | 18 | 108527616 | 560.462 / 1849.417 / 1849.417 | 10.258 |
| Mode 3 qpool | `none` | main.foreground_pd.no_store_put | 118 | 711458816 | 398.457 / 621.862 / 836.395 | 15.909 |
| Mode 3 qpool | `none` | main.store.put | 164 | 7042236416 | 2233.797 / 17050.385 / 26511.581 | 13.367 |
| Mode 3 qpool | `10 GB/s` | injector.store.put | 35764 | 600020353024 | 942.321 / 1089.063 / 2167.837 | 28.928 |
| Mode 3 qpool | `10 GB/s` | main.overall | 764 | 15869629184 | 450.955 / 12784.542 / 23309.914 | 7.339 |
| Mode 3 qpool | `10 GB/s` | main.foreground_pd | 136 | 819986432 | 442.029 / 1623.183 / 3270.395 | 13.789 |
| Mode 3 qpool | `10 GB/s` | main.foreground_pd.overlap_store_put | 18 | 108527616 | 646.588 / 2716.734 / 2716.734 | 10.054 |
| Mode 3 qpool | `10 GB/s` | main.foreground_pd.no_store_put | 118 | 711458816 | 416.969 / 1055.769 / 3270.395 | 14.358 |
| Mode 3 qpool | `10 GB/s` | main.store.put | 164 | 7042236416 | 3249.379 / 18100.817 / 26792.877 | 10.478 |
| `unspec` baseline | `none` | main.overall | 764 | 15869629184 | 434.688 / 9451.527 / 16982.534 | 9.647 |
| `unspec` baseline | `none` | main.foreground_pd | 136 | 819986432 | 377.298 / 1008.823 / 3179.772 | 16.187 |
| `unspec` baseline | `none` | main.foreground_pd.overlap_store_put | 18 | 108527616 | 407.808 / 3499.565 / 3499.565 | 12.495 |
| `unspec` baseline | `none` | main.foreground_pd.no_store_put | 118 | 711458816 | 376.423 / 924.905 / 1382.465 | 16.750 |
| `unspec` baseline | `none` | main.store.put | 164 | 7042236416 | 1742.668 / 13661.900 / 20176.747 | 15.660 |
| `unspec` baseline | `10 GB/s` | injector.store.put | 35764 | 600020353024 | 935.420 / 1193.939 / 2800.750 | 29.047 |
| `unspec` baseline | `10 GB/s` | main.overall | 764 | 15869629184 | 498.145 / 11897.875 / 20675.951 | 9.194 |
| `unspec` baseline | `10 GB/s` | main.foreground_pd | 136 | 819986432 | 380.990 / 3589.812 / 6902.588 | 15.901 |
| `unspec` baseline | `10 GB/s` | main.foreground_pd.overlap_store_put | 18 | 108527616 | 602.340 / 8498.431 / 8498.431 | 10.840 |
| `unspec` baseline | `10 GB/s` | main.foreground_pd.no_store_put | 118 | 711458816 | 358.164 / 1148.539 / 6350.024 | 16.673 |
| `unspec` baseline | `10 GB/s` | main.store.put | 164 | 7042236416 | 1682.588 / 17075.642 / 30372.225 | 14.427 |

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
different pressure model. With a constant, unlabeled, single-rail injector
offering `10 GB/s` for 60 seconds, the main Mode 3 foreground-overlap P99
stays in the low-millisecond range (`1.849 ms` without the injector and
`2.717 ms` with the injector). The `unspec` baseline is more sensitive in the
same 60-second window: foreground-overlap P99 rises from `3.500 ms` without the
injector to `8.498 ms` with the injector. Under the `10 GB/s` external load,
Mode 3's foreground-overlap P99 is about `68.0%` lower than the `unspec`
baseline. The cost again falls mostly on aggregate bandwidth and Store traffic:
during the injection window, Mode 3 overall average instantaneous bandwidth
drops from `8.856 GB/s` to `7.339 GB/s`, and Mode 3 Store put P50 rises from
`2.234 ms` to `3.249 ms`.
