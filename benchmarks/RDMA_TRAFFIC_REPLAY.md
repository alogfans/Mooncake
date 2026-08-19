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

- Date: 2026-08-18
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
cmake --build build-sglang-replay-cuda --target sglang_trace_replay_bench -j16
```

Binary:

```bash
build-sglang-replay-cuda/mooncake-transfer-engine/benchmark/sglang_trace_replay_bench
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
build-sglang-replay-cuda/mooncake-transfer-engine/benchmark/sglang_trace_replay_bench \
  --mode=target \
  --scenario=nonintent
'
```

Run replay locally:

```bash
env -u MC_TENT_CONF -u MC_TE_FILTERS -u MC_TE_FILTERS_EXCLUDE \
build-sglang-replay-cuda/mooncake-transfer-engine/benchmark/sglang_trace_replay_bench \
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
build-sglang-replay-cuda/mooncake-transfer-engine/benchmark/sglang_trace_replay_bench \
  --mode=target \
  --scenario=intent
'
```

Run replay locally:

```bash
env -u MC_TENT_CONF -u MC_TE_FILTERS -u MC_TE_FILTERS_EXCLUDE \
build-sglang-replay-cuda/mooncake-transfer-engine/benchmark/sglang_trace_replay_bench \
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
build-sglang-replay-cuda/mooncake-transfer-engine/benchmark/sglang_trace_replay_bench \
  --mode=target \
  --scenario=qpool
'
```

Run replay locally:

```bash
env -u MC_TENT_CONF -u MC_TE_FILTERS -u MC_TE_FILTERS_EXCLUDE \
build-sglang-replay-cuda/mooncake-transfer-engine/benchmark/sglang_trace_replay_bench \
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

All three modes use the same trace and 5-minute replay window.

### Summary

| Mode | Description | Overall P50/P95/P99 us | Overall Avg Inst GB/s | Foreground overlap P50/P95/P99 us | Foreground no-overlap P50/P95/P99 us | Store put P50/P95/P99 us |
|---|---|---:|---:|---:|---:|---:|
| 1 | Non-intent, single TE | 513.792 / 9525.474 / 20456.872 | 11.925 | 428.298 / 10283.509 / 18786.008 | 363.558 / 538.359 / 3262.263 | 1500.947 / 14233.027 / 26749.159 |
| 2 | Intent baseline, single TE | 375.298 / 9791.073 / 20131.161 | 10.703 | 1435.184 / 12586.658 / 19325.905 | 368.225 / 617.629 / 4606.575 | 1753.312 / 15121.696 / 25008.167 |
| 3 | Single TE + per-intent QP pool | 463.271 / 9237.326 / 21570.863 | 9.076 | 449.350 / 1083.835 / 1755.693 | 420.126 / 904.170 / 1287.079 | 2424.717 / 14703.433 / 26963.677 |

### Detailed Groups

| Mode | Group | Events | Bytes | Latency P50/P95/P99 us | Avg Inst GB/s |
|---|---|---:|---:|---:|---:|
| 1 | overall | 4252 | 100040981504 | 513.792 / 9525.474 / 20456.872 | 11.925 |
| 1 | pd.data.no_store_put | 1157 | 23429906432 | 471.016 / 2349.902 / 11285.240 | 21.253 |
| 1 | pd.data.overlap_store_put | 491 | 31485067264 | 3523.414 / 18258.181 / 26568.681 | 14.599 |
| 1 | pd.metadata | 1648 | 2636800 | 29.020 / 2304.871 / 6120.474 | 0.044 |
| 1 | store.put | 956 | 45123371008 | 1500.947 / 14233.027 / 26749.159 | 19.742 |
| 1 | foreground_pd | 692 | 4172283904 | 364.753 / 1197.417 / 6465.632 | 17.646 |
| 1 | foreground_pd.overlap_store_put | 80 | 482344960 | 428.298 / 10283.509 / 18786.008 | 14.566 |
| 1 | foreground_pd.no_store_put | 612 | 3689938944 | 363.558 / 538.359 / 3262.263 | 18.048 |
| 2 | overall | 4252 | 100040981504 | 375.298 / 9791.073 / 20131.161 | 10.703 |
| 2 | pd.data.no_store_put | 1111 | 21223178240 | 457.275 / 2766.874 / 12960.007 | 19.892 |
| 2 | pd.data.overlap_store_put | 537 | 33691795456 | 3917.004 / 17515.492 / 24647.455 | 12.134 |
| 2 | pd.metadata | 1648 | 2636800 | 23.630 / 47.629 / 102.722 | 0.066 |
| 2 | store.put | 956 | 45123371008 | 1753.312 / 15121.696 / 25008.167 | 17.559 |
| 2 | foreground_pd | 692 | 4172283904 | 372.366 / 2790.483 / 11238.847 | 16.464 |
| 2 | foreground_pd.overlap_store_put | 80 | 482344960 | 1435.184 / 12586.658 / 19325.905 | 8.189 |
| 2 | foreground_pd.no_store_put | 612 | 3689938944 | 368.225 / 617.629 / 4606.575 | 17.546 |
| 3 | overall | 4252 | 100040981504 | 463.271 / 9237.326 / 21570.863 | 9.076 |
| 3 | pd.data.no_store_put | 1043 | 20198195200 | 669.384 / 2435.340 / 11333.782 | 16.999 |
| 3 | pd.data.overlap_store_put | 605 | 34716778496 | 3431.548 / 16944.850 / 26420.763 | 12.648 |
| 3 | pd.metadata | 1648 | 2636800 | 22.628 / 41.362 / 105.051 | 0.069 |
| 3 | store.put | 956 | 45123371008 | 2424.717 / 14703.433 / 26963.677 | 13.698 |
| 3 | foreground_pd | 692 | 4172283904 | 425.098 / 920.483 / 1386.322 | 13.953 |
| 3 | foreground_pd.overlap_store_put | 98 | 590872576 | 449.350 / 1083.835 / 1755.693 | 12.646 |
| 3 | foreground_pd.no_store_put | 594 | 3581411328 | 420.126 / 904.170 / 1287.079 | 14.169 |

## Interpretation

The key metric is foreground PD latency while Store puts are active:

```text
Mode 1 foreground overlap P99: 18786.008 us
Mode 2 foreground overlap P99: 19325.905 us
Mode 3 foreground overlap P99:  1755.693 us
```

Mode 2 improves metadata latency significantly, but because PD and Store still
share the same TE/RDMA resources, foreground overlap tail latency is not
isolated. Mode 3 keeps one TE per TP, but separates foreground and background
at the QP-pool/traffic-class level. In this run it reduces foreground-overlap
P99 by about 90.7% versus Mode 1 and 90.9% versus Mode 2. The tradeoff is lower
overall average instantaneous bandwidth and higher Store put tail latency,
because background Store traffic is explicitly pushed into the lower-TC,
smaller QP pool.
