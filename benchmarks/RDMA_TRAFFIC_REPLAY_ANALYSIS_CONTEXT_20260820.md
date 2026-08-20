# SGLang RDMA Replay Analysis Context

This handoff file collects the experiment context, raw summary data, and
current interpretation state for continuing the RDMA trace replay analysis in a
separate session.

## Question Under Analysis

We want to evaluate whether enabling the intent-aware mechanism affects
foreground task performance during Store burst overlap.

The main metric for that question is:

```text
[summary][foreground_pd]
[summary][foreground_pd.overlap_store_put]
[summary][foreground_pd.no_store_put]
```

Do not confuse it with the overall large-PD-overlap metric:

```text
[summary][pd.data.overlap_store_put]
```

The latter is dominated by large PD data transfers and remains in the
20-30 ms P99 range even with qpool enabled. The qpool mechanism is intended to
protect foreground/latency-sensitive requests, not to reduce every large
overlapped transfer.

## Repository And Artifacts

- Repository root: `/mnt/qjh000/rf/Mooncake`
- Main benchmark source:
  `mooncake-transfer-engine/benchmark/tools/sglang_trace_replay_bench.cpp`
- Main documentation:
  `benchmarks/RDMA_TRAFFIC_REPLAY.md`
- Full same-day retest script:
  `build/sglang_full_retest_20260820/run_full_retest.sh`
- Full same-day retest logs:
  `build/sglang_full_retest_20260820/`
- Original 2026-08-19 three-mode logs:
  `build/sglang_three_mode_rerun/`
- Original 2026-08-19 qpool sweep logs:
  `build/sglang_qpool_bg_sweep/`
- 2026-08-20 qpool sweep retest logs:
  `build/sglang_qpool_bg_sweep_retest_20260820/`
- Fault-injection logs:
  `build/sglang_fault_injection/`

## Environment

- Replay node: local machine under `/mnt/qjh000/rf/Mooncake`
- Target node: `qjh001`
- Metadata server: `http://qjh001:8080/metadata`
- Trace: `build/rdma_traffic.csv`
- TP size: 4
- Each TP uses GPU buffer `cuda:{tp}`
- DRAM buffers use NUMA node 0: `cpu:0`
- P/D transfer route: GPU-to-GPU `WRITE`
- Mooncake Store put route: GPU-to-DRAM `WRITE`
- Mooncake Store get route: DRAM-to-GPU `READ`
- 5-minute experiments: `duration_s=300`, `stats_skip_s=10`
- Fault injection experiments: `duration_s=65`, `stats_skip_s=5`
- External injector: 60 s synthetic Store-put traffic

Environment variables are unset for every run:

```bash
unset MC_TENT_CONF
unset MC_TE_FILTERS
unset MC_TE_FILTERS_EXCLUDE
```

## Config Files

### Non-Intent

File: `benchmarks/sglang_rdma_blacklist.json`

```json
{
  "topology": {
    "rdma_blacklist": ["mlx5_0"]
  }
}
```

### Intent Baseline

File: `benchmarks/sglang_intent_baseline.json`

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

### Enhanced QPool

File: `benchmarks/sglang_qpool_enhanced.json`

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

### External Injector

File: `benchmarks/sglang_external_unlabeled_1rail.json`

```json
{
  "topology": {
    "rdma_whitelist": ["mlx5_1"]
  }
}
```

## Metric Definitions

- `foreground_pd`: latency-sensitive PD transfers. The log reports
  `foreground_max_bytes=6029312`.
- `foreground_pd.overlap_store_put`: foreground PD transfers whose lifetime
  overlaps Store-put activity.
- `foreground_pd.no_store_put`: foreground PD transfers outside Store-put
  overlap.
- `pd.data.overlap_store_put`: all non-metadata PD data transfers overlapping
  Store-put activity. This includes large transfers and is not the foreground
  latency metric.
- `store.put`: Store put traffic.
- `overall`: all reported transfer events after warmup.

## Key Derived Results

### Foreground Overlap P99 Across Two Days

These are the most relevant numbers for foreground-task isolation.

| Run | nonintent | intent baseline | qpool |
|---|---:|---:|---:|
| 2026-08-19 three-mode | 10.710 ms | 21.195 ms | 2.904 ms |
| 2026-08-20 full retest | 9.478 ms | 18.126 ms | 2.955 ms |

Interpretation: qpool keeps foreground-overlap P99 around 3 ms in both
independent 5-minute runs. Nonintent and intent baseline remain much higher.

### QPool Sweep Foreground Overlap P99

| BG ratio | 2026-08-19 | 2026-08-20 retest | 2026-08-20 full retest |
|---:|---:|---:|---:|
| 0.5x | 2.744 ms | 1.989 ms | 3.096 ms |
| 1x | 2.970 ms | 1.814 ms | 2.928 ms |
| 2x | 2.504 ms | 1.709 ms | 4.049 ms |

Interpretation: qpool foreground-overlap P99 stays in the low-millisecond
range under 0.5x-2x Store background load. There is jitter, but it does not
return to the 9-21 ms range seen in nonintent/intent baseline.

### Overall PD-Data Overlap P99

These are not the foreground-task metric.

| Run | nonintent | intent baseline | qpool |
|---|---:|---:|---:|
| 2026-08-19 three-mode | 31.951 ms | 24.307 ms | 24.365 ms |
| 2026-08-20 full retest | 25.557 ms | 23.837 ms | 33.188 ms |

Interpretation: qpool does not reliably reduce the P99 of all overlapped
large-PD data traffic. It protects foreground traffic; large overlapped PD
transfers are still dominated by bandwidth, queueing, and resource contention.

### Fault-Injection Caveat

The fault-injection windows are 65 seconds and contain only 16-18 foreground
overlap events. Their P99 is too noisy to be a primary conclusion.

| Run | qpool no inject | qpool inject | unspec no inject | unspec inject |
|---|---:|---:|---:|---:|
| 2026-08-19 fault window | 1.849 ms | 2.717 ms | 3.500 ms | 8.498 ms |
| 2026-08-20 full retest | 1.136 ms | 2.128 ms | 15.140 ms | 6.008 ms |

Interpretation: qpool's fault-injection direction is consistently worse under
injection, but the unspec result flips direction on 2026-08-20. Treat this as
directional stress evidence only, not a standalone reliability claim.

## Raw Summary Data

The following sections preserve the raw summary lines for the official runs
used above. Complete logs are available at the paths listed in each section.

### 2026-08-19 Three-Mode Raw Summaries

Log: `build/sglang_three_mode_rerun/replay_nonintent.log`

```text
[summary][overall] events=4252 bytes=100040981504 latency_us(p50/p95/p99)=531.562000/9357.936000/21372.567000 avg_inst_GB/s=11.127608 inst_GB/s(p50/p95/p99)=11.804384/28.126356/40.434842
[summary][pd.data.no_store_put] events=1117 bytes=21596995584 latency_us(p50/p95/p99)=464.814000/2163.680000/12113.412000 avg_inst_GB/s=20.433444 inst_GB/s(p50/p95/p99)=18.871027/36.978301/43.160079
[summary][pd.data.overlap_store_put] events=531 bytes=33317978112 latency_us(p50/p95/p99)=3704.631000/19199.280000/31951.183000 avg_inst_GB/s=13.312445 inst_GB/s(p50/p95/p99)=13.150910/23.907816/38.040303
[summary][pd.metadata] events=1648 bytes=2636800 latency_us(p50/p95/p99)=29.378000/1939.576000/4862.992000 avg_inst_GB/s=0.043270 inst_GB/s(p50/p95/p99)=0.054451/0.073499/0.080004
[summary][store.put] events=956 bytes=45123371008 latency_us(p50/p95/p99)=1637.175000/14182.239000/28191.208000 avg_inst_GB/s=18.148759 inst_GB/s(p50/p95/p99)=17.616273/30.916483/41.502350
[summary][foreground_pd] events=692 bytes=4172283904 latency_us(p50/p95/p99)=366.907000/1053.414000/4236.089000 avg_inst_GB/s=17.409823 inst_GB/s(p50/p95/p99)=16.430434/29.853400/38.386146
[summary][foreground_pd.overlap_store_put] events=78 bytes=470286336 latency_us(p50/p95/p99)=519.318000/9325.965000/10710.299000 avg_inst_GB/s=14.072829 inst_GB/s(p50/p95/p99)=11.458780/38.040303/38.777451
[summary][foreground_pd.no_store_put] events=614 bytes=3701997568 latency_us(p50/p95/p99)=365.062000/543.358000/1184.003000 avg_inst_GB/s=17.833740 inst_GB/s(p50/p95/p99)=16.515360/28.740571/38.210263
```

Log: `build/sglang_three_mode_rerun/replay_intent.log`

```text
[summary][overall] events=4252 bytes=100040981504 latency_us(p50/p95/p99)=368.446000/9219.833000/20640.438000 avg_inst_GB/s=11.860368 inst_GB/s(p50/p95/p99)=12.536568/27.748557/42.966923
[summary][pd.data.no_store_put] events=1165 bytes=23634903040 latency_us(p50/p95/p99)=480.938000/2367.139000/12988.362000 avg_inst_GB/s=21.090164 inst_GB/s(p50/p95/p99)=20.243324/39.303401/45.003732
[summary][pd.data.overlap_store_put] events=483 bytes=31280070656 latency_us(p50/p95/p99)=3432.327000/19065.355000/24307.221000 avg_inst_GB/s=14.220379 inst_GB/s(p50/p95/p99)=12.851155/23.152028/35.818192
[summary][pd.metadata] events=1648 bytes=2636800 latency_us(p50/p95/p99)=23.353000/46.306000/92.346000 avg_inst_GB/s=0.066044 inst_GB/s(p50/p95/p99)=0.068478/0.101317/0.109402
[summary][store.put] events=956 bytes=45123371008 latency_us(p50/p95/p99)=1525.836000/15723.282000/29872.811000 avg_inst_GB/s=19.752048 inst_GB/s(p50/p95/p99)=21.121150/30.567629/44.501957
[summary][foreground_pd] events=692 bytes=4172283904 latency_us(p50/p95/p99)=366.049000/1333.191000/5961.708000 avg_inst_GB/s=17.065988 inst_GB/s(p50/p95/p99)=16.462600/27.732705/38.143304
[summary][foreground_pd.overlap_store_put] events=75 bytes=452198400 latency_us(p50/p95/p99)=1131.722000/11969.799000/21194.583000 avg_inst_GB/s=8.773869 inst_GB/s(p50/p95/p99)=5.327556/24.040798/38.075136
[summary][foreground_pd.no_store_put] events=617 bytes=3720085504 latency_us(p50/p95/p99)=357.531000/547.632000/1260.666000 avg_inst_GB/s=18.073944 inst_GB/s(p50/p95/p99)=16.863746/27.875540/38.143304
```

Log: `build/sglang_three_mode_rerun/replay_qpool.log`

```text
[summary][overall] events=4252 bytes=100040981504 latency_us(p50/p95/p99)=450.828000/11344.172000/22489.473000 avg_inst_GB/s=9.505683 inst_GB/s(p50/p95/p99)=8.408612/27.183842/37.128428
[summary][pd.data.no_store_put] events=1029 bytes=19408355328 latency_us(p50/p95/p99)=573.207000/2493.077000/12462.674000 avg_inst_GB/s=18.221333 inst_GB/s(p50/p95/p99)=16.717162/34.792555/43.342254
[summary][pd.data.overlap_store_put] events=619 bytes=35506618368 latency_us(p50/p95/p99)=4060.342000/16765.045000/24364.997000 avg_inst_GB/s=11.670483 inst_GB/s(p50/p95/p99)=10.552661/22.467736/30.467997
[summary][pd.metadata] events=1648 bytes=2636800 latency_us(p50/p95/p99)=22.268000/42.126000/107.314000 avg_inst_GB/s=0.069325 inst_GB/s(p50/p95/p99)=0.071823/0.099243/0.108482
[summary][store.put] events=956 bytes=45123371008 latency_us(p50/p95/p99)=2289.001000/19300.610000/28558.473000 avg_inst_GB/s=14.989682 inst_GB/s(p50/p95/p99)=13.122976/28.740320/36.968778
[summary][foreground_pd] events=692 bytes=4172283904 latency_us(p50/p95/p99)=421.850000/881.835000/1406.842000 avg_inst_GB/s=15.091548 inst_GB/s(p50/p95/p99)=14.215717/27.710018/34.612230
[summary][foreground_pd.overlap_store_put] events=96 bytes=578813952 latency_us(p50/p95/p99)=605.326000/1546.242000/2903.871000 avg_inst_GB/s=12.455981 inst_GB/s(p50/p95/p99)=9.896919/29.612062/33.341326
[summary][foreground_pd.no_store_put] events=596 bytes=3593469952 latency_us(p50/p95/p99)=402.235000/756.143000/966.026000 avg_inst_GB/s=15.516069 inst_GB/s(p50/p95/p99)=14.958177/27.373986/34.995745
```

### 2026-08-20 Full Same-Day Retest Raw Summaries

Run script:
`build/sglang_full_retest_20260820/run_full_retest.sh`

Run window:
`2026-08-20T11:32:07+08:00` to `2026-08-20T12:13:36+08:00`

Log: `build/sglang_full_retest_20260820/three_mode_nonintent_replay.log`

```text
[summary][overall] events=4252 bytes=100040981504 latency_us(p50/p95/p99)=536.382000/9201.924000/20612.425000 avg_inst_GB/s=10.665795 inst_GB/s(p50/p95/p99)=11.041348/27.445460/40.019033
[summary][pd.data.no_store_put] events=1089 bytes=20668481536 latency_us(p50/p95/p99)=461.972000/2433.075000/13307.109000 avg_inst_GB/s=20.074656 inst_GB/s(p50/p95/p99)=18.463728/36.440576/44.092889
[summary][pd.data.overlap_store_put] events=559 bytes=34246492160 latency_us(p50/p95/p99)=4002.283000/17035.692000/25557.408000 avg_inst_GB/s=12.495084 inst_GB/s(p50/p95/p99)=11.978330/21.167884/34.355446
[summary][pd.metadata] events=1648 bytes=2636800 latency_us(p50/p95/p99)=28.578000/1931.094000/5190.635000 avg_inst_GB/s=0.045405 inst_GB/s(p50/p95/p99)=0.055962/0.073661/0.082220
[summary][store.put] events=956 bytes=45123371008 latency_us(p50/p95/p99)=1745.057000/14928.501000/25025.354000 avg_inst_GB/s=17.186279 inst_GB/s(p50/p95/p99)=16.572386/29.650682/42.304727
[summary][foreground_pd] events=692 bytes=4172283904 latency_us(p50/p95/p99)=370.965000/1106.104000/4195.325000 avg_inst_GB/s=16.927329 inst_GB/s(p50/p95/p99)=16.246435/29.273905/38.378327
[summary][foreground_pd.overlap_store_put] events=82 bytes=494403584 latency_us(p50/p95/p99)=635.539000/4333.642000/9478.125000 avg_inst_GB/s=12.173908 inst_GB/s(p50/p95/p99)=9.349452/34.355446/40.155258
[summary][foreground_pd.no_store_put] events=610 bytes=3677880320 latency_us(p50/p95/p99)=366.613000/544.246000/1101.569000 avg_inst_GB/s=17.566313 inst_GB/s(p50/p95/p99)=16.439394/28.560590/38.282561
```

Log: `build/sglang_full_retest_20260820/three_mode_intent_replay.log`

```text
[summary][overall] events=4252 bytes=100040981504 latency_us(p50/p95/p99)=409.861000/9210.360000/21579.343000 avg_inst_GB/s=9.133994 inst_GB/s(p50/p95/p99)=9.371759/23.383384/34.207169
[summary][pd.data.no_store_put] events=999 bytes=17786470400 latency_us(p50/p95/p99)=489.341000/2819.888000/12586.031000 avg_inst_GB/s=17.497726 inst_GB/s(p50/p95/p99)=16.593493/31.245706/39.291090
[summary][pd.data.overlap_store_put] events=649 bytes=37128503296 latency_us(p50/p95/p99)=4121.413000/16877.029000/23836.874000 avg_inst_GB/s=11.336122 inst_GB/s(p50/p95/p99)=11.131401/19.885962/26.477709
[summary][pd.metadata] events=1648 bytes=2636800 latency_us(p50/p95/p99)=22.104000/49.281000/99.965000 avg_inst_GB/s=0.069901 inst_GB/s(p50/p95/p99)=0.072372/0.103073/0.110627
[summary][store.put] events=956 bytes=45123371008 latency_us(p50/p95/p99)=2359.958000/14650.466000/27917.874000 avg_inst_GB/s=14.524242 inst_GB/s(p50/p95/p99)=13.283141/23.866230/29.335407
[summary][foreground_pd] events=692 bytes=4172283904 latency_us(p50/p95/p99)=389.738000/2825.444000/10283.040000 avg_inst_GB/s=14.606641 inst_GB/s(p50/p95/p99)=15.461439/24.603411/37.317027
[summary][foreground_pd.overlap_store_put] events=84 bytes=506462208 latency_us(p50/p95/p99)=1774.624000/13208.841000/18126.385000 avg_inst_GB/s=7.667278 inst_GB/s(p50/p95/p99)=3.273626/23.666728/27.406587
[summary][foreground_pd.no_store_put] events=608 bytes=3665821696 latency_us(p50/p95/p99)=383.477000/916.593000/2557.276000 avg_inst_GB/s=15.565369 inst_GB/s(p50/p95/p99)=15.706282/24.603411/37.317027
```

Log: `build/sglang_full_retest_20260820/three_mode_qpool_replay.log`

```text
[summary][overall] events=4252 bytes=100040981504 latency_us(p50/p95/p99)=448.918000/10176.289000/23928.868000 avg_inst_GB/s=9.595474 inst_GB/s(p50/p95/p99)=9.075279/24.080233/38.615647
[summary][pd.data.no_store_put] events=1075 bytes=20861419520 latency_us(p50/p95/p99)=636.823000/2493.843000/11942.035000 avg_inst_GB/s=17.865694 inst_GB/s(p50/p95/p99)=16.842877/34.205078/44.328194
[summary][pd.data.overlap_store_put] events=573 bytes=34053554176 latency_us(p50/p95/p99)=3545.671000/19089.038000/33187.974000 avg_inst_GB/s=12.724809 inst_GB/s(p50/p95/p99)=11.513198/22.903021/28.722848
[summary][pd.metadata] events=1648 bytes=2636800 latency_us(p50/p95/p99)=22.230000/39.319000/77.727000 avg_inst_GB/s=0.071386 inst_GB/s(p50/p95/p99)=0.071972/0.103040/0.113911
[summary][store.put] events=956 bytes=45123371008 latency_us(p50/p95/p99)=2228.490000/17109.610000/25787.349000 avg_inst_GB/s=14.838260 inst_GB/s(p50/p95/p99)=13.260474/24.308493/37.974775
[summary][foreground_pd] events=692 bytes=4172283904 latency_us(p50/p95/p99)=424.370000/1151.320000/2030.310000 avg_inst_GB/s=14.245655 inst_GB/s(p50/p95/p99)=14.207610/24.731378/37.543351
[summary][foreground_pd.overlap_store_put] events=88 bytes=530579456 latency_us(p50/p95/p99)=547.666000/1626.902000/2954.772000 avg_inst_GB/s=11.566436 inst_GB/s(p50/p95/p99)=10.985775/22.898324/38.254629
[summary][foreground_pd.no_store_put] events=604 bytes=3641704448 latency_us(p50/p95/p99)=410.556000/1059.949000/2007.505000 avg_inst_GB/s=14.636005 inst_GB/s(p50/p95/p99)=14.682219/25.002019/37.139798
```

## QPool Background Sweep Summary Tables

### 2026-08-19 Sweep

Raw logs:

- `build/sglang_qpool_bg_sweep/replay_qpool_bg_0x.log`
- `build/sglang_qpool_bg_sweep/replay_qpool_bg_0.5x.log`
- `build/sglang_qpool_bg_sweep/replay_qpool_bg_1x.log`
- `build/sglang_qpool_bg_sweep/replay_qpool_bg_2x.log`

| BG ratio | Overall P50/P95/P99 us | Overall Avg Inst GB/s | Foreground P50/P95/P99 us | Foreground overlap P50/P95/P99 us | PD data overlap P50/P95/P99 us | Store put P50/P95/P99 us |
|---:|---:|---:|---:|---:|---:|---:|
| 0x | 164.799 / 5694.080 / 11369.625 | 8.913 | 376.588 / 671.148 / 1848.576 | n/a | n/a | n/a |
| 0.5x | 370.445 / 7555.016 / 19180.804 | 8.902 | 436.017 / 911.903 / 2169.208 | 423.470 / 1112.425 / 2744.306 | 3705.451 / 16839.453 / 28284.705 | 2327.048 / 13996.307 / 23462.938 |
| 1x | 385.139 / 9472.040 / 21491.452 | 10.977 | 380.381 / 775.577 / 1142.370 | 454.457 / 1479.991 / 2969.827 | 3169.146 / 19295.799 / 24131.744 | 1755.582 / 16229.208 / 27729.423 |
| 2x | 624.798 / 10190.685 / 23118.628 | 11.567 | 395.648 / 1083.424 / 5927.868 | 522.480 / 1636.366 / 2504.047 | 3310.992 / 17998.466 / 31744.707 | 1798.106 / 14631.067 / 27896.365 |

### 2026-08-20 Sweep Retest

Raw logs:

- `build/sglang_qpool_bg_sweep_retest_20260820/replay_qpool_bg_0x.log`
- `build/sglang_qpool_bg_sweep_retest_20260820/replay_qpool_bg_0.5x.log`
- `build/sglang_qpool_bg_sweep_retest_20260820/replay_qpool_bg_0.5x_r2.log`
- `build/sglang_qpool_bg_sweep_retest_20260820/replay_qpool_bg_1x.log`
- `build/sglang_qpool_bg_sweep_retest_20260820/replay_qpool_bg_2x.log`

The first 0.5x run is an outlier for foreground overlap:
`foreground_pd.overlap_store_put` P99 `22343.725 us`. The second 0.5x run is
the value used for the reliability table.

| BG ratio | Overall P50/P95/P99 us | Overall Avg Inst GB/s | Foreground P50/P95/P99 us | Foreground overlap P50/P95/P99 us | PD data overlap P50/P95/P99 us | Store put P50/P95/P99 us |
|---:|---:|---:|---:|---:|---:|---:|
| 0x | 154.552 / 5297.239 / 11154.123 | 9.256 | 352.121 / 522.629 / 618.647 | n/a | n/a | n/a |
| 0.5x outlier | 401.001 / 15015.554 / 22212.027 | 6.470 | 484.950 / 11830.528 / 19930.733 | 1066.100 / 18741.353 / 22343.725 | 6152.653 / 19400.229 / 31205.871 | 4185.497 / 20123.270 / 32186.130 |
| 0.5x r2 | 325.378 / 8295.537 / 17835.053 | 9.270 | 368.581 / 678.549 / 1434.887 | 424.920 / 1434.887 / 1988.603 | 3840.083 / 15155.314 / 24408.581 | 2236.606 / 14355.208 / 27532.090 |
| 1x | 446.315 / 10377.688 / 22678.859 | 9.615 | 429.606 / 1014.673 / 5647.577 | 510.325 / 1288.862 / 1814.177 | 3608.522 / 18363.291 / 27223.812 | 2157.478 / 17795.081 / 30057.924 |
| 2x | 597.070 / 9321.420 / 21799.238 | 11.768 | 383.765 / 801.154 / 5471.501 | 500.193 / 1373.587 / 1708.673 | 3314.504 / 16639.949 / 25350.935 | 1784.625 / 14555.910 / 27910.670 |

### 2026-08-20 Full Retest Sweep

Raw logs:

- `build/sglang_full_retest_20260820/bg_sweep_0x_replay.log`
- `build/sglang_full_retest_20260820/bg_sweep_0.5x_replay.log`
- `build/sglang_full_retest_20260820/bg_sweep_1x_replay.log`
- `build/sglang_full_retest_20260820/bg_sweep_2x_replay.log`

| BG ratio | Overall P50/P95/P99 us | Overall Avg Inst GB/s | Foreground P50/P95/P99 us | Foreground overlap P50/P95/P99 us | PD data overlap P50/P95/P99 us | Store put P50/P95/P99 us |
|---:|---:|---:|---:|---:|---:|---:|
| 0x | 157.978 / 5350.477 / 11224.218 | 9.010 | 358.362 / 659.159 / 3895.399 | n/a | n/a | n/a |
| 0.5x | 329.350 / 7656.108 / 17169.652 | 9.390 | 375.852 / 714.410 / 2384.282 | 423.874 / 1279.371 / 3095.714 | 3852.000 / 13616.144 / 22344.399 | 2004.812 / 14965.253 / 28746.259 |
| 1x | 402.663 / 9206.186 / 22231.212 | 9.914 | 388.832 / 801.524 / 2071.576 | 480.622 / 1440.520 / 2928.440 | 3534.211 / 17751.107 / 29971.989 | 2144.890 / 17605.468 / 31878.862 |
| 2x | 763.169 / 13257.181 / 24107.763 | 9.889 | 427.903 / 932.314 / 1673.601 | 628.623 / 1960.353 / 4049.195 | 4241.282 / 20828.471 / 26744.936 | 2367.781 / 17807.560 / 29842.479 |

## Fault Injection Summary Tables

### 2026-08-19 Fault Injection

Raw logs:

- qpool no-inject: `build/sglang_fault_injection/main_replay_65s_baseline.log`
- qpool inject: `build/sglang_fault_injection/main_replay_65s_const_10GBps_unlabeled_1rail.log`
- unspec no-inject: `build/sglang_fault_injection/main_replay_65s_unspec_baseline.log`
- unspec inject:
  `build/sglang_fault_injection/main_replay_65s_unspec_const_10GBps_unlabeled_1rail.log`
- qpool injector:
  `build/sglang_fault_injection/external_replay_const_10GBps_unlabeled_1rail_60s.log`
- unspec injector:
  `build/sglang_fault_injection/external_replay_const_10GBps_unlabeled_1rail_60s_unspec_main.log`

| Main mode | External load | External measured Avg Inst GB/s | Main overall P50/P95/P99 us | Main overall Avg Inst GB/s | Foreground overlap P50/P95/P99 us | Foreground P50/P95/P99 us | Store put P50/P95/P99 us |
|---|---:|---:|---:|---:|---:|---:|---:|
| qpool | none | n/a | 399.947 / 8446.195 / 22083.855 | 8.856 | 560.462 / 1849.417 / 1849.417 | 408.275 / 849.626 / 1612.712 | 2233.797 / 17050.385 / 26511.581 |
| qpool | 10 GB/s | 28.928 | 450.955 / 12784.542 / 23309.914 | 7.339 | 646.588 / 2716.734 / 2716.734 | 442.029 / 1623.183 / 3270.395 | 3249.379 / 18100.817 / 26792.877 |
| unspec | none | n/a | 434.688 / 9451.527 / 16982.534 | 9.647 | 407.808 / 3499.565 / 3499.565 | 377.298 / 1008.823 / 3179.772 | 1742.668 / 13661.900 / 20176.747 |
| unspec | 10 GB/s | 29.047 | 498.145 / 11897.875 / 20675.951 | 9.194 | 602.340 / 8498.431 / 8498.431 | 380.990 / 3589.812 / 6902.588 | 1682.588 / 17075.642 / 30372.225 |

### 2026-08-20 Full Retest Fault Injection

Raw logs:

- qpool no-inject:
  `build/sglang_full_retest_20260820/fault_qpool_none_main_replay.log`
- qpool inject:
  `build/sglang_full_retest_20260820/fault_qpool_inject_main_replay.log`
- qpool injector:
  `build/sglang_full_retest_20260820/fault_qpool_inject_external_replay.log`
- unspec no-inject:
  `build/sglang_full_retest_20260820/fault_unspec_none_main_replay.log`
- unspec inject:
  `build/sglang_full_retest_20260820/fault_unspec_inject_main_replay.log`
- unspec injector:
  `build/sglang_full_retest_20260820/fault_unspec_inject_external_replay.log`

| Main mode | External load | External measured Avg Inst GB/s | Main overall P50/P95/P99 us | Main overall Avg Inst GB/s | Foreground overlap P50/P95/P99 us | Foreground P50/P95/P99 us | Store put P50/P95/P99 us |
|---|---:|---:|---:|---:|---:|---:|---:|
| qpool | none | n/a | 411.911 / 10054.185 / 19852.051 | 8.510 | 581.119 / 1135.515 / 1135.515 | 417.072 / 755.266 / 954.809 | 2058.321 / 18285.403 / 23523.766 |
| qpool | 10 GB/s | 27.720 | 483.937 / 13574.655 / 22185.775 | 6.891 | 878.617 / 2127.767 / 2127.767 | 468.725 / 1364.798 / 2127.767 | 3139.985 / 20324.883 / 25013.637 |
| unspec | none | n/a | 422.263 / 9274.532 / 16536.852 | 9.113 | 1612.761 / 15139.642 / 15139.642 | 402.007 / 1702.668 / 12653.385 | 2495.812 / 15545.388 / 23939.740 |
| unspec | 10 GB/s | 26.112 | 483.782 / 11983.557 / 25984.313 | 7.339 | 974.914 / 6007.672 / 6007.672 | 437.672 / 5151.877 / 6007.672 | 2577.233 / 18452.922 / 32874.226 |

## Suggested Next Analysis Steps

1. Treat foreground isolation as the primary claim:
   compare `foreground_pd.overlap_store_put` across nonintent, intent, and
   qpool.
2. Treat `pd.data.overlap_store_put` as a separate aggregate data-plane metric,
   not as evidence against foreground protection.
3. For publication-quality confidence, repeat the 5-minute three-mode and
   qpool sweep experiments at least 3 times and report median run plus min/max
   or confidence intervals.
4. Do not use the 65s fault-injection P99 as the main proof because it has only
   16-18 foreground-overlap events.
5. If fault injection remains important, extend the main replay to 300 s and
   align the external injector with the whole stats window so the overlap sample
   count is comparable to the sweep.

## Current Working Conclusion

The most defensible conclusion is:

> Per-intent QP pool does not reliably reduce overall large-PD overlap P99.
> However, it does protect foreground PD transfers during Store overlap: across
> two independent 5-minute three-mode runs, foreground-overlap P99 stays around
> 3 ms with qpool, while nonintent/intent baseline remain around 9-21 ms.

