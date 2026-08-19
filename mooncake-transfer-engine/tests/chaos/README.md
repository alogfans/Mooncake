# TE/TENT Chaos Runner

`te_chaos.py` runs long-lived `tebench` workloads between two hosts and injects
network faults on RDMA rails. It is intended for correctness and stability
testing, not peak-performance measurement.

Defaults are tailored for `qjh000 -> qjh001`:

- workload initiator: local host
- workload target: `qjh001`
- allowed RDMA devices: `mlx5_1,mlx5_2,mlx5_3,mlx5_4`
- faulted netdevs: `eth1,eth2,eth3,eth4`
- excluded control device/netdev: `mlx5_0` / `eth0`
- metadata mode: P2P handshake

The runner refuses a fault target set that includes `mlx5_0` or `eth0`.

## Build

```bash
cmake --build build --target tebench -j16
```

## Check Environment

```bash
python3 mooncake-transfer-engine/tests/chaos/te_chaos.py doctor \
  --target qjh001
```

`doctor` checks SSH, `tebench`, RDMA link visibility, and passwordless `sudo`
for `tc` / `ip` fault injection. If `--metadata-mode http` is selected, it
also checks that the metadata host can launch the HTTP metadata server.

## Run A Smoke Test

```bash
python3 mooncake-transfer-engine/tests/chaos/te_chaos.py run \
  --target qjh001 \
  --metadata-mode p2p \
  --suite smoke \
  --backend tent-fallback \
  --duration 60 \
  --threads 4 \
  --block-size 65536 \
  --batch-size 8
```

## Run Deterministic Chaos

```bash
python3 mooncake-transfer-engine/tests/chaos/te_chaos.py run \
  --target qjh001 \
  --metadata-mode p2p \
  --suite deterministic \
  --backend all \
  --duration 900 \
  --threads 8 \
  --block-size 65536 \
  --batch-size 16
```

`--backend all` runs:

- `tent-fallback`: TENT with RDMA and TCP fallback enabled
- `tent-rdma-only`: TENT restricted to RDMA
- `classic`: classic TransferEngine backend

The workload uses `tebench --op_type=mix --check_consistency=true` by default.

## Run Random / Soak Chaos

```bash
python3 mooncake-transfer-engine/tests/chaos/te_chaos.py run \
  --target qjh001 \
  --metadata-mode p2p \
  --suite random \
  --backend tent-fallback \
  --seed 12345 \
  --duration 7200
```

Use `--suite soak` for the same random action stream with longer durations.
The seed is recorded in `summary.json` and can be reused to replay the schedule.

HTTP metadata mode is still supported:

```bash
python3 mooncake-transfer-engine/tests/chaos/te_chaos.py run \
  --target qjh001 \
  --metadata-mode http \
  --suite smoke \
  --backend tent-fallback
```

The standalone Python metadata server requires `aiohttp` on the metadata host.

## Cleanup

If a run is interrupted, restore the RDMA rails with:

```bash
python3 mooncake-transfer-engine/tests/chaos/te_chaos.py cleanup \
  --target qjh001
```

This removes root `tc` qdiscs and brings `eth1..eth4` back up on both hosts.

## Outputs

Each run writes under `build/chaos-runs/<timestamp>/` by default:

- `events.jsonl`: action schedule, command results, process lifecycle
- `summary.json`: seed and per-backend exit status
- `*-target.log`: target-side `tebench` output
- `*-initiator.log`: initiator-side `tebench` output
- `*-tent.json`: generated TENT config for device filtering and fallback policy
