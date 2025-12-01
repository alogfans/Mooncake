#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

import argparse
import torch, time, csv, numpy as np, os
from dataclasses import dataclass
from time import perf_counter_ns
from kvbench_backend import RDMAEngineBackend, NIXLBackend
from kvbench_common import recv_handshake

@dataclass
class XferStat:
    start: int
    end: int
    nbytes: int
    def dur_ms(self): return (self.end - self.start) / 1e6


def apply_fault_env(fault_envs):
    """Parse KEY=VALUE pairs and set them in os.environ"""
    for item in fault_envs:
        if "=" not in item:
            print(f"[WARN] Invalid fault env format: {item}, skipped.")
            continue
        k, v = item.split("=", 1)
        os.environ[k.strip()] = v.strip()
        print(f"[Env] SET {k.strip()} = {v.strip()}")
    if fault_envs:
        print(f"[Env] Fault injection environment applied ({len(fault_envs)} vars).")


def clear_fault_env(fault_envs):
    """Clear the env keys specified in fault_envs from os.environ"""
    for item in fault_envs:
        if "=" in item:
            k, _ = item.split("=", 1)
            key = k.strip()
        else:
            key = item.strip()

        if not key:
            continue

        if key in os.environ:
            del os.environ[key]
            print(f"[Env] UNSET {key}")
        else:
            print(f"[Env] {key} not set, skip UNSET")

    if fault_envs:
        print(f"[Env] Fault injection environment cleared ({len(fault_envs)} vars).")


def run_single_proc(remote: str, zmq_port: int,
                    backend: str, buf_size: int,
                    block_bytes: int, warmup_s: float,
                    duration_s: float, csv_path: str,
                    interval_ms: float, fault_env):
    print(f"[Main] Connecting to {remote}:{zmq_port} ...")
    handshake = recv_handshake(remote, default_port=zmq_port)
    target = handshake["remote"]
    dst_addrs = handshake["buf_addrs"]
    dst_base = dst_addrs[0]

    torch.cuda.set_device(0)
    engine = RDMAEngineBackend() if backend == "mooncake" else NIXLBackend()
    if backend == "nixl":
        engine.attach_remote(handshake)

    buf = torch.ones(buf_size // 4, dtype=torch.float32, device="cpu")
    src = buf.data_ptr()
    engine.register_memory(src, buf_size)

    srcs, dsts, lens = [], [], []
    offset = 0
    srcs.append(src + offset)
    dsts.append(dst_base + offset)
    lens.append(block_bytes)
    offset += block_bytes

    # 每次 transfer_write 的总传输字节数
    bytes_per_iter = int(sum(lens))

    print(f"[Main] Using GPU0, {len(srcs)} blocks × {block_bytes/1e6:.2f} MB")
    print(f"[Main] Sampling interval = {interval_ms} ms")
    print(f"[Main] Bytes per iteration = {bytes_per_iter/1e6:.2f} MB")

    print(f"[Main] Warmup for {warmup_s:.1f}s ...")
    t_end_warmup = time.time() + warmup_s
    while time.time() < t_end_warmup:
        engine.transfer_write(target, srcs, dsts, lens)
    print("[Main] Warmup done.")

    print(f"[Main] Start measurement for {duration_s:.1f}s ...")
    t_start = time.time()
    t_end = t_start + duration_s
    t_next_sample = t_start + interval_ms / 1000.0

    # window_stats 现在存的是吞吐（GB/s）
    window_stats = []
    csv_records = []

    # 标记：第 2 秒注入 env，第 4 秒清除 env（当前具体设为 0.5s/2.5s）
    fault_applied = False
    fault_cleared = False
    inject_time = 1.0   # s after t_start
    clear_time = 3.0    # s after t_start

    while True:
        st = perf_counter_ns()
        engine.transfer_write(target, srcs, dsts, lens)
        ed = perf_counter_ns()

        # 单次传输耗时（毫秒）
        dur_ms = (ed - st) / 1e6
        if dur_ms > 0:
            # 吞吐量：GB/s（基于字节，1 GB = 1e9 bytes）
            throughput_gbps = (bytes_per_iter / (dur_ms / 1000.0)) / 1e9
            window_stats.append(throughput_gbps)

        now = time.time()
        elapsed = now - t_start

        # 在正式执行的第 2 秒注入环境变量（当前设为 inject_time）
        if (not fault_applied) and elapsed >= inject_time and fault_env:
            print(f"[Main] Elapsed {elapsed:.3f}s >= {inject_time:.1f}s, applying fault envs.")
            apply_fault_env(fault_env)
            fault_applied = True

        # 在正式执行的第 4 秒清除环境变量（当前设为 clear_time）
        if fault_applied and (not fault_cleared) and elapsed >= clear_time and fault_env:
            print(f"[Main] Elapsed {elapsed:.3f}s >= {clear_time:.1f}s, clearing fault envs.")
            clear_fault_env(fault_env)
            fault_cleared = True

        # 到采样时间点，计算窗口内平均吞吐
        if now >= t_next_sample:
            if window_stats:
                avg_thr = float(np.mean(window_stats))  # GB/s
                elapsed_ms = (now - t_start) * 1000
                csv_records.append((elapsed_ms, avg_thr))
                window_stats.clear()
            t_next_sample += interval_ms / 1000.0

        if now >= t_end:
            break

    print(f"[Main] Writing {len(csv_records)} samples to {csv_path}")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_ms", "avg_throughput_GBps"])
        w.writerows(csv_records)

    all_thr = [thr for _, thr in csv_records]
    if all_thr:
        print(f"[Main] Summary: samples={len(all_thr)}, "
              f"avg={np.mean(all_thr):.3f} GB/s, "
              f"min={np.min(all_thr):.3f} GB/s, "
              f"max={np.max(all_thr):.3f} GB/s")
    else:
        print("[Main] Summary: no samples recorded.")
    print("[Main] Done.")


def main():
    ap = argparse.ArgumentParser(
        description="Single-process KVCache throughput microbenchmark with fault injection."
    )
    ap.add_argument("--remote", type=str, default="127.0.0.1",
                    help="Remote Mooncake/NIXL server address")
    ap.add_argument("--zmq_port", type=int, default=5555,
                    help="ZeroMQ handshake port")
    ap.add_argument("--backend", choices=["mooncake", "nixl"], default="mooncake",
                    help="Transfer backend to use")
    ap.add_argument("--buf_size", type=int, default=512 * 1024 * 1024,
                    help="GPU buffer size (default: 512MB)")
    ap.add_argument("--block_mb", type=float, default=128.0,
                    help="Transfer block size in MB (default: 64MB)")
    ap.add_argument("--warmup", type=float, default=5.0,
                    help="Warmup duration in seconds (default: 5s)")
    ap.add_argument("--duration", type=float, default=5.0,
                    help="Measurement duration in seconds (default: 5s)")
    ap.add_argument("--interval_ms", type=float, default=5.0,
                    help="Sampling interval in milliseconds (default: 10ms)")
    ap.add_argument("--csv", type=str, default="latency_samples.csv",
                    help="Output CSV path (default: latency_samples.csv)")
    ap.add_argument("--fault-env", nargs="*", default=[],
                    help="Set fault injection env vars, e.g. MC_FAULT_MODE=1 MC_FAULT_DELAY=5ms")
    args = ap.parse_args()

    # 不再一开始就设置 env，而是传入 run_single_proc，在第 2/4 秒动态切换
    run_single_proc(
        args.remote, args.zmq_port, args.backend,
        args.buf_size, int(args.block_mb * 1024 * 1024),
        args.warmup, args.duration, args.csv, args.interval_ms,
        args.fault_env
    )


if __name__ == "__main__":
    main()
