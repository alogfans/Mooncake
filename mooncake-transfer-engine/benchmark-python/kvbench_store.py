#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import multiprocessing as mp
import time
import torch
import os
import queue
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from time import perf_counter_ns
from collections import deque

from mooncake.store import MooncakeDistributedStore
from kvbench_common import load_trace, Dispatcher, pct, get_local_ip


# ================================
# Store Client Setup
# ================================

def make_store_client(
    local_hostname: str,
    metadata_server: str,
    global_segment_size: int,
    local_buffer_size: int,
    protocol: str,
    device_name: str,
    master_server: str,
) -> MooncakeDistributedStore:
    store = MooncakeDistributedStore()
    ret = store.setup(
        local_hostname,
        metadata_server,
        global_segment_size,
        local_buffer_size,
        protocol,
        device_name,
        master_server,
    )
    if ret:
        raise RuntimeError(f"Store setup failed: ret={ret}")
    return store


# ================================
# Data Structures
# ================================

@dataclass
class XferStat:
    tp: int
    # st 视为“请求提交时间”（完成 sleep / due 对齐后，真正开始发第一个 op 的时刻）
    start: int
    end: int
    write_bytes: int
    read_bytes: int
    read_reqs: int
    read_hits: int
    due: Optional[float]

    # 每个 request 内的读/写操作时延（单位：ms）
    read_latencies: List[float]
    write_latencies: List[float]

    @property
    def dur_ms(self) -> float:
        """
        单个 request 完成所有 block 读写的总时延
        = 从提交（start）到最后一个 op 结束（end）。
        """
        return (self.end - self.start) / 1e6


# ================================
# 构造每个 request 的 hash 访问序列
# ================================

def build_hash_ops_for_event(
    ev: Dict,
    hash_block_tokens: int,
    bytes_per_token: int,
) -> List[Tuple[str, int]]:
    """
    从单条 trace 记录构造访问序列：
      返回 List[(key, nbytes)]
    不提前决定 read/write，只决定：
      - 对哪个 key（hash_<id>）
      - 理论上传输的字节数（nbytes）
    """
    tokens = int(ev.get("input_length", 0))
    hash_ids = ev.get("hash_ids") or []
    ops: List[Tuple[str, int]] = []

    if tokens <= 0 or not hash_ids:
        return ops

    for i, h in enumerate(hash_ids):
        start_tok = i * hash_block_tokens
        if start_tok >= tokens:
            break
        block_tokens = min(hash_block_tokens, tokens - start_tok)
        nbytes = block_tokens * bytes_per_token
        key = f"hash_{h}"
        ops.append((key, nbytes))

    return ops


# ================================
# Run Loop: 实时判断 read / write + 剔除
# ================================

def run_batches(
    store: MooncakeDistributedStore,
    batches: List[Tuple[List[Tuple[str, int]], Optional[float]]],
    buf: torch.Tensor,
    warmup: int,
    result_q: mp.Queue,
    tp_id: int,
    barrier: Optional[mp.Barrier],
    max_total_elems: int,
    deadline_wall: Optional[float],   # 新增：全局截止时间（wall-clock）
):
    """
    batches: List[(ops, due)]
      ops: List[(key, logical_bytes)]
      due: 该 request 对应的 wall-clock 时间

    对每个 (key, nbytes)：
      - 必须先调用一次 store.get_tensor(key)：
          * 若返回非 None：认为「命中」，计入 read_bytes / read_hits
          * 若返回 None 或异常：认为「未命中」，随后执行 store.put_tensor(key)
            并计入 write_bytes
      - 命中率 strictly 由 get_tensor 是否成功决定。
    """
    elem_size = buf.element_size()

    # 客户端侧对象池：只用于容量控制，不再作为命中判定依据
    live_keys: Dict[str, int] = {}     # key -> elems
    live_order: deque[str] = deque()   # FIFO 顺序
    current_elems: int = 0

    def ensure_capacity(required_elems: int):
        nonlocal current_elems

        if max_total_elems <= 0:
            return

        if required_elems > max_total_elems:
            # 单个对象超过上限：让它独占池子
            required_elems = max_total_elems

        while current_elems + required_elems > max_total_elems and live_order:
            victim = live_order.popleft()
            vict_elems = live_keys.pop(victim, 0)
            if vict_elems > 0:
                current_elems -= vict_elems
                try:
                    store.remove(victim)
                except Exception as e:
                    print(f"[TP{tp_id}] remove({victim}) failed: {e}")

    def record_new_object(key: str, elems: int):
        nonlocal current_elems

        old = live_keys.get(key, 0)
        if old > 0:
            current_elems -= old

        live_keys[key] = elems
        current_elems += elems
        live_order.append(key)

    # warmup：简单写一点数据，不计入统计
    if warmup > 0 and batches:
        dummy_key = f"warmup_tp{tp_id}"
        dummy_bytes = min(1024 * 1024, buf.nelement() * elem_size)
        dummy_elems = dummy_bytes // elem_size
        dummy_tensor = buf.narrow(0, 0, dummy_elems)
        for _ in range(warmup):
            store.put_tensor(dummy_key, dummy_tensor)

    if barrier is not None:
        try:
            barrier.wait()
        except Exception as e:
            print(f"[TP{tp_id}] barrier wait failed: {e}")

    stats: List[XferStat] = []

    for ops, due in batches:
        # 先看全局截止时间：超过就直接终止后续请求
        if deadline_wall is not None and time.time() >= deadline_wall:
            print(f"[TP{tp_id}] Reached deadline, stop issuing new requests.")
            break

        if due:
            now = time.time()
            # 如果 due 已经落在截止时间之后，可以直接结束
            if deadline_wall is not None and due >= deadline_wall:
                print(f"[TP{tp_id}] Next due ({due}) >= deadline ({deadline_wall}), stop.")
                break
            if now < due:
                # 即使 sleep，也不会跨过 deadline，因为上面已经检查过 due < deadline
                time.sleep(due - now)

        # 此处 st 视为“提交时间”：即真正开始执行第一个读写 op 的时间
        st = perf_counter_ns()
        write_bytes = 0
        read_bytes = 0
        read_reqs = 0
        read_hits = 0

        # 本 request 内单次读/写操作时延（ms）
        per_req_read_lat: List[float] = []
        per_req_write_lat: List[float] = []
        
        for key, logical_bytes in ops:
            logical_bytes = int(logical_bytes)
            if logical_bytes <= 0:
                continue

            # 每个 hash_id 访问都先计为一次「读请求」
            read_reqs += 1

            # 1) 先必定执行一次 get_tensor，并对这次读操作计时
            r = None
            hit = False
            t0 = perf_counter_ns()
            try:
                r = store.get_tensor(key)
                hit = (r is not None)
            except Exception as e:
                print(f"[TP{tp_id}] get_tensor({key}) failed: {e}")
                hit = False
            t1 = perf_counter_ns()
            per_req_read_lat.append((t1 - t0) / 1e6)  # ms

            if hit:
                # get_tensor 成功 → 命中
                read_hits += 1
                read_bytes += logical_bytes

                # 若是远端已有、本地没记录，给对象池补一条记录（估算元素数）
                if key not in live_keys:
                    elems = max(1, logical_bytes // elem_size)
                    record_new_object(key, elems)
                continue

            # 如果对象池里记录了，但这次没读到，修正本地状态
            if key in live_keys and r is None:
                current_elems -= live_keys.get(key, 0)
                live_keys.pop(key, None)
                # 不特意 remove，后续 put 会覆盖

            # 2) miss：写一个新对象，并对 put_tensor 计时
            required_elems = logical_bytes // elem_size
            if required_elems <= 0:
                continue

            # 本地 buf 容量限制：如果太大就截断
            if required_elems > buf.nelement():
                required_elems = buf.nelement()
                logical_bytes = required_elems * elem_size

            ensure_capacity(required_elems)
            tensor_slice = buf.narrow(0, 0, required_elems)

            t2 = perf_counter_ns()
            try:
                store.put_tensor(key, tensor_slice)
                record_new_object(key, required_elems)
                write_bytes += logical_bytes
            except Exception as e:
                print(f"[TP{tp_id}] put_tensor({key}) failed: {e}")
                # 写失败不计入 write_bytes
            t3 = perf_counter_ns()
            per_req_write_lat.append((t3 - t2) / 1e6)  # ms

        ed = perf_counter_ns()
        s = XferStat(
            tp=tp_id,
            start=st,
            end=ed,
            write_bytes=write_bytes,
            read_bytes=read_bytes,
            read_reqs=read_reqs,
            read_hits=read_hits,
            due=due,
            read_latencies=per_req_read_lat,
            write_latencies=per_req_write_lat,
        )
        result_q.put(s)


# ================================
# Worker: Replay
# ================================

def worker_replay(
    tp_id: int,
    client_args,
    buf_size: int,
    events: List[Dict],
    base_ts: int,
    start_wall: float,
    speedup: float,
    dispatch_policy: str,
    num_tp: int,
    result_q: mp.Queue,
    barrier: Optional[mp.Barrier],
    max_total_elems: int,
    hash_block_tokens: int,
    bytes_per_token: int,
    deadline_wall: Optional[float],   # 新增：全局截止时间
):
    torch.cuda.set_device(tp_id)

    store = make_store_client(*client_args)
    # 内容随意，CPU tensor 即可；buf_size 表示“最大可写字节”
    buf = torch.ones(buf_size // 4, dtype=torch.float32, device="cpu")

    disp = Dispatcher(dispatch_policy, num_tp, num_tp)
    batches: List[Tuple[List[Tuple[str, int]], Optional[float]]] = []

    for idx_ev, ev in enumerate(events):
        if disp.choose_local_tp(ev, idx_ev) != tp_id:
            continue

        ops = build_hash_ops_for_event(ev, hash_block_tokens, bytes_per_token)
        if not ops:
            continue

        due = start_wall + ((ev["timestamp"] - base_ts) / speedup) / 1000.0
        batches.append((ops, due))

    run_batches(
        store=store,
        batches=batches,
        buf=buf,
        warmup=0,
        result_q=result_q,
        tp_id=tp_id,
        barrier=barrier,
        max_total_elems=max_total_elems,
        deadline_wall=deadline_wall,
    )


# ================================
# 汇总与 Ctrl-C 提前退出
# ================================

def summarize_stats(stats: List[XferStat]):
    if not stats:
        print("[CLIENT] No stats collected.")
        return

    # request 级别 latency：从提交到所有 block 完成
    durs = [s.dur_ms for s in stats]

    total_write = sum(s.write_bytes for s in stats)
    total_read = sum(s.read_bytes for s in stats)
    total_bytes = total_write + total_read

    total_read_reqs = sum(s.read_reqs for s in stats)
    total_read_hits = sum(s.read_hits for s in stats)
    hit_rate = (total_read_hits / total_read_reqs) if total_read_reqs > 0 else 0.0

    # 汇总所有 op 级别的读写 latency（ms）
    read_lats = [lat for s in stats for lat in s.read_latencies]
    write_lats = [lat for s in stats for lat in s.write_latencies]

    # 基于操作时延估计的“忙时间”（秒）
    read_time_s = sum(read_lats) / 1e3 if read_lats else 0.0
    write_time_s = sum(write_lats) / 1e3 if write_lats else 0.0
    total_time_s = read_time_s + write_time_s

    # 吞吐：总字节 / 总操作时延（并非根据钟面时间）
    gbps_read = total_read / read_time_s / 1e9 if read_time_s > 0 else 0.0
    gbps_write = total_write / write_time_s / 1e9 if write_time_s > 0 else 0.0
    gbps_total = total_bytes / total_time_s / 1e9 if total_time_s > 0 else 0.0

    # 仍然保留一个基于 wall-clock 的参考值
    tspan = (max(s.end for s in stats) - min(s.start for s in stats)) / 1e9
    gbps_total_wall = total_bytes / tspan / 1e9 if tspan > 0 else 0.0

    print(f"\n===== STORE REPLAY (hash-id based, runtime read/write) =====")
    print(f"Samples (requests)                  : {len(stats)}")
    print(f"Req total latency (all blocks)      : "
          f"{np.mean(durs):.3f} / {pct(durs,95):.3f} / {pct(durs,99):.3f} ms (avg/p95/p99)")

    # op 级别 latency 统计
    if read_lats:
        print(f"Read ops                            : {len(read_lats)}")
        print(f"Read op latency avg/p95/p99         : "
              f"{np.mean(read_lats):.3f} / {pct(read_lats,95):.3f} / {pct(read_lats,99):.3f} ms")
    else:
        print("Read ops                            : 0")

    if write_lats:
        print(f"Write ops                           : {len(write_lats)}")
        print(f"Write op latency avg/p95/p99        : "
              f"{np.mean(write_lats):.3f} / {pct(write_lats,95):.3f} / {pct(write_lats,99):.3f} ms")
    else:
        print("Write ops                           : 0")

    print(f"Bytes (write)                       : {total_write/1e9:.3f} GB")
    print(f"Bytes (read)                        : {total_read/1e9:.3f} GB")
    print(f"Bytes (total)                       : {total_bytes/1e9:.3f} GB")

    print(f"Throughput (write, op-time)         : {gbps_write:.2f} GB/s")
    print(f"Throughput (read,  op-time)         : {gbps_read:.2f} GB/s")
    print(f"Throughput (total, op-time)         : {gbps_total:.2f} GB/s")

    print(f"[Ref] Total wall-clock span         : {tspan:.3f} s")
    print(f"[Ref] Throughput (total, wall-clock): {gbps_total_wall:.2f} GB/s")

    print(f"Read requests                       : {total_read_reqs}")
    print(f"Read hits                           : {total_read_hits}")
    print(f"Hit rate                            : {hit_rate*100:.2f}%")

    return {
        "total_write": total_write,
        "total_read": total_read,
        "total_bytes": total_bytes,
        "hit_rate": hit_rate,
    }


def run_client(args):
    client_args = (
        args.local_hostname,
        args.metadata_server,
        args.global_segment_size,
        args.local_buffer_size,
        args.protocol,
        args.device_name,
        args.master_server,
    )

    events = load_trace(args.trace)
    if not events:
        print("[CLIENT] empty trace.")
        return

    base_ts = events[0]["timestamp"]
    start_wall = time.time()
    deadline_wall: Optional[float] = None
    if args.max_duration_sec is not None and args.max_duration_sec > 0:
        deadline_wall = start_wall + args.max_duration_sec
        print(f"[CLIENT] Max duration = {args.max_duration_sec} s, "
              f"deadline_wall = {deadline_wall:.3f}")

    result_q: mp.Queue = mp.Queue()
    ps: List[mp.Process] = []
    stats: List[XferStat] = []

    tp_degree = args.num_gpus
    barrier = mp.Barrier(tp_degree) if tp_degree > 1 else None

    # 启动子进程
    for tp in range(tp_degree):
        p = mp.Process(
            target=worker_replay,
            args=(
                tp,
                client_args,
                args.buf_size,
                events,
                base_ts,
                start_wall,
                args.speedup,
                args.dispatch_policy,
                tp_degree,
                result_q,
                barrier,
                args.max_total_elems,
                args.hash_block_tokens,
                args.bytes_per_token,
                deadline_wall,
            ),
        )
        p.start()
        ps.append(p)

    try:
        # 主循环：持续从队列取结果，直到所有子进程结束或达到最长时间
        while True:
            if deadline_wall is not None and time.time() >= deadline_wall:
                print(f"[CLIENT] Reached max duration {args.max_duration_sec}s, "
                      f"stop collecting and terminate workers.")
                break

            alive = any(p.is_alive() for p in ps)
            try:
                s = result_q.get(timeout=1.0)
                stats.append(s)
            except queue.Empty:
                if not alive:
                    break

        # 超时或正常结束后的进程清理
        if deadline_wall is not None and time.time() >= deadline_wall:
            for p in ps:
                if p.is_alive():
                    p.terminate()
        else:
            for p in ps:
                p.join()

        # 把队列里剩下的结果拿出来
        while True:
            try:
                s = result_q.get_nowait()
                print("get")
                stats.append(s)
            except queue.Empty:
                break

    except KeyboardInterrupt:
        print("\n[CLIENT] Caught Ctrl-C, aggregating partial stats...")
        for p in ps:
            if p.is_alive():
                p.terminate()

        # 把队列里剩下的结果拿出来
        while True:
            try:
                s = result_q.get_nowait()
                stats.append(s)
            except queue.Empty:
                break

    # 汇总输出
    summarize_stats(stats)

    # 输出 CSV（依然是 request 级别的统计）
    if args.csv_out and stats:
        with open(args.csv_out, "w") as f:
            f.write(
                "tp,start_ms,end_ms,due_ms,lat_ms,"
                "write_bytes,read_bytes,read_reqs,read_hits\n"
            )
            for s in stats:
                f.write(
                    f"{s.tp},{s.start/1e6:.3f},{s.end/1e6:.3f},"
                    f"{(s.due or 0)*1e3:.3f},{s.dur_ms:.3f},"
                    f"{s.write_bytes},{s.read_bytes},{s.read_reqs},{s.read_hits}\n"
                )
        print(f"CSV written → {args.csv_out}")


# ================================
# Main
# ================================

def main():
    ap = argparse.ArgumentParser("kvbench-store-client-hash")

    # 只实现 replay 模式
    ap.add_argument("--trace", type=str, required=True)
    ap.add_argument("--num_gpus", type=int, default=1)
    ap.add_argument("--buf_size", type=int, default=512 * 1024 * 1024)
    ap.add_argument("--bytes_per_token", type=int, default=163840)
    ap.add_argument("--hash_block_tokens", type=int, default=512)
    ap.add_argument("--speedup", type=float, default=1.0)
    ap.add_argument(
        "--dispatch_policy",
        choices=["hash", "roundrobin", "random"],
        default="roundrobin",
    )
    ap.add_argument("--csv_out", type=str, default=None)
    ap.add_argument("--verify", action="store_true")

    # Store config
    ap.add_argument("--local_hostname", type=str, default=get_local_ip())
    ap.add_argument("--metadata_server", type=str, required=True)
    ap.add_argument("--master_server", type=str, required=True)
    ap.add_argument("--protocol", type=str, default="rdma")
    ap.add_argument("--device_name", type=str, default="")
    ap.add_argument("--global_segment_size", type=int, default=0)
    ap.add_argument("--local_buffer_size", type=int, default=512 * 1024 * 1024)

    # 原来的元素数量上限
    ap.add_argument(
        "--max_total_elems",
        type=int,
        default=0,
        help="Max total elements stored across all keys in this process (0 = unlimited)",
    )

    # 按 GB 指定总容量
    ap.add_argument(
        "--max_total_gb",
        type=float,
        default=None,
        help="Total memory capacity in GB (automatically converted to max_total_elems)",
    )

    # 新增：最长测试时间（秒）
    ap.add_argument(
        "--max_duration_sec",
        type=int,
        default=300,
        help="Max wall-clock test duration in seconds (0 or negative means no limit)",
    )

    args = ap.parse_args()

    # ========== 自动换算逻辑 ==========
    if args.max_total_gb is not None:
        # float32 = 4 bytes
        BYTES_PER_ELEM = 4
        total_bytes = args.max_total_gb * (1024 ** 3)
        args.max_total_elems = int(total_bytes / BYTES_PER_ELEM)
        print(f"[CONFIG] max_total_gb={args.max_total_gb} GB → "
              f"max_total_elems={args.max_total_elems}")

    mp.set_start_method("spawn", force=True)
    run_client(args)


if __name__ == "__main__":
    main()
