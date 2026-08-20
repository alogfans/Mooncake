#!/usr/bin/env python3
"""Two-node chaos runner for Mooncake classic TE and TENT.

The runner starts tebench on a target and an initiator, injects network
faults on non-control RDMA rails, and records enough state to reproduce a
failed run. It intentionally avoids mlx5_0 / eth0 by default.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_DEVICES = ("mlx5_1", "mlx5_2", "mlx5_3", "mlx5_4")
DEFAULT_NETDEVS = ("eth1", "eth2", "eth3", "eth4")
DEFAULT_EXCLUDED_DEVICES = ("mlx5_0",)
DEFAULT_EXCLUDED_NETDEVS = ("eth0",)
DEFAULT_FAULT_KINDS = (
    "delay",
    "loss",
    "reorder",
    "corrupt",
    "duplicate",
    "rate-limit",
    "link-down",
)


def now_s() -> float:
    return time.time()


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def sh_join(items: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in items)


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def local_hostnames() -> set[str]:
    names = {"localhost", "127.0.0.1", socket.gethostname()}
    try:
        names.add(socket.getfqdn())
    except Exception:
        pass
    return names


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


class Host:
    def __init__(
        self,
        name: str,
        repo: Path,
        ssh_options: list[str],
        dry_run: bool = False,
    ) -> None:
        self.name = name
        self.repo = repo
        self.ssh_options = ssh_options
        self.dry_run = dry_run
        self.is_local = name in local_hostnames()

    def shell_prefix(self) -> list[str]:
        if self.is_local:
            return []
        return ["ssh", *self.ssh_options, self.name]

    def shell_command(self, script: str) -> list[str]:
        if self.is_local:
            return ["bash", "-lc", script]
        return [*self.shell_prefix(), f"bash -lc {shlex.quote(script)}"]

    def run(
        self,
        script: str,
        *,
        check: bool = False,
        timeout: Optional[int] = None,
    ) -> CommandResult:
        if self.dry_run:
            print(f"[dry-run:{self.name}] {script}")
            return CommandResult(0, "", "")
        cmd = self.shell_command(script)
        proc = subprocess.run(
            cmd, text=True, capture_output=True, timeout=timeout
        )
        if check and proc.returncode != 0:
            raise RuntimeError(
                f"{self.name}: command failed with {proc.returncode}: "
                f"{script}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        return CommandResult(proc.returncode, proc.stdout, proc.stderr)

    def popen(self, script: str, log_path: Path) -> subprocess.Popen[str]:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.dry_run:
            print(f"[dry-run:{self.name}] popen {script} > {log_path}")
            return subprocess.Popen(
                ["bash", "-lc", "sleep 0.1"],
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        if not self.is_local:
            pid_path = f"{log_path}.pid"
            wrapped = (
                f"echo $$ > {shlex.quote(pid_path)}; "
                f"trap 'rm -f {shlex.quote(pid_path)}' EXIT; "
                f"{script}"
            )
            remote_script = (
                f"mkdir -p {shlex.quote(str(log_path.parent))} && "
                f"setsid bash -lc {shlex.quote(wrapped)} "
                f"> {shlex.quote(str(log_path))} 2>&1"
            )
            return subprocess.Popen(
                self.shell_command(remote_script),
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        cmd = self.shell_command(script)
        log_file = log_path.open("w", encoding="utf-8")
        return subprocess.Popen(
            cmd,
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def copy_to(self, src: Path, dst: str) -> None:
        if self.dry_run:
            print(f"[dry-run:{self.name}] copy {src} -> {dst}")
            return
        if self.is_local:
            dst_path = Path(dst)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            dst_path.write_bytes(src.read_bytes())
            return
        self.run(f"mkdir -p {shlex.quote(str(Path(dst).parent))}", check=True)
        cmd = ["scp", *self.ssh_options, str(src), f"{self.name}:{dst}"]
        proc = subprocess.run(cmd, text=True, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"scp to {self.name} failed: {proc.stdout}\n{proc.stderr}"
            )


@dataclass
class ManagedProcess:
    name: str
    host: Host
    proc: subprocess.Popen[str]
    log_path: Path
    remote_pid_path: Optional[str] = None
    remote_kill_pattern: Optional[str] = None

    def poll(self) -> Optional[int]:
        return self.proc.poll()

    def terminate(self, grace_s: float = 10.0) -> None:
        if not self.host.is_local:
            self.terminate_remote("TERM")
            if self.proc.poll() is not None:
                time.sleep(min(2.0, grace_s))
                self.terminate_remote("KILL")
                return
        if self.proc.poll() is not None:
            return
        try:
            if self.host.is_local:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            else:
                self.proc.terminate()
            self.proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            if self.host.is_local:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            else:
                self.terminate_remote("KILL")
                self.proc.kill()
            self.proc.wait(timeout=5)
        except ProcessLookupError:
            pass

    def terminate_remote(self, sig: str) -> None:
        commands: list[str] = []
        if self.remote_pid_path:
            pid_file = shlex.quote(self.remote_pid_path)
            commands.append(
                "if test -s "
                f"{pid_file}; then pid=$(cat {pid_file}); "
                "pgid=$(ps -o pgid= -p $pid 2>/dev/null | tr -d ' '); "
                f"test -n \"$pgid\" && kill -{sig} -- -$pgid 2>/dev/null || "
                f"kill -{sig} $pid 2>/dev/null || true; fi"
            )
        if self.remote_kill_pattern:
            commands.append(
                f"pkill -{sig} -f {shlex.quote(self.remote_kill_pattern)} "
                "2>/dev/null || true"
            )
        if commands:
            self.host.run("; ".join(commands))


@dataclass
class Workload:
    backend: str
    duration_s: int
    threads: int
    block_size: int
    batch_size: int
    buffer_size: int
    op_type: str = "mix"
    check_consistency: bool = True
    seg_type: str = "DRAM"
    local_gpu_id: int = 0
    target_gpu_id: int = 0
    request_interval_us: int = 0
    extra_args: list[str] = field(default_factory=list)

    def backend_kind(self) -> str:
        if self.backend.startswith("tent"):
            return "tent"
        if self.backend == "classic":
            return "classic"
        raise ValueError(f"unsupported backend: {self.backend}")

    def tent_transport_mode(self) -> str:
        if self.backend == "tent-rdma-only":
            return "rdma-only"
        if self.backend in ("tent", "tent-fallback"):
            return "rdma-tcp-fallback"
        return "classic"


@dataclass
class RunConfig:
    initiator: Host
    target: Host
    metadata_host: Host
    run_dir: Path
    remote_run_dir: str
    tebench: str
    metadata_port: int
    metadata_mode: str
    devices: list[str]
    excluded_devices: list[str]
    netdevs: list[str]
    excluded_netdevs: list[str]
    sudo: str
    startup_wait_s: float
    warmup_s: float
    action_gap_s: float
    fault_kinds: list[str]
    dry_run: bool

    @property
    def metadata_url(self) -> str:
        return f"http://{self.metadata_host.name}:{self.metadata_port}/metadata"


class EventLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **fields: object) -> None:
        record = {"ts": now_s(), "event": event, **fields}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True))


@dataclass
class ChaosAction:
    name: str
    host_name: str
    netdev: str
    command: str
    revert_command: str
    hold_s: float


class ChaosRunner:
    def __init__(self, cfg: RunConfig, events: EventLog) -> None:
        self.cfg = cfg
        self.events = events
        self.hosts = {
            cfg.initiator.name: cfg.initiator,
            cfg.target.name: cfg.target,
            cfg.metadata_host.name: cfg.metadata_host,
        }

    def host(self, name: str) -> Host:
        return self.hosts[name]

    def sudo_cmd(self, command: str) -> str:
        if not self.cfg.sudo:
            return command
        return f"{self.cfg.sudo} {command}"

    def cleanup_network(self) -> None:
        for host in (self.cfg.initiator, self.cfg.target):
            for netdev in self.cfg.netdevs:
                if netdev in self.cfg.excluded_netdevs:
                    continue
                script = (
                    f"{self.sudo_cmd(f'tc qdisc del dev {shlex.quote(netdev)} root')} "
                    "2>/dev/null || true; "
                    f"{self.sudo_cmd(f'ip link set dev {shlex.quote(netdev)} up')} "
                    "2>/dev/null || true"
                )
                result = host.run(script)
                self.events.write(
                    "cleanup-netdev",
                    host=host.name,
                    netdev=netdev,
                    rc=result.returncode,
                )

    def apply_action(self, action: ChaosAction) -> None:
        host = self.host(action.host_name)
        self.events.write(
            "action-apply",
            name=action.name,
            host=host.name,
            netdev=action.netdev,
            command=action.command,
            hold_s=action.hold_s,
        )
        result = host.run(action.command)
        self.events.write(
            "action-applied",
            name=action.name,
            host=host.name,
            netdev=action.netdev,
            rc=result.returncode,
            stdout=result.stdout[-4000:],
            stderr=result.stderr[-4000:],
        )
        if result.returncode != 0:
            raise RuntimeError(f"failed to apply action {action.name}")

    def revert_action(self, action: ChaosAction) -> None:
        host = self.host(action.host_name)
        self.events.write(
            "action-revert",
            name=action.name,
            host=host.name,
            netdev=action.netdev,
            command=action.revert_command,
        )
        result = host.run(action.revert_command)
        self.events.write(
            "action-reverted",
            name=action.name,
            host=host.name,
            netdev=action.netdev,
            rc=result.returncode,
            stdout=result.stdout[-4000:],
            stderr=result.stderr[-4000:],
        )

    def make_netem(
        self,
        *,
        host: Host,
        netdev: str,
        name: str,
        netem_args: str,
        hold_s: float,
    ) -> ChaosAction:
        quoted = shlex.quote(netdev)
        return ChaosAction(
            name=name,
            host_name=host.name,
            netdev=netdev,
            command=self.sudo_cmd(
                f"tc qdisc replace dev {quoted} root netem {netem_args}"
            ),
            revert_command=(
                self.sudo_cmd(f"tc qdisc del dev {quoted} root")
                + " 2>/dev/null || true"
            ),
            hold_s=hold_s,
        )

    def make_link_down(self, host: Host, netdev: str, hold_s: float) -> ChaosAction:
        quoted = shlex.quote(netdev)
        return ChaosAction(
            name="link-down",
            host_name=host.name,
            netdev=netdev,
            command=self.sudo_cmd(f"ip link set dev {quoted} down"),
            revert_command=self.sudo_cmd(f"ip link set dev {quoted} up"),
            hold_s=hold_s,
        )

    def make_rate_limit(
        self, host: Host, netdev: str, rate: str, hold_s: float
    ) -> ChaosAction:
        quoted = shlex.quote(netdev)
        quoted_rate = shlex.quote(rate)
        return ChaosAction(
            name=f"rate-limit-{rate}",
            host_name=host.name,
            netdev=netdev,
            command=self.sudo_cmd(
                "tc qdisc replace dev "
                f"{quoted} root tbf rate {quoted_rate} burst 32mb latency 400ms"
            ),
            revert_command=(
                self.sudo_cmd(f"tc qdisc del dev {quoted} root")
                + " 2>/dev/null || true"
            ),
            hold_s=hold_s,
        )

    def deterministic_actions(self) -> list[ChaosAction]:
        actions: list[ChaosAction] = []
        host_order = [self.cfg.initiator, self.cfg.target]
        enabled = set(self.cfg.fault_kinds)
        for host in host_order:
            for netdev in self.cfg.netdevs:
                if "delay" in enabled:
                    actions.append(
                        self.make_netem(
                            host=host,
                            netdev=netdev,
                            name="delay-100ms-jitter-20ms",
                            netem_args="delay 100ms 20ms distribution normal",
                            hold_s=30,
                        )
                    )
                if "loss" in enabled:
                    actions.append(
                        self.make_netem(
                            host=host,
                            netdev=netdev,
                            name="loss-1pct",
                            netem_args="loss 1%",
                            hold_s=30,
                        )
                    )
                if "reorder" in enabled:
                    actions.append(
                        self.make_netem(
                            host=host,
                            netdev=netdev,
                            name="reorder-25pct",
                            netem_args="delay 20ms reorder 25% 50%",
                            hold_s=30,
                        )
                    )
                if "rate-limit" in enabled:
                    actions.append(
                        self.make_rate_limit(
                            host=host,
                            netdev=netdev,
                            rate="5gbit",
                            hold_s=30,
                        )
                    )
                if "link-down" in enabled:
                    actions.append(self.make_link_down(host, netdev, hold_s=10))
        return actions

    def random_action(self, rnd: random.Random) -> ChaosAction:
        host = rnd.choice([self.cfg.initiator, self.cfg.target])
        netdev = rnd.choice(self.cfg.netdevs)
        kind = rnd.choice(self.cfg.fault_kinds)
        hold_s = rnd.uniform(5, 45)
        if kind == "delay":
            delay = rnd.choice([20, 50, 100, 200])
            jitter = max(1, delay // 5)
            return self.make_netem(
                host=host,
                netdev=netdev,
                name=f"delay-{delay}ms",
                netem_args=f"delay {delay}ms {jitter}ms distribution normal",
                hold_s=hold_s,
            )
        if kind == "loss":
            loss = rnd.choice([0.1, 0.5, 1.0, 2.0, 5.0])
            return self.make_netem(
                host=host,
                netdev=netdev,
                name=f"loss-{loss:g}pct",
                netem_args=f"loss {loss:g}%",
                hold_s=hold_s,
            )
        if kind == "corrupt":
            corrupt = rnd.choice([0.01, 0.05, 0.1])
            return self.make_netem(
                host=host,
                netdev=netdev,
                name=f"corrupt-{corrupt:g}pct",
                netem_args=f"corrupt {corrupt:g}%",
                hold_s=hold_s,
            )
        if kind == "duplicate":
            duplicate = rnd.choice([0.1, 0.5, 1.0])
            return self.make_netem(
                host=host,
                netdev=netdev,
                name=f"duplicate-{duplicate:g}pct",
                netem_args=f"duplicate {duplicate:g}%",
                hold_s=hold_s,
            )
        if kind == "reorder":
            return self.make_netem(
                host=host,
                netdev=netdev,
                name="reorder-random",
                netem_args="delay 20ms reorder 25% 50%",
                hold_s=hold_s,
            )
        if kind == "rate-limit":
            rate = rnd.choice(["1gbit", "5gbit", "10gbit"])
            return self.make_rate_limit(host, netdev, rate, hold_s=hold_s)
        return self.make_link_down(host, netdev, hold_s=min(hold_s, 15))

    def action_stream(self, suite: str, seed: int) -> Iterable[ChaosAction]:
        if suite == "smoke":
            return []
        if suite == "deterministic":
            return self.deterministic_actions()
        rnd = random.Random(seed)

        def gen() -> Iterable[ChaosAction]:
            while True:
                yield self.random_action(rnd)

        return gen()

    def run_actions_until_done(
        self,
        suite: str,
        seed: int,
        initiator: ManagedProcess,
        target: ManagedProcess,
        deadline: float,
    ) -> None:
        if self.cfg.warmup_s > 0:
            self.events.write("warmup-start", seconds=self.cfg.warmup_s)
            self.sleep_or_until_done(self.cfg.warmup_s, initiator, target)
        for action in self.action_stream(suite, seed):
            if now_s() >= deadline:
                break
            if initiator.poll() is not None:
                break
            target_rc = target.poll()
            if target_rc is not None:
                raise RuntimeError(f"target exited early with {target_rc}")
            self.apply_action(action)
            try:
                self.sleep_or_until_done(action.hold_s, initiator, target)
            finally:
                self.revert_action(action)
            if self.cfg.action_gap_s > 0:
                self.sleep_or_until_done(self.cfg.action_gap_s, initiator, target)

    def sleep_or_until_done(
        self,
        seconds: float,
        initiator: ManagedProcess,
        target: ManagedProcess,
    ) -> None:
        end = now_s() + seconds
        while now_s() < end:
            if initiator.poll() is not None:
                return
            target_rc = target.poll()
            if target_rc is not None:
                raise RuntimeError(f"target exited early with {target_rc}")
            time.sleep(min(1.0, end - now_s()))


def make_tent_config(path: Path, workload: Workload, devices: list[str]) -> None:
    rdma_transports = ["rdma"]
    tcp_enabled = False
    if workload.tent_transport_mode() == "rdma-tcp-fallback":
        rdma_transports = ["rdma", "tcp"]
        tcp_enabled = True
    cfg = {
        "topology": {
            "rdma_whitelist": devices,
            "rdma_blacklist": list(DEFAULT_EXCLUDED_DEVICES),
        },
        "max_failover_attempts": 3,
        "enable_auto_failover_on_poll": True,
        "transports": {
            "rdma": {
                "enable": True,
                "rail_error_threshold": 3,
                "rail_error_window_secs": 10,
                "rail_cooldown_secs": 5,
            },
            "tcp": {"enable": tcp_enabled},
            "shm": {"enable": False},
            "gds": {"enable": False},
            "io_uring": {"enable": False},
            "nvlink": {"enable": False},
            "mnnvl": {"enable": False},
            "mpcomm": {"enable": False},
        },
        "policy": [
            {
                "name": "chaos-memory",
                "segment_type": "memory",
                "transports": rdma_transports,
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")


def tebench_env(cfg: RunConfig, tent_conf: Optional[str]) -> str:
    env = {
        "MC_TE_FILTERS": ",".join(cfg.devices),
        "MC_TE_FILTERS_EXCLUDE": ",".join(cfg.excluded_devices),
    }
    if tent_conf:
        env["MC_TENT_CONF"] = tent_conf
    return " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())


def tebench_args(
    cfg: RunConfig,
    workload: Workload,
    *,
    role: str,
    seg_name: str,
    target_seg_name: Optional[str],
) -> list[str]:
    args = [
        cfg.tebench,
        f"--backend={workload.backend_kind()}",
        f"--metadata_type={cfg.metadata_mode}",
        f"--seg_type={workload.seg_type}",
        f"--total_buffer_size={workload.buffer_size}",
        f"--local_gpu_id={workload.local_gpu_id}",
        f"--target_gpu_id={workload.target_gpu_id}",
        "--logtostderr=true",
    ]
    if cfg.metadata_mode == "http":
        args.append(f"--metadata_url_list={cfg.metadata_url}")
        args.append(f"--seg_name={seg_name}")
    else:
        args.append(f"--rpc_server_port={cfg.metadata_port}")
        args.append(f"--seg_name={seg_name}")
    if workload.backend == "tent-rdma-only":
        args.append("--xport_type=rdma")
    if role == "initiator":
        if not target_seg_name:
            raise ValueError("initiator requires target segment name")
        args.extend(
            [
                f"--target_seg_name={target_seg_name}",
                f"--op_type={workload.op_type}",
                f"--duration={workload.duration_s}",
                f"--start_num_threads={workload.threads}",
                f"--max_num_threads={workload.threads}",
                f"--start_block_size={workload.block_size}",
                f"--max_block_size={workload.block_size}",
                f"--start_batch_size={workload.batch_size}",
                f"--max_batch_size={workload.batch_size}",
                f"--request_interval_us={workload.request_interval_us}",
            ]
        )
        if workload.check_consistency:
            args.append("--check_consistency=true")
    args.extend(workload.extra_args)
    return args


def start_metadata(cfg: RunConfig, events: EventLog) -> Optional[ManagedProcess]:
    if cfg.metadata_mode != "http":
        return None
    log = cfg.run_dir / "metadata.log"
    metadata_script = "mooncake-wheel/mooncake/http_metadata_server.py"
    cmd = (
        f"cd {shlex.quote(str(cfg.metadata_host.repo))} && "
        f"if test -f {shlex.quote(metadata_script)}; then "
        f"python3 {shlex.quote(metadata_script)} "
        f"--host 0.0.0.0 --port {cfg.metadata_port}; "
        f"else mooncake_http_metadata_server --host 0.0.0.0 "
        f"--port {cfg.metadata_port}; fi"
    )
    proc = cfg.metadata_host.popen(cmd, log)
    handle = ManagedProcess(
        "metadata",
        cfg.metadata_host,
        proc,
        log,
        None if cfg.metadata_host.is_local else f"{log}.pid",
    )
    events.write("metadata-started", host=cfg.metadata_host.name, log=str(log))
    if cfg.dry_run:
        return handle
    time.sleep(1.0)
    if handle.poll() is not None:
        raise RuntimeError(f"metadata server exited early; see {log}")
    return handle


def discover_p2p_target_seg_name(log_path: Path, timeout_s: float) -> str:
    deadline = now_s() + timeout_s
    patterns = [
        re.compile(r"--target_seg_name=([^\s]+)"),
        re.compile(r"listening on ([^\s]+:\d+)"),
        re.compile(r"Transfer Engine ([^\s]+:\d+) started successfully"),
    ]
    while now_s() < deadline:
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    return match.group(1)
        time.sleep(0.2)
    raise RuntimeError(f"could not discover P2P target segment from {log_path}")


def start_tebench_pair(
    cfg: RunConfig,
    workload: Workload,
    run_id: str,
    events: EventLog,
) -> tuple[ManagedProcess, ManagedProcess]:
    local_conf: Optional[Path] = None
    remote_conf: Optional[str] = None
    if workload.backend_kind() == "tent":
        local_conf = cfg.run_dir / f"{run_id}-tent.json"
        make_tent_config(local_conf, workload, cfg.devices)
        remote_conf = str(Path(cfg.remote_run_dir) / local_conf.name)
        cfg.target.copy_to(local_conf, remote_conf)

    seg_name = f"{run_id}-target"
    if cfg.metadata_mode == "http":
        target_seg_name = seg_name
    else:
        target_seg_name = f"{cfg.target.name}:{cfg.metadata_port}"

    target_env = tebench_env(cfg, remote_conf)
    target_args = tebench_args(
        cfg,
        workload,
        role="target",
        seg_name=seg_name,
        target_seg_name=None,
    )
    target_cmd = (
        f"cd {shlex.quote(str(cfg.target.repo))} && "
        f"{target_env} {sh_join(target_args)}"
    )
    target_log = cfg.run_dir / f"{run_id}-target.log"
    target_proc = ManagedProcess(
        "target",
        cfg.target,
        cfg.target.popen(target_cmd, target_log),
        target_log,
        None if cfg.target.is_local else f"{target_log}.pid",
        None if cfg.target.is_local else run_id,
    )
    events.write(
        "target-started",
        host=cfg.target.name,
        seg_name=seg_name,
        target_seg_name=target_seg_name,
        log=str(target_log),
        command=target_cmd,
    )
    time.sleep(cfg.startup_wait_s)
    if not cfg.dry_run and target_proc.poll() is not None:
        raise RuntimeError(f"target exited during startup; see {target_log}")
    if cfg.metadata_mode == "p2p" and not cfg.dry_run:
        target_seg_name = discover_p2p_target_seg_name(target_log, 3.0)
        events.write(
            "target-segment-discovered",
            host=cfg.target.name,
            target_seg_name=target_seg_name,
        )

    initiator_conf = str(local_conf) if local_conf else None
    initiator_env = tebench_env(cfg, initiator_conf)
    initiator_args = tebench_args(
        cfg,
        workload,
        role="initiator",
        seg_name=f"{run_id}-initiator",
        target_seg_name=target_seg_name,
    )
    initiator_cmd = (
        f"cd {shlex.quote(str(cfg.initiator.repo))} && "
        f"{initiator_env} {sh_join(initiator_args)}"
    )
    initiator_log = cfg.run_dir / f"{run_id}-initiator.log"
    initiator_proc = ManagedProcess(
        "initiator",
        cfg.initiator,
        cfg.initiator.popen(initiator_cmd, initiator_log),
        initiator_log,
        None if cfg.initiator.is_local else f"{initiator_log}.pid",
        None if cfg.initiator.is_local else run_id,
    )
    events.write(
        "initiator-started",
        host=cfg.initiator.name,
        log=str(initiator_log),
        command=initiator_cmd,
    )
    return target_proc, initiator_proc


def run_one_workload(
    cfg: RunConfig,
    workload: Workload,
    *,
    suite: str,
    seed: int,
    run_id: str,
    events: EventLog,
) -> int:
    metadata = None
    target = None
    initiator = None
    runner = ChaosRunner(cfg, events)
    rc = 1
    try:
        runner.cleanup_network()
        metadata = start_metadata(cfg, events)
        target, initiator = start_tebench_pair(cfg, workload, run_id, events)
        deadline = now_s() + workload.duration_s + cfg.startup_wait_s + 30
        runner.run_actions_until_done(suite, seed, initiator, target, deadline)
        while initiator.poll() is None and now_s() < deadline:
            time.sleep(1)
        if initiator.poll() is None:
            events.write("initiator-timeout", run_id=run_id)
            initiator.terminate()
            rc = 124
        else:
            rc = initiator.poll() or 0
        events.write("initiator-exited", run_id=run_id, rc=rc)
        if not cfg.dry_run and target.poll() is not None:
            events.write("target-exited", run_id=run_id, rc=target.poll())
            if rc == 0:
                rc = target.poll() or 1
    except Exception as exc:
        events.write("run-error", run_id=run_id, error=str(exc))
        rc = 1
    finally:
        runner.cleanup_network()
        if initiator is not None:
            initiator.terminate(grace_s=5)
        if target is not None:
            target.terminate(grace_s=10)
        if metadata is not None:
            metadata.terminate(grace_s=5)
    return rc


def run_doctor(cfg: RunConfig) -> int:
    checks = []
    for host in (cfg.initiator, cfg.target):
        checks.append((host, "hostname", "hostname"))
        checks.append(
            (
                host,
                "tebench",
                f"cd {shlex.quote(str(host.repo))} && "
                f"test -x {shlex.quote(cfg.tebench)}",
            )
        )
        checks.append((host, "rdma", "command -v rdma >/dev/null && rdma link show"))
        checks.append((host, "ip", "ip -brief link show"))
        if cfg.sudo:
            probe_netdev = shlex.quote(cfg.netdevs[0])
            checks.append(
                (
                    host,
                    "sudo-tc",
                    f"{cfg.sudo} tc qdisc show dev {probe_netdev} >/dev/null",
                )
            )
            checks.append(
                (
                    host,
                    "sudo-ip",
                    f"{cfg.sudo} ip link show dev {probe_netdev} >/dev/null",
                )
            )
    if cfg.metadata_mode == "http":
        checks.append(
            (
                cfg.metadata_host,
                "http-metadata",
                "command -v mooncake_http_metadata_server >/dev/null || "
                "python3 -c 'import aiohttp' >/dev/null",
            )
        )
    ok = True
    for host, name, script in checks:
        result = host.run(script)
        status = "OK" if result.returncode == 0 else "FAIL"
        print(f"[{status}] {host.name}:{name}")
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)
        ok = ok and result.returncode == 0
    banned = set(cfg.excluded_netdevs)
    active = set(cfg.netdevs)
    if banned & active:
        print(f"[FAIL] netdev selection includes excluded devices: {banned & active}")
        ok = False
    print(f"selected RDMA devices: {','.join(cfg.devices)}")
    print(f"selected netdevs: {','.join(cfg.netdevs)}")
    print(f"excluded RDMA devices: {','.join(cfg.excluded_devices)}")
    print(f"excluded netdevs: {','.join(cfg.excluded_netdevs)}")
    return 0 if ok else 1


def run_cleanup(cfg: RunConfig) -> int:
    events = EventLog(cfg.run_dir / "cleanup-events.jsonl")
    ChaosRunner(cfg, events).cleanup_network()
    return 0


def workloads_from_backend(args: argparse.Namespace) -> list[Workload]:
    backends = (
        ["tent-fallback", "tent-rdma-only", "classic"]
        if args.backend == "all"
        else [args.backend]
    )
    return [
        Workload(
            backend=backend,
            duration_s=args.duration,
            threads=args.threads,
            block_size=args.block_size,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            op_type=args.op_type,
            check_consistency=not args.no_check_consistency,
            seg_type=args.seg_type,
            local_gpu_id=args.local_gpu_id,
            target_gpu_id=args.target_gpu_id,
            request_interval_us=args.request_interval_us,
            extra_args=args.extra_tebench_arg,
        )
        for backend in backends
    ]


def build_config(args: argparse.Namespace) -> RunConfig:
    ssh_options = ["-o", "BatchMode=yes", "-o", f"ConnectTimeout={args.ssh_timeout}"]
    repo = Path(args.repo).resolve()
    run_dir = Path(args.run_dir or (repo / "build" / "chaos-runs" / timestamp()))
    remote_run_dir = args.remote_run_dir or str(run_dir)
    dry_run = bool(getattr(args, "dry_run", False))
    initiator = Host(args.initiator, repo, ssh_options, dry_run=dry_run)
    target = Host(args.target, Path(args.remote_repo or args.repo), ssh_options, dry_run=dry_run)
    metadata_name = args.metadata_host or args.target
    metadata_host = target if metadata_name == target.name else Host(
        metadata_name, Path(args.remote_repo or args.repo), ssh_options, dry_run=dry_run
    )
    devices = parse_csv(args.devices)
    netdevs = parse_csv(args.netdevs)
    excluded_devices = parse_csv(args.exclude_devices)
    excluded_netdevs = parse_csv(args.exclude_netdevs)
    fault_kinds = parse_csv(args.fault_kinds)
    if "mlx5_0" in devices or "eth0" in netdevs:
        raise ValueError("refusing to include mlx5_0/eth0 in the chaos target set")
    allowed_fault_kinds = set(DEFAULT_FAULT_KINDS)
    unknown_fault_kinds = sorted(set(fault_kinds) - allowed_fault_kinds)
    if unknown_fault_kinds:
        raise ValueError(f"unsupported fault kinds: {','.join(unknown_fault_kinds)}")
    if not fault_kinds:
        raise ValueError("at least one fault kind must be selected")
    return RunConfig(
        initiator=initiator,
        target=target,
        metadata_host=metadata_host,
        run_dir=run_dir,
        remote_run_dir=remote_run_dir,
        tebench=args.tebench,
        metadata_port=args.metadata_port,
        metadata_mode=args.metadata_mode,
        devices=devices,
        excluded_devices=excluded_devices,
        netdevs=netdevs,
        excluded_netdevs=excluded_netdevs,
        sudo=args.sudo,
        startup_wait_s=args.startup_wait,
        warmup_s=args.warmup,
        action_gap_s=args.action_gap,
        fault_kinds=fault_kinds,
        dry_run=dry_run,
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--initiator", default=socket.gethostname())
    parser.add_argument("--target", default="qjh001")
    parser.add_argument("--metadata-host", default=None)
    parser.add_argument("--repo", default="/mnt/qjh000/rf/Mooncake")
    parser.add_argument("--remote-repo", default=None)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--remote-run-dir", default="")
    parser.add_argument(
        "--tebench",
        default="build/mooncake-transfer-engine/benchmark/tebench",
    )
    parser.add_argument("--metadata-mode", choices=["http", "p2p"], default="p2p")
    parser.add_argument("--metadata-port", type=int, default=18080)
    parser.add_argument("--devices", default=",".join(DEFAULT_DEVICES))
    parser.add_argument("--exclude-devices", default=",".join(DEFAULT_EXCLUDED_DEVICES))
    parser.add_argument("--netdevs", default=",".join(DEFAULT_NETDEVS))
    parser.add_argument("--exclude-netdevs", default=",".join(DEFAULT_EXCLUDED_NETDEVS))
    parser.add_argument("--sudo", default="sudo -n")
    parser.add_argument("--ssh-timeout", type=int, default=8)
    parser.add_argument("--startup-wait", type=float, default=5.0)
    parser.add_argument("--warmup", type=float, default=10.0)
    parser.add_argument("--action-gap", type=float, default=2.0)
    parser.add_argument(
        "--fault-kinds",
        default=",".join(DEFAULT_FAULT_KINDS),
        help=(
            "Comma-separated fault kinds: delay,loss,reorder,corrupt,"
            "duplicate,rate-limit,link-down"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")


def add_workload_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        choices=["tent-fallback", "tent-rdma-only", "classic", "all"],
        default="tent-fallback",
    )
    parser.add_argument(
        "--suite",
        choices=["smoke", "deterministic", "random", "soak"],
        default="deterministic",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration", type=int, default=600)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=65536)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--buffer-size", type=int, default=1 << 30)
    parser.add_argument("--op-type", choices=["read", "write", "mix"], default="mix")
    parser.add_argument("--no-check-consistency", action="store_true")
    parser.add_argument("--seg-type", choices=["DRAM", "VRAM"], default="DRAM")
    parser.add_argument("--local-gpu-id", type=int, default=0)
    parser.add_argument("--target-gpu-id", type=int, default=0)
    parser.add_argument("--request-interval-us", type=int, default=0)
    parser.add_argument(
        "--extra-tebench-arg",
        action="append",
        default=[],
        help="Additional raw tebench flag; may be repeated.",
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Check hosts, devices and sudo access")
    add_common_args(doctor)

    cleanup = sub.add_parser("cleanup", help="Remove qdisc faults and bring rails up")
    add_common_args(cleanup)

    run = sub.add_parser("run", help="Run tebench with chaos injection")
    add_common_args(run)
    add_workload_args(run)

    args = parser.parse_args(argv)
    cfg = build_config(args)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "doctor":
        return run_doctor(cfg)
    if args.command == "cleanup":
        return run_cleanup(cfg)

    events = EventLog(cfg.run_dir / "events.jsonl")
    seed = args.seed or random.randrange(1, 2**31)
    events.write(
        "run-start",
        suite=args.suite,
        seed=seed,
        run_dir=str(cfg.run_dir),
        initiator=cfg.initiator.name,
        target=cfg.target.name,
        metadata_mode=cfg.metadata_mode,
    )
    results = []
    for index, workload in enumerate(workloads_from_backend(args)):
        run_id = f"{timestamp()}-{index}-{workload.backend}"
        events.write("workload-start", run_id=run_id, workload=workload.__dict__)
        rc = run_one_workload(
            cfg,
            workload,
            suite=("random" if args.suite == "soak" else args.suite),
            seed=seed + index,
            run_id=run_id,
            events=events,
        )
        events.write("workload-finish", run_id=run_id, rc=rc)
        results.append({"run_id": run_id, "backend": workload.backend, "rc": rc})
        if rc != 0:
            break
    summary = {"seed": seed, "results": results, "run_dir": str(cfg.run_dir)}
    (cfg.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    events.write("run-finish", summary=summary)
    return 0 if all(item["rc"] == 0 for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
