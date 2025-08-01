import json
import time
import os
import argparse
import paramiko
from tqdm import tqdm
from itertools import product

def parse_args():
    parser = argparse.ArgumentParser(description="RDMA connectivity test between two hosts.")
    parser.add_argument("--src-host", required=True, help="Source hostname")
    parser.add_argument("--dst-host", required=True, help="Destination hostname")
    parser.add_argument("--src-port", type=int, default=22, help="SSH port for source host")
    parser.add_argument("--dst-port", type=int, default=22, help="SSH port for destination host")
    parser.add_argument("--sudo", action="store_true", help="Use sudo for remote commands")
    parser.add_argument("--file", default="cluster_topology.json", help="Path of the generated cluster topology file")
    return parser.parse_args()


def ssh_exec(host, port, command):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port=port)
    stdin, stdout, stderr = client.exec_command(command)
    stdout.channel.recv_exit_status()
    out = stdout.read().decode()
    err = stderr.read().decode()
    client.close()
    return out + err


def list_rdma_devices(host, port, use_sudo):
    cmd = ("sudo " if use_sudo else "") + "ibv_devices | grep -v device | awk '{print $1}'"
    out = ssh_exec(host, port, cmd)
    devs = out.strip().splitlines()

    results = []
    for dev in devs:
        sysfs_cmd = f"cat /sys/class/infiniband/{dev}/device/numa_node"
        if use_sudo:
            sysfs_cmd = "sudo " + sysfs_cmd
        numa_str = ssh_exec(host, port, sysfs_cmd).strip()
        try:
            numa = int(numa_str)
        except:
            numa = -1
        results.append({"name": dev, "numa_node": numa})
    return results


def parse_bandwidth(output):
    for line in output.strip().splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0].isdigit():
            try:
                return float(parts[3])
            except ValueError:
                return None
    return None


def parse_latency(output):
    for line in output.strip().splitlines():
        parts = line.split()
        if len(parts) >= 6 and parts[0].isdigit():
            try:
                return float(parts[5])
            except ValueError:
                return None
    return None


def run_rdmatest(src, dst, dev1, dev2, use_sudo):
    prefix = "sudo " if use_sudo else ""

    def numactl_prefix(numa):
        if numa < 0:
            return ""
        return f"numactl --cpunodebind={numa} --membind={numa} "

    cmd_prefix_src = numactl_prefix(dev1["numa_node"]) + prefix
    cmd_prefix_dst = numactl_prefix(dev2["numa_node"]) + prefix

    # Start ib_write_bw server
    ssh_exec(dst["host"], dst["port"],
             f"{prefix}pkill ib_write_bw; {cmd_prefix_dst}nohup ib_write_bw --ib-dev={dev2['name']} > /tmp/bw_server.log 2>&1 &")
    time.sleep(0.5)

    bw_output = ssh_exec(src["host"], src["port"],
                         f"{cmd_prefix_src}ib_write_bw {dst['host']} --ib-dev={dev1['name']}")
    bw_val = parse_bandwidth(bw_output)
    if bw_val is None:
        return None

    # Start ib_read_lat server
    ssh_exec(dst["host"], dst["port"],
             f"{prefix}pkill ib_read_lat; {cmd_prefix_dst}nohup ib_read_lat --ib-dev={dev2['name']} > /tmp/lat_server.log 2>&1 &")
    time.sleep(0.5)

    lat_output = ssh_exec(src["host"], src["port"],
                          f"{cmd_prefix_src}ib_read_lat {dst['host']} --ib-dev={dev1['name']}")
    lat_val = parse_latency(lat_output)

    return {
        "src_dev": dev1["name"],
        "dst_dev": dev2["name"],
        "src_numa": dev1["numa_node"],
        "dst_numa": dev2["numa_node"],
        "bandwidth": bw_val,
        "latency": lat_val
    }


def load_results(filepath):
    if os.path.exists(filepath):
        with open(filepath) as f:
            return json.load(f)
    return []


def save_results(filepath, results):
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)


def main():
    args = parse_args()

    src = {"host": args.src_host, "port": args.src_port}
    dst = {"host": args.dst_host, "port": args.dst_port}
    use_sudo = args.sudo

    result_file = args.file
    all_results = load_results(result_file)

    existing_idx = next((i for i, e in enumerate(all_results)
                         if e["src_host"] == src["host"] and e["dst_host"] == dst["host"]),
                        None)

    if existing_idx is not None:
        confirm = input(f"\nEntry already exists for {src['host']} → {dst['host']}. "
                        f"Do you want to overwrite and re-test? (y = overwrite and retest / n = skip): ").strip().lower()
        if confirm != 'y':
            print("Skipping test and keeping existing result.")
            return
        else:
            print("Overwriting existing entry after re-testing.")

    print(f"Discovering RDMA devices on {src['host']} and {dst['host']}...")
    devices_src = list_rdma_devices(src["host"], src["port"], use_sudo)
    devices_dst = list_rdma_devices(dst["host"], dst["port"], use_sudo)

    total = len(devices_src) * len(devices_dst)
    endpoints = []

    for dev1, dev2 in tqdm(product(devices_src, devices_dst), total=total, desc="Testing", unit="test"):
        result = run_rdmatest(src, dst, dev1, dev2, use_sudo)
        if result:
            endpoints.append(result)

    new_entry = {
        "src_host": src["host"],
        "dst_host": dst["host"],
        "endpoints": endpoints
    }

    if existing_idx is not None:
        all_results[existing_idx] = new_entry
    else:
        all_results.append(new_entry)

    save_results(result_file, all_results)
    print(f"\nResults written to {result_file}")


if __name__ == "__main__":
    main()
