import json
import argparse
import numpy as np
from collections import defaultdict, Counter
from scipy.optimize import linear_sum_assignment

def build_partition_map(endpoints):
    """Group endpoints by (src_numa, dst_numa)."""
    partition_map = defaultdict(list)
    for ep in endpoints:
        if not np.isfinite(ep.get("latency", float('inf'))):
            continue  # skip unreachable
        key = f"{ep['src_numa']}-{ep['dst_numa']}"
        partition_map[key].append(ep)
    return partition_map

def solve_partition_group(pairs, allow_partial=False):
    src_devs = sorted(set(ep['src_dev'] for ep in pairs))
    dst_devs = sorted(set(ep['dst_dev'] for ep in pairs))

    N_src = len(src_devs)
    N_dst = len(dst_devs)
    N = max(N_src, N_dst)

    idx_src = {dev: i for i, dev in enumerate(src_devs)}
    idx_dst = {dev: i for i, dev in enumerate(dst_devs)}

    cost = np.full((N, N), 1e9)
    valid = np.zeros((N, N), dtype=bool)
    latency_map = {}

    for ep in pairs:
        i = idx_src[ep['src_dev']]
        j = idx_dst[ep['dst_dev']]
        cost[i, j] = ep['latency']
        valid[i, j] = True
        latency_map[(i, j)] = ep

    # Normalize cost
    finite_costs = cost[np.isfinite(cost)]
    if len(finite_costs) == 0:
        return []

    min_cost = np.min(finite_costs)
    max_cost = np.max(finite_costs)
    norm = (cost - min_cost) / (max_cost - min_cost + 1e-6)

    # Solve assignment
    row_ind, col_ind = linear_sum_assignment(norm)

    # If not allow_partial, we require both src and dst to be covered evenly
    matched = []
    used_src = Counter()
    used_dst = Counter()
    for i, j in zip(row_ind, col_ind):
        if i < N_src and j < N_dst and valid[i, j]:
            matched.append(latency_map[(i, j)])
            used_src[src_devs[i]] += 1
            used_dst[dst_devs[j]] += 1
        elif not allow_partial:
            return []

    return matched

def process_host_pair(record):
    endpoints = record.get("endpoints", [])
    partition_map = build_partition_map(endpoints)
    result = {}

    for part_key, part_eps in partition_map.items():
        optimal = solve_partition_group(part_eps)
        if optimal:
            result[part_key] = optimal

        # add _extra candidates: same partition, but exclude used src/dst
        used_src = set(ep['src_dev'] for ep in optimal)
        used_dst = set(ep['dst_dev'] for ep in optimal)
        extras = [ep for ep in part_eps if ep['src_dev'] not in used_src and ep['dst_dev'] not in used_dst]
        extra_opt = solve_partition_group(extras, allow_partial=True)
        if extra_opt:
            result[part_key + "_extra"] = extra_opt

    record['partition_matchings'] = result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", help="Input RDMA JSON file")
    parser.add_argument("--save", action="store_true", help="Overwrite JSON file with matchings")
    args = parser.parse_args()

    with open(args.json_file) as f:
        data = json.load(f)

    for record in data:
        process_host_pair(record)

    if args.save:
        with open(args.json_file, 'w') as f:
            json.dump(data, f, indent=2)
    else:
        print(json.dumps(data, indent=2))

if __name__ == "__main__":
    main()
