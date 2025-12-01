#!/usr/bin/env python3
"""
Simple analyzer: consume bench_result (stdout JSON) or saved latencies.npy with stats.csv to produce summary.
Usage: python analyze_results.py --bench-json experiments/.../bench_result.json --stats experiments/.../stats.csv
"""
import argparse, json, numpy as np, pandas as pd
from pathlib import Path

def summarize_latencies(latencies_path):
    a = np.load(latencies_path)
    return {
        "count": int(len(a)),
        "mean_ms": float(a.mean()),
        "p50_ms": float(np.percentile(a,50)),
        "p95_ms": float(np.percentile(a,95)),
        "p99_ms": float(np.percentile(a,99)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max())
    }

def summarize_stats_csv(stats_csv):
    df = pd.read_csv(stats_csv)
    # aggregate by container
    groups = {}
    for name, g in df.groupby("container"):
        # parse cpu perc like "0.12%" or "NA"
        def parse_cpu(x):
            try:
                return float(str(x).replace("%",""))
            except:
                return None
        cpu = g["cpu_perc"].apply(parse_cpu).dropna()
        groups[name] = {
            "cpu_mean": float(cpu.mean()) if not cpu.empty else None,
            "cpu_max": float(cpu.max()) if not cpu.empty else None,
            "mem_mean": None  # could parse mem_usage field if needed
        }
    return groups

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latencies", required=False)
    p.add_argument("--stats", required=False)
    args = p.parse_args()
    out = {}
    if args.latencies:
        out["latency_summary"] = summarize_latencies(args.latencies)
    if args.stats:
        out["container_summary"] = summarize_stats_csv(args.stats)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
