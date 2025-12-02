#!/usr/bin/env python3
"""
Aggregate summary.json and monitor stats for all experiments under ./experiments/

Usage:
  python tools/bench/aggregate_results.py

Outputs:
  - experiments/aggregate_results.csv
  - experiments/plots/throughput_vs_p95.png
  - experiments/plots/cpu_mean_bar.png
"""
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

ROOT = Path("experiments")
OUT_CSV = ROOT / "aggregate_results.csv"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

rows = []

for summary_file in ROOT.rglob("summary.json"):
    try:
        s = json.loads(summary_file.read_text())
    except Exception as e:
        print("Skipping unreadable", summary_file, e)
        continue

    # Basic metadata from folder name
    exp_dir = summary_file.parent
    name = exp_dir.name  # e.g., onnx_int8_run1 or onnx_int8_batch4_run1
    # try to parse fields from name: model, batch, run, concurrency might be stored in a notes file otherwise
    parts = name.split("_")
    model = parts[0] if len(parts)>0 else name
    batch = None
    run = None
    # try to locate bench.json for more info
    bench_json = exp_dir / "bench.json"
    bench_summary = {}
    if bench_json.exists():
        try:
            bench_summary = json.loads(bench_json.read_text())
        except:
            bench_summary = {}
    # fallback to summary.json content for latencies
    latency_summary = s.get("latency_summary", {})
    container_summary = s.get("container_summary", {})

    # pick cpu_mean from first container if available
    cpu_mean = None
    if container_summary:
        # pick the first container's cpu_mean
        first = next(iter(container_summary.values()))
        cpu_mean = first.get("cpu_mean")

    row = {
        "experiment_dir": str(exp_dir),
        "name": name,
        "model": model,
        "requests": bench_summary.get("requests") or latency_summary.get("count"),
        "rps": bench_summary.get("rps") or None,
        "latency_mean_ms": latency_summary.get("mean_ms") or bench_summary.get("latency_ms_mean"),
        "p50_ms": latency_summary.get("p50_ms") or bench_summary.get("latency_ms_p50"),
        "p95_ms": latency_summary.get("p95_ms") or bench_summary.get("latency_ms_p95"),
        "p99_ms": latency_summary.get("p99_ms") or bench_summary.get("latency_ms_p99"),
        "latency_min_ms": latency_summary.get("min_ms") or bench_summary.get("latency_ms_min"),
        "latency_max_ms": latency_summary.get("max_ms") or bench_summary.get("latency_ms_max"),
        "cpu_mean_pct": cpu_mean,
    }
    rows.append(row)

if not rows:
    print("No summary.json files found under experiments/. Nothing to aggregate.")
    raise SystemExit(0)

df = pd.DataFrame(rows)
# Normalize numeric columns
for c in ["rps","latency_mean_ms","p50_ms","p95_ms","p99_ms","cpu_mean_pct"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

df = df.sort_values(by=["model","name"]).reset_index(drop=True)
df.to_csv(OUT_CSV, index=False)
print("Wrote", OUT_CSV)

# Simple plot: throughput (rps) vs p95 latency
plt.figure(figsize=(8,6))
# drop rows missing rps or p95
plot_df = df.dropna(subset=["rps","p95_ms"])
if not plot_df.empty:
    plt.scatter(plot_df["rps"], plot_df["p95_ms"])
    for i, row in plot_df.iterrows():
        plt.text(row["rps"], row["p95_ms"], Path(row["experiment_dir"]).name, fontsize=8)
    plt.xlabel("Throughput (requests/sec)")
    plt.ylabel("p95 latency (ms)")
    plt.title("Throughput vs p95 latency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "throughput_vs_p95.png")
    print("Wrote plot:", PLOTS_DIR / "throughput_vs_p95.png")
else:
    print("Not enough data for throughput vs p95 plot.")

# Bar plot: cpu_mean_pct per experiment (if available)
plt.figure(figsize=(10,6))
cpu_df = df.dropna(subset=["cpu_mean_pct"])
if not cpu_df.empty:
    names = cpu_df["name"].tolist()
    cpu_vals = cpu_df["cpu_mean_pct"].tolist()
    y_pos = np.arange(len(names))
    plt.barh(y_pos, cpu_vals, align='center')
    plt.yticks(y_pos, names, fontsize=8)
    plt.xlabel("CPU mean (%)")
    plt.title("Container CPU mean per experiment")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cpu_mean_bar.png")
    print("Wrote plot:", PLOTS_DIR / "cpu_mean_bar.png")
else:
    print("No cpu_mean_pct data available for CPU plot.")
