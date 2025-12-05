#!/usr/bin/env python3
"""
Unified results loader + plotting script.

- Scans ./experiments/* for summary.json and bench.json
- Produces:
    - all_results.json (root)
    - experiments/aggregate_results.csv
    - experiments/plots/* (same as before)
    - plots/* (same images + CSVs also in project root)
Usage:
    python tools/bench/plot_all_results.py
"""
import json
from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

ROOT = Path(".")
EXPERIMENTS = ROOT / "experiments"
OUT_PLOTS_ROOT = ROOT / "plots"
EXPT_PLOTS = EXPERIMENTS / "plots"
OUT_PLOTS_ROOT.mkdir(parents=True, exist_ok=True)
EXPT_PLOTS.mkdir(parents=True, exist_ok=True)

ALL_RESULTS_PATH = ROOT / "all_results.json"
AGG_CSV = EXPERIMENTS / "aggregate_results.csv"

# helper: parse batch number from experiment folder name
_batch_re = re.compile(r"(?:batch[_-]?|b[_-]?|_b|_batch)(\d+)", re.IGNORECASE)

def detect_model_and_batch_from_name(name: str):
    # Try to parse names like onnx_int8_batch4_run1 or onnx_int8_run1 or torch_run2 or onnx-int8-b4
    # model = first two parts joined if it looks like onnx_int8 else first part
    parts = re.split(r"[._\-]", name)
    # Heuristic: if name contains 'onnx' and 'int8' or 'int32' join them
    low = name.lower()
    if "onnx" in low and ("int8" in low or "int32" in low):
        # find int8 or int32 token
        if "int8" in low:
            model = "int8"
        elif "int32" in low:
            model = "int32"
        else:
            model = parts[0]
    elif "torch" in low or "pytorch" in low or "torch-api" in low:
        model = "torch"
    else:
        # fallback: take first token
        model = parts[0]

    # detect batch via regex
    m = _batch_re.search(name)
    batch = int(m.group(1)) if m else 1
    return model, batch

# ------------------------
# 1) Scan experiments and build a unified all_results dict
# ------------------------
all_results = {}  # structure: { model: { batch: { metrics... } } }
aggregate_rows = []  # for aggregate CSV

for summary_path in EXPERIMENTS.rglob("summary.json"):
    try:
        s = json.loads(summary_path.read_text())
    except Exception as e:
        print(f"Warning: skipping unreadable {summary_path}: {e}")
        continue

    exp_dir = summary_path.parent
    name = exp_dir.name
    model, batch = detect_model_and_batch_from_name(name)

    # get bench.json if present (for rps/requests)
    bench_summary = {}
    bench_json = exp_dir / "bench.json"
    if bench_json.exists():
        try:
            bench_summary = json.loads(bench_json.read_text())
        except Exception:
            bench_summary = {}

    latency_summary = s.get("latency_summary", {})
    container_summary = s.get("container_summary", {})

    # cpu_mean pick first container if available
    cpu_mean = None
    if isinstance(container_summary, dict) and container_summary:
        first = next(iter(container_summary.values()))
        cpu_mean = first.get("cpu_mean")

    # Compose tidy dict for all_results
    model_dict = all_results.setdefault(model, {})
    batch_key = str(batch)
    # pick values
    mean_ms = latency_summary.get("mean_ms") or bench_summary.get("latency_ms_mean")
    p50 = latency_summary.get("p50_ms") or bench_summary.get("latency_ms_p50")
    p95 = latency_summary.get("p95_ms") or bench_summary.get("latency_ms_p95")
    p99 = latency_summary.get("p99_ms") or bench_summary.get("latency_ms_p99")
    # store straightforward fields
    entry = {
        "mean_ms": mean_ms,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "cpu_mean": cpu_mean,
        "requests": bench_summary.get("requests") or latency_summary.get("count"),
        "rps": bench_summary.get("rps") or None,
        "experiment_dir": str(exp_dir),
        "summary_json": str(summary_path)
    }
    model_dict[batch_key] = entry

    # also append row for aggregate CSV
    aggregate_rows.append({
        "experiment_dir": str(exp_dir),
        "name": name,
        "model": model,
        "batch": batch,
        "requests": entry["requests"],
        "rps": entry["rps"],
        "latency_mean_ms": mean_ms,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "cpu_mean_pct": cpu_mean
    })

# write all_results.json
ALL_RESULTS_PATH.write_text(json.dumps(all_results, indent=2))
print("Wrote", ALL_RESULTS_PATH)

# ------------------------
# 2) Write aggregate CSV (experiments/aggregate_results.csv)
# ------------------------
if aggregate_rows:
    df_agg = pd.DataFrame(aggregate_rows)
    # normalize numeric columns
    for c in ["rps","latency_mean_ms","p50_ms","p95_ms","p99_ms","cpu_mean_pct"]:
        if c in df_agg.columns:
            df_agg[c] = pd.to_numeric(df_agg[c], errors='coerce')
    df_agg = df_agg.sort_values(by=["model","batch","name"]).reset_index(drop=True)
    df_agg.to_csv(AGG_CSV, index=False)
    print("Wrote", AGG_CSV)
else:
    print("No experiments discovered. Exiting.")
    raise SystemExit(0)

# ------------- helper to write files to both output locations -------------
def save_png(fig, name):
    out1 = OUT_PLOTS_ROOT / name
    out2 = EXPT_PLOTS / name
    fig.savefig(out1, bbox_inches='tight')
    fig.savefig(out2, bbox_inches='tight')
    plt.close(fig)
    print("Saved", out1, "and", out2)

def save_csv(df, name):
    out1 = OUT_PLOTS_ROOT / name
    out2 = EXPT_PLOTS / name
    df.to_csv(out1, index=False)
    df.to_csv(out2, index=False)
    print("Wrote", out1, "and", out2)

# ------------------------
# 3) Build tidy DataFrames used for plotting (latency + throughput)
# ------------------------
rows_lat = []
rows_thr = []
models = sorted(all_results.keys())
batch_sizes = sorted({int(b) for m in models for b in all_results[m].keys()})

for m in models:
    for b in batch_sizes:
        entry = all_results.get(m, {}).get(str(b))
        if not entry:
            continue
        mean_ms = entry.get("mean_ms")
        p50 = entry.get("p50_ms")
        p95 = entry.get("p95_ms")
        p99 = entry.get("p99_ms")
        cpu = entry.get("cpu_mean")
        ips = None
        if mean_ms and mean_ms > 0:
            # images / sec = batch * (1000 / mean_latency_ms)
            ips = (1000.0 * int(b)) / float(mean_ms)
        rows_lat.append({
            "model": m,
            "batch": int(b),
            "mean_ms": mean_ms,
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99,
            "cpu_mean": cpu
        })
        rows_thr.append({
            "model": m,
            "batch": int(b),
            "images_per_sec": ips
        })

df_lat = pd.DataFrame(rows_lat).sort_values(["model","batch"]).reset_index(drop=True)
df_thr = pd.DataFrame(rows_thr).sort_values(["model","batch"]).reset_index(drop=True)

# Save CSV tables
save_csv(df_lat, "latency_table.csv")
save_csv(df_thr, "throughput_table.csv")

# ------------------------
# 4) Plot 1: Latency vs Batch size per model
# ------------------------
fig = plt.figure(figsize=(8,5), dpi=200)
ax = fig.gca()
for m in models:
    dfm = df_lat[df_lat["model"]==m]
    if dfm.empty: continue
    ax.plot(dfm["batch"], dfm["mean_ms"], marker="o", label=m.upper())
    # add p95 errorbars if available
    yerr_lower = (dfm["mean_ms"] - dfm["p50_ms"]).fillna(0)
    yerr_upper = (dfm["p95_ms"] - dfm["mean_ms"]).fillna(0)
    ax.errorbar(dfm["batch"], dfm["mean_ms"],
                yerr=[yerr_lower, yerr_upper], fmt="none", alpha=0.4)
ax.set_xticks(batch_sizes)
ax.set_xlabel("Batch size")
ax.set_ylabel("Mean latency (ms)")
ax.set_title("Mean Latency vs Batch Size (per model)")
ax.grid(axis="y", linestyle="--", alpha=0.6)
ax.legend()
plt.tight_layout()
save_png(fig, "latency_vs_batch.png")

# ------------------------
# 5) Plot 2: Throughput vs Batch size per model
# ------------------------
fig = plt.figure(figsize=(8,5), dpi=200)
ax = fig.gca()
for m in models:
    dfm = df_thr[df_thr["model"]==m]
    if dfm.empty: continue
    ax.plot(dfm["batch"], dfm["images_per_sec"], marker="o", label=m.upper())
ax.set_xticks(batch_sizes)
ax.set_xlabel("Batch size")
ax.set_ylabel("Images / sec")
ax.set_title("Throughput (images/sec) vs Batch Size")
ax.grid(axis="y", linestyle="--", alpha=0.6)
ax.legend()
plt.tight_layout()
save_png(fig, "throughput_vs_batch.png")

# ------------------------
# 6) Plot 3: Grouped bar chart throughput comparison
# ------------------------
labels = [m.upper() for m in models]
x = np.arange(len(labels))
width = 0.25
b_values = {}
for bs in batch_sizes:
    b_values[bs] = [ df_thr[(df_thr.model==m)&(df_thr.batch==bs)]["images_per_sec"].values[0]
                    if not df_thr[(df_thr.model==m)&(df_thr.batch==bs)].empty else 0
                    for m in models]

fig = plt.figure(figsize=(9,5), dpi=200)
ax = fig.gca()
for i,bs in enumerate(batch_sizes):
    ax.bar(x + (i - len(batch_sizes)/2 + 0.5)*width, b_values[bs], width, label=f"B={bs}")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Images / sec")
ax.set_title("Throughput Comparison Across Models")
ax.legend()
plt.tight_layout()
save_png(fig, "throughput_comparison_grouped.png")

# ------------------------
# 7) Create pretty table images (latency + throughput)
# ------------------------
def df_to_table_image(df, title, out_name, fontsize=8, dpi=200):
    nrows, ncols = df.shape
    fig_w = max(6, 1.2 * ncols)
    fig_h = max(1.5, 0.28 * (nrows + 3))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.axis('off')
    ax.set_title(title, fontweight='bold', pad=8)
    tbl = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    try:
        tbl.auto_set_column_width(col=list(range(ncols)))
    except Exception:
        pass
    tbl.scale(1.0, 1.2)
    plt.tight_layout(pad=0.5)
    out1 = OUT_PLOTS_ROOT / out_name
    out2 = EXPT_PLOTS / out_name
    fig.savefig(out1, bbox_inches='tight')
    fig.savefig(out2, bbox_inches='tight')
    plt.close(fig)
    print("Saved", out1, "and", out2)

lat_table_df = df_lat[["model","batch","mean_ms","p50_ms","p95_ms","p99_ms","cpu_mean"]].copy()
lat_table_df.columns = ["Model","Batch","Mean(ms)","p50(ms)","p95(ms)","p99(ms)","CPU mean"]
lat_table_df = lat_table_df.sort_values(["Model","Batch"]).reset_index(drop=True)
lat_table_df.to_csv(OUT_PLOTS_ROOT / "latency_table.csv", index=False)
lat_table_df.to_csv(EXPT_PLOTS / "latency_table.csv", index=False)
df_to_table_image(lat_table_df, "Latency summary (per model & batch)", "latency_table.png", fontsize=8)

thr_table_df = df_thr.pivot(index="model", columns="batch", values="images_per_sec").reset_index().fillna(0)
# rename columns to readable form
col_map = {col: (f"B={col}" if isinstance(col,int) else f"B={col}") for col in thr_table_df.columns if col != "model"}
thr_table_df.columns = ["Model"] + [f"B={c}" for c in thr_table_df.columns[1:]]
thr_table_df.to_csv(OUT_PLOTS_ROOT / "throughput_table.csv", index=False)
thr_table_df.to_csv(EXPT_PLOTS / "throughput_table.csv", index=False)
df_to_table_image(thr_table_df, "Throughput (images/sec)", "throughput_table.png", fontsize=9)

# ------------------------
# 8) Aggregate plot: throughput (rps) vs p95 latency (similar to old aggregate script)
# ------------------------
plot_df = df_agg = pd.read_csv(AGG_CSV)
plot_df = plot_df.dropna(subset=["rps","p95_ms"])
fig = plt.figure(figsize=(8,6), dpi=200)
if not plot_df.empty:
    ax = fig.gca()
    ax.scatter(plot_df["rps"], plot_df["p95_ms"])
    for i, row in plot_df.iterrows():
        ax.text(row["rps"], row["p95_ms"], Path(row["experiment_dir"]).name, fontsize=8)
    ax.set_xlabel("Throughput (requests/sec)")
    ax.set_ylabel("p95 latency (ms)")
    ax.set_title("Throughput vs p95 latency")
    ax.grid(True)
    plt.tight_layout()
    save_png(fig, "throughput_vs_p95.png")
else:
    print("Not enough data for throughput vs p95 plot.")

# ------------------------
# 9) CPU mean bar (also like old aggregate)
# ------------------------
cpu_df = df_agg.dropna(subset=["cpu_mean_pct"]).sort_values(by="cpu_mean_pct", ascending=False)
if not cpu_df.empty:
    fig = plt.figure(figsize=(10, max(3, 0.3*len(cpu_df))), dpi=200)
    ax = fig.gca()
    names = cpu_df["name"].tolist()
    vals = cpu_df["cpu_mean_pct"].tolist()
    y_pos = np.arange(len(names))
    ax.barh(y_pos, vals, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("CPU mean (%)")
    ax.set_title("Container CPU mean per experiment")
    plt.tight_layout()
    save_png(fig, "cpu_mean_bar.png")
else:
    print("No cpu_mean_pct data available for CPU plot.")

print("\nAll outputs written to:")
print(" -", ALL_RESULTS_PATH.resolve())
print(" -", AGG_CSV.resolve())
print(" -", OUT_PLOTS_ROOT.resolve(), "and", EXPT_PLOTS.resolve())
