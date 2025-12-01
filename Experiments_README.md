# Benchmarking, Monitoring, and Evaluation Guide  
This README explains how to reproduce the ONNX INT8, ONNX INT32, and PyTorch benchmarking experiments including warm-up, full benchmark, resource monitoring, and summary analysis.

---

## Project Structure Requirements
Before running, ensure the following exist:

```
docker-compose.yml
tools/
  bench/
    benchmark_client.py
    monitor_docker_stats.sh
    analyze_results.py
experiments/
data/
  imagenette2/val/<class-folders>/*.JPEG
```

---

# 0 — Go to Repo Root

Run:

```
cd /path/to/project
pwd
ls -la
```

You should see `docker-compose.yml` and `tools/bench/`.

---

# 1 — Start the Target Model Service

Start only one model at a time.

### ONNX INT8
```
docker-compose up --build onnx-int8-api
```

### ONNX INT32
```
docker-compose up --build onnx-int32-api
```

### PyTorch
```
docker-compose up --build torch-api
```

Confirm the server is running:

```
curl http://localhost:<PORT>/health
```

Ports:
- INT8  → 8000  
- TORCH → 8001  
- INT32 → 8003  

---

# 2 — Create Experiment Folder

Example for INT8:

```
mkdir -p experiments/onnx_int8_run1
ls experiments/onnx_int8_run1
```

Repeat for INT32 and TORCH using new folders.

---

# 3 — Start Resource Monitor (runs in background)

This records CPU/memory/network stats every second.

```
nohup ./tools/bench/monitor_docker_stats.sh \
  experiments/onnx_int8_run1/stats.csv \
  onnx-int8-api \
  > experiments/onnx_int8_run1/monitor_stdout.log 2>&1 & echo $! > experiments/onnx_int8_run1/monitor.pid
```

Check monitor started:

```
cat experiments/onnx_int8_run1/monitor.pid
```

---

# 4 — Warm-up Run (small 50-request run)

```
python tools/bench/benchmark_client.py \
  --url http://localhost:8000/predict \
  --image-dir data/imagenette2/val/n01440764 \
  --concurrency 1 \
  --requests 50 \
  --payload-type multipart \
  --randomize \
  --save-latencies \
  > experiments/onnx_int8_run1/warmup_bench.json 2>&1
```

Move warm-up latencies file:

```
mv latencies.npy experiments/onnx_int8_run1/latencies_warmup.npy
```

---

# 5 — Full Benchmark Run (2000 requests)

```
python tools/bench/benchmark_client.py \
  --url http://localhost:8000/predict \
  --image-dir data/imagenette2/val/n01440764 \
  --concurrency 8 \
  --requests 2000 \
  --payload-type multipart \
  --randomize \
  --save-latencies \
  > experiments/onnx_int8_run1/bench.json 2>&1
```

Move latencies file:

```
mv latencies.npy experiments/onnx_int8_run1/latencies.npy
```

---

# 6 — Collect Container Logs

```
docker logs onnx-int8-api --since 5m \
  > experiments/onnx_int8_run1/docker_logs.txt 2>&1
```

---

# 7 — Stop Resource Monitor

```
kill $(cat experiments/onnx_int8_run1/monitor.pid)
rm experiments/onnx_int8_run1/monitor.pid
```

Verify monitor stopped:

```
ps aux | grep monitor_docker_stats | grep -v grep || true
```

---

# 8 — Analyze Results (Optional)

```
python tools/bench/analyze_results.py \
  --latencies experiments/onnx_int8_run1/latencies.npy \
  --stats experiments/onnx_int8_run1/stats.csv \
  > experiments/onnx_int8_run1/summary.json
```

Open summary:

```
cat experiments/onnx_int8_run1/summary.json
```

---

# 9 — Repeat for INT32 and TORCH

### INT32 Benchmark Example

```
mkdir -p experiments/onnx_int32_run1

nohup ./tools/bench/monitor_docker_stats.sh \
  experiments/onnx_int32_run1/stats.csv \
  onnx-int32-api \
  > experiments/onnx_int32_run1/monitor_stdout.log 2>&1 & echo $! > experiments/onnx_int32_run1/monitor.pid

python tools/bench/benchmark_client.py \
  --url http://localhost:8003/predict \
  --image-dir data/imagenette2/val/n01440764 \
  --concurrency 8 \
  --requests 2000 \
  --payload-type multipart \
  --randomize \
  --save-latencies \
  > experiments/onnx_int32_run1/bench.json 2>&1

mv latencies.npy experiments/onnx_int32_run1/latencies.npy
```

Stop monitor:

```
kill $(cat experiments/onnx_int32_run1/monitor.pid)
rm experiments/onnx_int32_run1/monitor.pid
```

---

### TORCH Benchmark Example

```
mkdir -p experiments/torch_run1

nohup ./tools/bench/monitor_docker_stats.sh \
  experiments/torch_run1/stats.csv \
  torch-api \
  > experiments/torch_run1/monitor_stdout.log 2>&1 & echo $! > experiments/torch_run1/monitor.pid

python tools/bench/benchmark_client.py \
  --url http://localhost:8001/predict \
  --image-dir data/imagenette2/val/n01440764 \
  --concurrency 8 \
  --requests 2000 \
  --payload-type multipart \
  --randomize \
  --save-latencies \
  > experiments/torch_run1/bench.json 2>&1

mv latencies.npy experiments/torch_run1/latencies.npy
```

Stop monitor:

```
kill $(cat experiments/torch_run1/monitor.pid)
rm experiments/torch_run1/monitor.pid
```

---

# 10 — What Each File Does

### `benchmark_client.py`
Sends concurrent requests to measure:
- mean latency  
- p50 / p95 / p99  
- throughput (requests/sec)  
- errors  

### `monitor_docker_stats.sh`
Records:
- CPU %
- memory use
- network I/O
- block I/O  
every second for the container.

### `bench.json`
Summary of the benchmark (latency, RPS, error rates).

### `latencies.npy`
Raw latency measurements for every request.

### `stats.csv`
Recorded container metrics during the run.

### `docker_logs.txt`
Container logs for debugging or tracing slowdowns.

### `summary.json`
Optional combined summary (latencies + resource usage).

---

# 11 — Notes
- Always run **one container at a time** for fair measurements.
- Always **warm up first**.
- Always **move latencies.npy** after each run because the script saves it to project root.
- Use different experiment folders for each model.

---

# 12 — Example Folder After All Experiments

```
experiments/
  onnx_int8_run1/
    bench.json
    warmup_bench.json
    latencies.npy
    stats.csv
    docker_logs.txt
    summary.json
  onnx_int32_run1/
    ...
  torch_run1/
    ...
```

---