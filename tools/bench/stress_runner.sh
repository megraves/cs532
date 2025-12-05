#!/usr/bin/env bash
# Usage: ./stress_runner.sh <procs> <url> <image> <reqs_per_proc> <concurrency_per_proc>
PROCS=${1:-4}
URL=${2:-http://localhost:8000/predict}
IMAGE=${3:-tests/sample.jpg}
REQS=${4:-500}
CONC=${5:-8}
mkdir -p tools/bench/logs
for i in $(seq 1 $PROCS); do
  python tools/bench/benchmark_client.py --url "$URL" --image "$IMAGE" --concurrency $CONC --requests $REQS --save-latencies > tools/bench/logs/bench_$i.json 2>&1 &
  echo "Started bench process $i"
done
echo "Spawned $PROCS processes. Logs -> tools/bench/logs/"
