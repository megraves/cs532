#!/usr/bin/env bash

# Usage:
#   ./monitor_docker_stats.sh output.csv container1 [container2 ...]

OUT=${1:-docker_stats.csv}
shift || true
CONTAINERS=("$@")

if [ ${#CONTAINERS[@]} -eq 0 ]; then
  echo "Usage: $0 <out.csv> <container1> [container2 ...]"
  exit 1
fi

echo "timestamp,container,cpu_perc,mem_usage,mem_limit,mem_perc,net_io,block_io" > "$OUT"

while true; do
  # macOS-compatible timestamp
  ts=$(date +"%Y-%m-%dT%H:%M:%S")

  for c in "${CONTAINERS[@]}"; do
    raw=$(docker stats --no-stream --format '{{.Name}},{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}},{{.NetIO}},{{.BlockIO}}' "$c" 2>/dev/null)

    # If container not found (e.g., crashed), avoid breaking the script
    if [ -z "$raw" ]; then
      raw="$c,NA,NA,NA,NA,NA,NA"
    fi

    echo "$ts,$raw" >> "$OUT"
  done

  sleep 1
done
