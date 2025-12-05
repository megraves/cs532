#!/usr/bin/env python3
"""
Simple async benchmarking client.
ONLY supports JPEG files (uppercase .JPEG or lowercase .jpeg).

Usage:
  python benchmark_client.py \
    --url http://localhost:8001/predict \
    --image-dir data/imagenette2/val/n01440764 \
    --concurrency 1 --requests 50 \
    --payload-type multipart \
    --randomize --save-latencies
"""

import argparse, asyncio, time, json, base64, random
from pathlib import Path
import statistics
import numpy as np
import httpx


# ---------------------------
# HTTP send helpers
# ---------------------------

async def send_request(client, url, payload, files=None, timeout=30):
    start = time.perf_counter()
    try:
        if files:
            r = await client.post(url, files=files, timeout=timeout)
        else:
            r = await client.post(url, json=payload, timeout=timeout)
        latency = (time.perf_counter() - start) * 1000.0
        return latency, r.status_code
    except Exception:
        latency = (time.perf_counter() - start) * 1000.0
        return latency, "ERR"


async def worker(q, url, results, client):
    while True:
        item = await q.get()
        if item is None:
            q.task_done()
            break
        payload, files = item
        latency, status = await send_request(client, url, payload, files)
        results.append((latency, status))
        q.task_done()


async def run_benchmark(url, payloads, concurrency, requests_total):
    q = asyncio.Queue()
    results = []

    async with httpx.AsyncClient() as client:
        # populate queue
        for i in range(requests_total):
            q.put_nowait(payloads[i % len(payloads)])

        # start workers
        tasks = [asyncio.create_task(worker(q, url, results, client))
                 for _ in range(concurrency)]

        # stop tokens
        for _ in range(concurrency):
            q.put_nowait(None)

        start = time.perf_counter()
        await q.join()
        total_time = time.perf_counter() - start

        for t in tasks:
            t.cancel()

    # aggregate metrics
    latencies = [r[0] for r in results if r[1] != "ERR"]
    errors = len([r for r in results if r[1] == "ERR"])

    return {
        "requests": len(results),
        "errors": errors,
        "total_time_sec": total_time,
        "rps": len(results)/total_time if total_time > 0 else None,
        "latency_ms_mean": statistics.mean(latencies) if latencies else None,
        "latency_ms_p50": np.percentile(latencies, 50) if latencies else None,
        "latency_ms_p95": np.percentile(latencies, 95) if latencies else None,
        "latency_ms_p99": np.percentile(latencies, 99) if latencies else None,
        "latency_ms_min": min(latencies) if latencies else None,
        "latency_ms_max": max(latencies) if latencies else None,
    }, latencies


# ---------------------------
# Payload builders
# ---------------------------

def load_jpeg_images(image_dir, randomize=False):
    p = Path(image_dir)

    # ONLY JPEG support
    imgs = sorted([str(x) for x in p.glob("*.JPEG")] +
                  [str(x) for x in p.glob("*.jpeg")])

    if not imgs:
        raise ValueError(f"No JPEG images found in: {image_dir}")

    if randomize:
        random.shuffle(imgs)

    return imgs


def build_json_payloads_from_dir(image_dir, batch_size=1, randomize=False):
    imgs = load_jpeg_images(image_dir, randomize)
    payloads = []

    if batch_size == 1:
        for path in imgs:
            b = base64.b64encode(Path(path).read_bytes()).decode("ascii")
            payloads.append(({"image": b}, None))
    else:
        if randomize:
            # create many payloads each with batch_size distinct random images
            for _ in range(len(imgs)):
                chosen = random.sample(imgs, batch_size)
                b64s = [base64.b64encode(Path(p).read_bytes()).decode("ascii") for p in chosen]
                payloads.append(({"images": b64s}, None))
        else:
            # deterministic chunking, wrapping if necessary
            n = len(imgs)
            for i in range(0, n, batch_size):
                chunk = [imgs[(i + j) % n] for j in range(batch_size)]
                b64s = [base64.b64encode(Path(p).read_bytes()).decode("ascii") for p in chunk]
                payloads.append(({"images": b64s}, None))

    return payloads


def build_multipart_payloads_from_dir(image_dir, batch_size=1, randomize=False):
    """
    Returns list of payloads: each is (None, files_list)
    files_list must be a list of tuples:
      [("files", (filename, bytes, "image/jpeg")), ...]
    """
    imgs = load_jpeg_images(image_dir, randomize)
    payloads = []

    if batch_size == 1:
        for path in imgs:
            files = [("files", (Path(path).name, Path(path).read_bytes(), "image/jpeg"))]
            payloads.append((None, files))
    else:
        if randomize:
            for _ in range(len(imgs)):
                chosen = random.sample(imgs, batch_size)
                files = [("files", (Path(p).name, Path(p).read_bytes(), "image/jpeg")) for p in chosen]
                payloads.append((None, files))
        else:
            n = len(imgs)
            for i in range(0, n, batch_size):
                chunk = [imgs[(i + j) % n] for j in range(batch_size)]
                files = [("files", (Path(p).name, Path(p).read_bytes(), "image/jpeg")) for p in chunk]
                payloads.append((None, files))

    return payloads

# ---------------------------
# Main
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--image-dir", required=True,
                   help="Directory with .JPEG images")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--requests", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--payload-type", choices=["json","multipart"], default="json")
    p.add_argument("--randomize", action="store_true")
    p.add_argument("--save-latencies", action="store_true")
    p.add_argument("--latency-path", type=str, default=None,
               help="Where to save latencies.npy. Default = current directory.")
    return p.parse_args()


def main():
    args = parse_args()

    # Build list of payloads
    if args.payload_type == "json":
        payloads = build_json_payloads_from_dir(
            args.image-dir, batch_size=args.batch_size, randomize=args.randomize
        )
    else:
        payloads = build_multipart_payloads_from_dir(
            args.image_dir, batch_size=args.batch_size, randomize=args.randomize
        )

    summary, latencies = asyncio.run(
        run_benchmark(args.url, payloads, args.concurrency, args.requests)
    )

    print(json.dumps(summary, indent=2))

    if args.save_latencies:
        if args.latency_path:
            out = Path(args.latency_path)
        else:
            out = Path("latencies.npy")  # default
        np.save(out, np.array(latencies))
        print("Saved latencies ->", out.resolve())


if __name__ == "__main__":
    main()
