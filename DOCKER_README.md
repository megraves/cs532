# Docker Setup and Stress Testing Guide

Quick start guide for running the ML inference APIs in Docker containers for stress testing.

## Prerequisites

1. **Docker Desktop** installed and running
   - Download: https://www.docker.com/products/docker-desktop/
   - Verify: `docker --version`

2. **Git LFS** (for pulling model files and data)
   ```bash
   brew install git-lfs  # macOS
   git lfs install
   git lfs pull data/imagenette2
   ```

3. **Sample test image** (optional, for testing)
   - Use any image from `data/imagenette2/train/` or `data/imagenette2/val/`

## Quick Start

### 1. Build All Containers

```bash
docker-compose build
```

This builds three containers:
- `onnx-int8-api` - ONNX INT8 quantized model service (port 8000)
- `onnx-int32-api` - ONNX INT32 model service (port 8003)
- `torch-api` - PyTorch model service (port 8001)
- `coordinator-api` - Coordinator service that routes requests (port 8002)

### 2. Start All Services

**Option A: Foreground mode (see logs in real-time)**
```bash
docker-compose up
```

**Option B: Detached mode (run in background)**
```bash
docker-compose up -d
```

### 3. Verify Services Are Running

Check container status:
```bash
docker-compose ps
```

You should see all 4 containers with status "Up".

### 4. Test Health Endpoints

```bash
# Test individual services
curl http://localhost:8000/health  # ONNX INT8
curl http://localhost:8003/health  # ONNX INT32
curl http://localhost:8001/health   # Torch
curl http://localhost:8002/health  # Coordinator

# Test coordinator routing
curl http://localhost:8002/models
```

## API Endpoints

### Coordinator API (Main Entry Point)
- **Base URL**: `http://localhost:8002`
- **Health**: `GET /health`
- **List Models**: `GET /models`
- **Single Prediction**: `POST /predict/{model_id}`
  - `model_id` options: `onnx-int8`, `onnx-int32`, `torch`
- **Batch Prediction**: `POST /predict/{model_id}/batch`

### Direct Service Endpoints
- **ONNX INT8**: `http://localhost:8000`
- **ONNX INT32**: `http://localhost:8003`
- **Torch**: `http://localhost:8001`

## Testing Commands

### Single Image Prediction

```bash
# Via Coordinator (recommended)
curl -X POST \
  -F "file=@data/imagenette2/train/n01440764/ILSVRC2012_val_00000293.JPEG" \
  http://localhost:8002/predict/torch

# Direct to Torch API
curl -X POST \
  -F "file=@data/imagenette2/train/n01440764/ILSVRC2012_val_00000293.JPEG" \
  http://localhost:8001/predict

# Direct to ONNX API
curl -X POST \
  -F "file=@data/imagenette2/train/n01440764/ILSVRC2012_val_00000293.JPEG" \
  http://localhost:8000/predict
```

### Batch Prediction

```bash
curl -X POST \
  -F "files=@data/imagenette2/train/n01440764/ILSVRC2012_val_00000293.JPEG" \
  -F "files=@data/imagenette2/train/n02102040/ILSVRC2012_val_00002138.JPEG" \
  http://localhost:8002/predict/torch/batch
```

## Stress Testing

### Using Apache Bench (ab)

Install Apache Bench:
```bash
# macOS
brew install httpd

# Linux
sudo apt-get install apache2-utils
```

**Single endpoint stress test:**
```bash
# Test coordinator with 1000 requests, 10 concurrent
ab -n 1000 -c 10 -p test_image.json -T multipart/form-data \
  http://localhost:8002/predict/torch
```

**Note**: For file uploads, use a tool that supports multipart/form-data like `wrk` or `hey`.

### Using hey (HTTP load testing tool)

Install:
```bash
# macOS
brew install hey

# Or download from: https://github.com/rakyll/hey
```

**Stress test with file upload:**
```bash
# Create a test script (test_upload.sh)
cat > test_upload.sh << 'EOF'
#!/bin/bash
curl -X POST \
  -F "file=@data/imagenette2/train/n01440764/ILSVRC2012_val_00000293.JPEG" \
  http://localhost:8002/predict/torch
EOF

chmod +x test_upload.sh

# Run with hey (1000 requests, 50 concurrent)
hey -n 1000 -c 50 -m POST -H "Content-Type: multipart/form-data" \
  -D data/imagenette2/train/n01440764/ILSVRC2012_val_00000293.JPEG \
  http://localhost:8002/predict/torch
```

### Using Python Script for Stress Testing

Create `stress_test.py`:
```python
import requests
import concurrent.futures
import time
from pathlib import Path

def send_request(image_path, url):
    """Send a single prediction request"""
    with open(image_path, 'rb') as f:
        files = {'file': f}
        start = time.time()
        try:
            response = requests.post(url, files=files, timeout=30)
            elapsed = time.time() - start
            return {
                'status': response.status_code,
                'time': elapsed,
                'success': response.status_code == 200
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e), 'success': False}

def stress_test(url, image_path, num_requests=100, concurrency=10):
    """Run stress test"""
    print(f"Stress testing {url}")
    print(f"Requests: {num_requests}, Concurrency: {concurrency}")
    
    results = []
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(send_request, image_path, url)
            for _ in range(num_requests)
        ]
        
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r.get('success', False))
    
    print(f"\nResults:")
    print(f"  Total requests: {num_requests}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {num_requests - successful}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Requests/sec: {num_requests/total_time:.2f}")
    
    if successful > 0:
        times = [r['time'] for r in results if r.get('success')]
        print(f"  Avg response time: {sum(times)/len(times):.2f}s")
        print(f"  Min response time: {min(times):.2f}s")
        print(f"  Max response time: {max(times):.2f}s")

if __name__ == "__main__":
    image_path = "data/imagenette2/train/n01440764/ILSVRC2012_val_00000293.JPEG"
    
    # Test coordinator
    stress_test(
        "http://localhost:8002/predict/torch",
        image_path,
        num_requests=100,
        concurrency=10
    )
```

Run:
```bash
pip install requests
python stress_test.py
```

## Monitoring Container Resources

### View Container Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f torch-api
docker-compose logs -f coordinator-api
```

### Monitor Resource Usage

```bash
# Real-time stats
docker stats

# Or for specific containers
docker stats onnx-int8-api torch-api coordinator-api
```

### Check Container Health

```bash
# Container status
docker-compose ps

# Detailed info
docker inspect coordinator-api
```

## Stopping Services

```bash
# Stop containers (keeps them)
docker-compose stop

# Stop and remove containers
docker-compose down

# Stop, remove containers, and remove volumes
docker-compose down -v
```

## Troubleshooting

### Containers won't start

1. **Check if ports are already in use:**
   ```bash
   lsof -i :8000
   lsof -i :8001
   lsof -i :8002
   lsof -i :8003
   ```

2. **Check Docker is running:**
   ```bash
   docker ps
   ```

3. **View error logs:**
   ```bash
   docker-compose logs
   ```

### Models not loading

1. **Verify model files exist:**
   ```bash
   ls -lh models/
   ```

2. **Check container can access models:**
   ```bash
   docker-compose exec onnx-int8-api ls -la /app/models
   ```

### Class mapping issues

1. **Verify class mapping file is pulled:**
   ```bash
   head data/imagenette2/index_to_class.txt
   ```

2. **If it shows Git LFS pointer, pull it:**
   ```bash
   git lfs pull data/imagenette2/index_to_class.txt
   ```

### High memory usage

- Reduce concurrent requests in stress tests
- Check individual container memory: `docker stats`
- Consider increasing Docker Desktop memory limit

## Performance Tips for Stress Testing

1. **Warm up services** - Send a few requests before starting stress tests
2. **Monitor resources** - Watch CPU/memory during tests
3. **Start small** - Begin with low concurrency and gradually increase
4. **Test individually** - Test each service separately before coordinator
5. **Check logs** - Monitor for errors or warnings during stress tests

## Example Stress Test Scenarios

### Scenario 1: Light Load
- 100 requests, 5 concurrent
- Good for initial testing

### Scenario 2: Medium Load
- 500 requests, 20 concurrent
- Simulates moderate traffic

### Scenario 3: Heavy Load
- 1000 requests, 50 concurrent
- Tests system limits

### Scenario 4: Sustained Load
- Continuous requests for 5-10 minutes
- Tests for memory leaks and stability

## Next Steps

Once stress testing is complete, you can:
1. Analyze response times and error rates
2. Identify bottlenecks
3. Optimize container resources (CPU/memory)
4. Scale services horizontally if needed
5. Prepare for AWS deployment

For AWS deployment, see `aws/README.md`.

