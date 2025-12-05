# Model Inference API Documentation

This project provides containerized RESTful APIs for running SqueezeNet model inference using ONNX and PyTorch formats.

## Architecture

The system consists of:
1. **ONNX INT8 API** - Quantized INT8 ONNX model service (port 8000)
2. **ONNX INT32 API** - Standard INT32 ONNX model service (port 8003)
3. **PyTorch API** - Native PyTorch model service (port 8001)
4. **Coordinator API** - Routes requests to appropriate model services (port 8002)

## Quick Start

### 1. Build and Start All Services

```bash
docker-compose up --build
```

This will:
- Build Docker images for all services
- Start all containers
- Set up networking between container

### 2. Check Service Health

```bash
# Check coordinator
curl http://localhost:8002/health

# Check individual services
curl http://localhost:8000/health  # ONNX INT8
curl http://localhost:8003/health  # ONNX INT32
curl http://localhost:8001/health  # PyTorch
```

### 3. List Available Models

```bash
curl http://localhost:8002/models
```

### 4. Make Predictions

#### Using Coordinator API (Recommended)

```bash
# Predict with ONNX INT8 model
curl -X POST "http://localhost:8002/predict/onnx-int8" \
  -F "file=@path/to/image.jpg"

# Predict with PyTorch model
curl -X POST "http://localhost:8002/predict/torch" \
  -F "file=@path/to/image.jpg"

# Predict with ONNX INT32 model
curl -X POST "http://localhost:8002/predict/onnx-int32" \
  -F "file=@path/to/image.jpg"
```

#### Direct API Calls

```bash
# ONNX INT8 API
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path/to/image.jpg"

# PyTorch API
curl -X POST "http://localhost:8001/predict" \
  -F "file=@path/to/image.jpg"
```

### 5. Batch Predictions

```bash
curl -X POST "http://localhost:8002/predict/torch/batch" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg"
```

## API Endpoints

### Coordinator API (Port 8002)

- `GET /` - Root endpoint with service info
- `GET /health` - Health check
- `GET /models` - List available models
- `GET /health/{model_id}` - Check health of specific model
- `POST /predict/{model_id}` - Route prediction to model
- `POST /predict/{model_id}/batch` - Route batch prediction to model

**Model IDs:**
- `onnx-int8` - ONNX INT8 quantized model
- `onnx-int32` - ONNX INT32 model
- `torch` - PyTorch model

### Model APIs

Each model service provides:

- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /predict` - Single image prediction
- `POST /predict/batch` - Batch image prediction

## Response Format

### Single Prediction Response

```json
{
  "predicted_class_index": 285,
  "predicted_class_name": "Egyptian cat",
  "confidence": 0.9876,
  "top5_predictions": [
    {
      "class_index": 285,
      "class_name": "Egyptian cat",
      "confidence": 0.9876
    },
    ...
  ]
}
```

### Batch Prediction Response

```json
{
  "predictions": [
    {
      "filename": "image1.jpg",
      "predicted_class_index": 285,
      "predicted_class_name": "Egyptian cat",
      "confidence": 0.9876
    },
    ...
  ]
}
```

## Docker Commands

### Build Individual Services

```bash
# Build ONNX service
docker build -f dockerfiles/Dockerfile.onnx -t onnx-api .

# Build PyTorch service
docker build -f dockerfiles/Dockerfile.torch -t torch-api .

# Build Coordinator
docker build -f dockerfiles/Dockerfile.coordinator -t coordinator-api .
```

### Run Individual Containers

```bash
# ONNX INT8
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/data:/app/data:ro \
  -e MODEL_PATH=models/SqueezeNet_int8.onnx \
  -e USE_GPU=false \
  onnx-api

# PyTorch
docker run -p 8001:8001 \
  -v $(pwd)/data:/app/data:ro \
  -e MODEL_NAME=squeezenet1_1 \
  -e USE_GPU=false \
  torch-api
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f onnx-int8-api
docker-compose logs -f torch-api
docker-compose logs -f coordinator-api
```

### Stop Services

```bash
docker-compose down
```

## Environment Variables

### ONNX API
- `MODEL_PATH` - Path to ONNX model file (default: `models/squeezenet.onnx`)
- `USE_GPU` - Enable GPU (default: `false`)
- `CLASS_MAPPING` - Path to class mapping file
- `PORT` - Server port (default: `8000`)

### PyTorch API
- `MODEL_NAME` - Model name (default: `squeezenet1_1`)
- `USE_GPU` - Enable GPU (default: `false`)
- `CLASS_MAPPING` - Path to class mapping file
- `PORT` - Server port (default: `8001`)

### Coordinator API
- `ONNX_INT8_URL` - ONNX INT8 service URL
- `ONNX_INT32_URL` - ONNX INT32 service URL
- `TORCH_URL` - PyTorch service URL
- `PORT` - Server port (default: `8002`)

## Testing with Python

```python
import requests

# Single prediction
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8002/predict/torch',
        files={'file': f}
    )
    print(response.json())

# Batch prediction
files = [
    ('files', open('image1.jpg', 'rb')),
    ('files', open('image2.jpg', 'rb'))
]
response = requests.post(
    'http://localhost:8002/predict/torch/batch',
    files=files
)
print(response.json())
```

## Performance Monitoring

Monitor container resource usage:

```bash
# View container stats
docker stats

# View specific container stats
docker stats onnx-int8-api torch-api coordinator-api
```

## Troubleshooting

1. **Port conflicts**: Change ports in `docker-compose.yml` if ports are already in use
2. **Model not found**: Ensure model files exist in `models/` directory
3. **Class mapping errors**: Verify `data/imagenette2/index_to_class.txt` exists
4. **GPU support**: Set `USE_GPU=true` and ensure NVIDIA Docker runtime is installed

## Development

To run APIs locally without Docker:

```bash
# Install dependencies
pip install -r requirements.txt

# Run ONNX API
MODEL_PATH=models/SqueezeNet_int8.onnx python -m api.onnx_api

# Run PyTorch API
MODEL_NAME=squeezenet1_1 python -m api.torch_api

# Run Coordinator
python -m api.coordinator_api
```

