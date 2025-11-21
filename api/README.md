# Model Inference APIs

RESTful API services for SqueezeNet ONNX model inference.

## Services

- **ONNX INT8 API** (port 8000) - Quantized INT8 model
- **ONNX INT32 API** (port 8003) - Standard INT32 model
- **Coordinator API** (port 8002) - Routes requests to model services

## Quick Start

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

## API Endpoints

### Coordinator API (Recommended Entry Point)

```bash
# List models
GET http://localhost:8002/models

# Single prediction
POST http://localhost:8002/predict/{model_id}
  - model_id: onnx-int8 or onnx-int32
  - Body: multipart/form-data with 'file' field

# Batch prediction
POST http://localhost:8002/predict/{model_id}/batch
  - Body: multipart/form-data with multiple 'files' fields

# Health check
GET http://localhost:8002/health
```

### Direct Model APIs

Each ONNX service provides:
- `GET /health` - Health check
- `POST /predict` - Single image prediction
- `POST /predict/batch` - Batch prediction

## Examples

### cURL

```bash
# List models
curl http://localhost:8002/models

# Single prediction
curl -X POST "http://localhost:8002/predict/onnx-int8" \
  -F "file=@image.jpg"

# Batch prediction
curl -X POST "http://localhost:8002/predict/onnx-int8/batch" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

### Python

```python
import requests

# Single prediction
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8002/predict/onnx-int8',
        files={'file': f}
    )
    result = response.json()
    print(f"Class: {result['predicted_class_name']}")
    print(f"Confidence: {result['confidence']:.4f}")
```

## Response Format

### Single Prediction

```json
{
  "predicted_class_index": 285,
  "predicted_class_name": "Egyptian cat",
  "confidence": 0.9876,
  "top5_predictions": [...]
}
```

### Batch Prediction

```json
{
  "predictions": [
    {
      "filename": "image1.jpg",
      "predicted_class_index": 285,
      "predicted_class_name": "Egyptian cat",
      "confidence": 0.9876
    }
  ]
}
```

## Configuration

### Environment Variables

**ONNX Services:**
- `MODEL_PATH` - Path to ONNX model (default: `models/squeezenet.onnx`)
- `USE_GPU` - Enable GPU (default: `false`)
- `CLASS_MAPPING` - Class mapping file path
- `PORT` - Server port (default: `8000`)

**Coordinator:**
- `ONNX_INT8_URL` - INT8 service URL
- `ONNX_INT32_URL` - INT32 service URL
- `PORT` - Server port (default: `8002`)

## API Documentation

Interactive Swagger UI:
- Coordinator: http://localhost:8002/docs
- ONNX INT8: http://localhost:8000/docs
- ONNX INT32: http://localhost:8003/docs

## Troubleshooting

```bash
# Check logs
docker-compose logs onnx-int8-api
docker-compose logs coordinator-api

# Restart services
docker-compose restart

# Rebuild containers
docker-compose up --build -d
```

## Model Requirements

- **Input**: RGB images (any size, auto-resized to 224x224)
- **Format**: JPEG, PNG, or other PIL-supported formats
- **Preprocessing**: Automatic (ImageNet normalization)
