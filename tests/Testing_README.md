# Test Suite

## Quick Start

```bash
# Install dependencies
pip install -r requirements.common.txt

# Run all tests
pytest

```

## Test Files

- `test_utils.py` - Utility functions (config, CSV, class mapping)
- `test_image_preprocessing.py` - Image preprocessing
- `test_inference_utils.py` - ONNX inference utilities
- `test_onnx_api.py` - ONNX API endpoints
- `test_torch_api.py` - PyTorch API endpoints
- `test_coordinator_api.py` - Coordinator API endpoints


## Run Specific Tests

```bash
pytest tests/test_utils.py              # Run utility tests
pytest tests/test_onnx_api.py          # Run ONNX API tests
pytest -v                                # Verbose output
```

