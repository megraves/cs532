# Containerized ML Inference System

A containerized machine learning inference system using pre-trained SqueezeNet models. The system provides RESTful APIs for running inference with ONNX (INT8 and INT32) and PyTorch models, each running in isolated Docker containers.

## Project Overview

This project demonstrates best practices in packaging, deploying, and managing ML models as isolated services with reproducible environments. The system includes:
- **ONNX INT8 API** - Quantized INT8 ONNX model service
- **ONNX INT32 API** - Standard INT32 ONNX model service  
- **PyTorch API** - Native PyTorch model service
- **Coordinator API** - Routes requests to appropriate model services

## Documentation

This project includes three comprehensive documentation files:

### 📖 [How to Run](API_README.md)
**Location:** `API_README.md`

Complete guide for running the system, including:
- Building and starting Docker containers
- API endpoints and usage examples
- Environment variables and configuration
- Troubleshooting common issues

**Use this when:** You need to set up and run the inference services.

---

### 🧪 [How to Test](tests/Testing_README.md)
**Location:** `tests/Testing_README.md`

Testing documentation covering:
- Running the test suite
- Test file descriptions
- Coverage and validation strategies

**Use this when:** You need to verify the code works correctly.

---

### 📊 [Experiments](experiments/Experiments_README.md)
**Location:** `experiments/Experiments_README.md`

Step-by-step guide for reproducing performance experiments:
- Benchmarking setup and execution
- Resource monitoring (CPU, memory, network)
- Performance analysis and results
- Comparing ONNX vs PyTorch, quantization effects, and batching strategies

**Use this when:** You need to reproduce performance measurements and experiments.

---

## Quick Start

1. **Run the system:**
   ```bash
   docker-compose up --build
   ```
   See [API_README.md](API_README.md) for detailed instructions.

2. **Run tests:**
   ```bash
   pip install -r requirements.common.txt
   pytest
   ```
   See [tests/Testing_README.md](tests/Testing_README.md) for details.

3. **Run experiments:**
   Follow the guide in [experiments/Experiments_README.md](experiments/Experiments_README.md).

## Project Structure

```
├── api/                    # API service implementations
├── dockerfiles/            # Docker build files
├── experiments/            # Experiment results and scripts
├── models/                 # Model files (ONNX, PyTorch)
├── scripts/                # Utility scripts
├── tests/                  # Test suite
├── tools/bench/            # Benchmarking tools
├── API_README.md           # How to run documentation
├── tests/Testing_README.md # Testing documentation
└── experiments/Experiments_README.md # Experiments documentation
```

## Technologies

- **SqueezeNet** - Pre-trained models in ONNX (INT8/INT32) and PyTorch formats
- **Docker** - Containerization for isolated model services
- **FastAPI** - RESTful API framework
- **ONNX Runtime** - ONNX model inference
- **PyTorch** - Native PyTorch model inference
