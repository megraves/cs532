# COMPSCI 532 Project Proposal

## 1 Introduction

This project aims to build a containerized machine learning inference system using pre-trained models. Each model will run inside its own isolated Docker container, exposing a RESTful API endpoint for inference. The core objective is to demonstrate best practices in packaging, deploying, and managing ML models as isolated services with reproducible environments. The focus is on practical systems aspects such as latency, throughput, resource utilization, container isolation, and reliability..

## 2 Dataset

The project will utilize a subset of the ImageNet dataset, specifically the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2012–2017 image classification subset, which is the most commonly used and well-curated portion of ImageNet for benchmarking image classification models. We utilize a subset of the ImageNet validation dataset (100–500 images) to benchmark SqueezeNet inference.

## 3 Milestone Goals

1. **Model Preparation and Validation**  
   * Obtain pre-trained SqueezeNet models in ONNX and PyTorch formats.  
   * Validate inference correctness on sample images.  
2. **API and Container Development**  
   * Implement inference APIs for each model container using lightweight frameworks.  
   * Ensure each container independently loads its model and exposes a unique prediction endpoint.  
3. **Containerization and Deployment**  
   * Build a coordinator API to route requests to appropriate model containers.  
   * Dockerize the model services with isolated environments and dependencies.  
4. **Performance Measurement Setup**  
   * Develop benchmarking REST client for latency, throughput, and resource usage under varying request loads and concurrency.  
   * Monitor CPU, memory, and network utilization per container using system tools  
5. **Stress Testing and Reliability Experiments**  
   * Conduct stress tests with multiple concurrent clients.  
   * **ONNX Conversion**: Assessing the performance impact of ONNX format versus running them in their native frameworks.  
   * **Quantization**: Measuring latency, throughput, and accuracy changes resulting from model quantization.  
   * **Batching Strategies**: Evaluating the performance improvements achieved through batching inference requests.

## 4 Technologies

1. SqueezeNet Pre-Trained Models \- Using pre-trained SqueezeNet models in ONNX (INT8 quantized) and PyTorch formats for inference tasks.  
2. PyTorch \- Popular deep learning framework to run and serve the native SqueezeNet model, enabling comparison with ONNX versions.  
3. Docker \- Containerization technology used to package each model and its dependencies into isolated, reproducible environments for deployment and management.  
4. Lightweight web frameworks to build RESTful API endpoints that expose model inference services to clients.  
5. Python \- Programming language for model preprocessing, inference logic, API development, generating concurrent inference requests and performance benchmarking scripts.  
6. Monitoring Tools \- System monitoring utilities to capture container CPU, memory, network usage, and resource isolation during benchmarking (e.g. docker stats).

## 

## 5 Deliverables/Expected Outcomes

7. Containerized Model Services:  
   Fully functional Docker containers, each encapsulating a pre-trained SqueezeNet model (ONNX INT8 quantized, and native PyTorch), with RESTful API endpoints for inference.  
8. Inference API:  
   Extendable and reusable REST API service to accept image data, perform preprocessing, invoke model inference, and return labeled predictions.  
9. Benchmarking Framework:  
   Scripts and tools to generate concurrent inference requests, measure latency, throughput, and monitor CPU/memory usage of containers under load.  
10. Performance Analysis Report:  
    Comprehensive documentation of latency (average and tail), throughput under varying concurrency, resource utilization, and reliability insights.  
11. Reproducibility Artifacts:  
    Clear instructions, Dockerfiles, and source code to reproduce the entire ML serving environment on different hosts.  
    
