# AWS Deployment Guide

This guide walks you through deploying the ML inference APIs to AWS.

## Prerequisites

1. **AWS CLI installed and configured**
   ```bash
   aws configure
   ```

2. **Docker installed and running**

3. **AWS Account with appropriate permissions:**
   - ECR (Elastic Container Registry) - for storing images
   - ECS (Elastic Container Service) - for running containers
   - IAM - for service roles
   - CloudWatch Logs - for logging

## Deployment Options

### Option 1: ECS Fargate (Recommended for simplicity)

**Pros:**
- Serverless, no EC2 management
- Easy scaling
- Pay per use
- Good for CPU workloads

**Cons:**
- Limited GPU support (requires EC2 launch type)
- Slightly more expensive than EC2

### Option 2: ECS EC2 (For GPU support)

**Pros:**
- Full GPU support (g4dn, p3 instances)
- More control over infrastructure
- Better for high-performance workloads

**Cons:**
- Requires EC2 instance management
- More complex setup

### Option 3: EKS (Kubernetes)

**Pros:**
- Industry standard orchestration
- Excellent for complex deployments
- Great GPU support

**Cons:**
- More complex setup
- Higher operational overhead

## Step-by-Step Deployment

### Step 1: Build and Push Images to ECR

```bash
# Make script executable
chmod +x scripts/deploy_aws.sh

# Set your AWS region (optional, defaults to us-east-1)
export AWS_REGION=us-east-1

# For CPU-only deployment (Mac/local testing)
./scripts/deploy_aws.sh

# For GPU deployment (AWS)
USE_GPU=true ./scripts/deploy_aws.sh
```

This script will:
1. Check prerequisites
2. Create ECR repositories
3. Build Docker images
4. Push images to ECR

### Step 2: Create ECS Cluster

```bash
aws ecs create-cluster \
    --cluster-name cs532-ml-cluster \
    --region us-east-1
```

### Step 3: Create CloudWatch Log Groups

```bash
aws logs create-log-group --log-group-name /ecs/cs532-ml-onnx-api
aws logs create-log-group --log-group-name /ecs/cs532-ml-torch-api
aws logs create-log-group --log-group-name /ecs/cs532-ml-coordinator-api
```

### Step 4: Create Task Definitions

Update the image URIs in `aws/ecs-task-definitions.json` with your account ID, then:

```bash
# Register ONNX task definition
aws ecs register-task-definition \
    --cli-input-json file://aws/ecs-task-definitions.json \
    --region us-east-1

# Create similar task definitions for torch-api and coordinator-api
```

### Step 5: Create ECS Service

```bash
aws ecs create-service \
    --cluster cs532-ml-cluster \
    --service-name onnx-api-service \
    --task-definition cs532-ml-onnx-api \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}" \
    --region us-east-1
```

### Step 6: Set Up Load Balancer (Optional but Recommended)

1. Create Application Load Balancer (ALB)
2. Create target groups for each service
3. Configure health checks
4. Set up routing rules

## Environment Variables

Key environment variables for each service:

### ONNX API
- `MODEL_PATH`: Path to ONNX model file (default: `models/squeezenet.onnx`)
- `USE_GPU`: Enable GPU (default: `false`)
- `CLASS_MAPPING`: Path to class mapping file
- `PORT`: Service port (default: `8000`)

### Torch API
- `MODEL_NAME`: PyTorch model name (default: `squeezenet1_1`)
- `USE_GPU`: Enable GPU (default: `false`)
- `CLASS_MAPPING`: Path to class mapping file
- `PORT`: Service port (default: `8001`)

### Coordinator API
- `ONNX_INT8_URL`: URL to ONNX INT8 service
- `ONNX_INT32_URL`: URL to ONNX INT32 service
- `TORCH_URL`: URL to Torch service
- `PORT`: Service port (default: `8002`)

## GPU Deployment

For GPU support, you'll need:

1. **EC2 Launch Type** (Fargate doesn't support GPU)
2. **GPU Instance Types**: g4dn.xlarge, p3.2xlarge, etc.
3. **NVIDIA Container Runtime** on the EC2 instances
4. **Build images with GPU support**:
   ```bash
   USE_GPU=true ./scripts/deploy_aws.sh
   ```

## Cost Optimization Tips

1. **Use Fargate Spot** for non-critical workloads (up to 70% savings)
2. **Right-size containers** - adjust CPU/memory based on actual usage
3. **Use Auto Scaling** to scale down during low traffic
4. **Consider Reserved Capacity** for predictable workloads

## Monitoring and Logging

- **CloudWatch Logs**: All container logs are automatically sent to CloudWatch
- **CloudWatch Metrics**: Monitor CPU, memory, and request metrics
- **ECS Service Events**: Track service deployment and health events

## Troubleshooting

### Images not pulling
- Check ECR repository permissions
- Verify image tags match
- Ensure task execution role has ECR permissions

### Containers failing to start
- Check CloudWatch logs
- Verify environment variables
- Check health check configuration
- Ensure models and data files are accessible

### GPU not working
- Verify instance type supports GPU
- Check NVIDIA container runtime is installed
- Ensure `USE_GPU=true` in environment
- Verify GPU drivers are available

## Next Steps

1. Set up CI/CD pipeline (GitHub Actions, AWS CodePipeline)
2. Configure auto-scaling policies
3. Set up monitoring and alerting
4. Implement blue/green deployments
5. Add API Gateway for external access

