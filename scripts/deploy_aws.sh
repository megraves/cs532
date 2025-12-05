#!/bin/bash
# AWS Deployment Script
# This script builds and pushes Docker images to AWS ECR, then deploys to ECS

set -e

# Configuration
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-}"
ECR_REPO_PREFIX="${ECR_REPO_PREFIX:-cs532-ml-api}"
USE_GPU="${USE_GPU:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    echo_info "Checking prerequisites..."
    
    if ! command -v aws &> /dev/null; then
        echo_error "AWS CLI not found. Please install it: https://aws.amazon.com/cli/"
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        echo_error "Docker not found. Please install it: https://www.docker.com/"
        exit 1
    fi
    
    if [ -z "$AWS_ACCOUNT_ID" ]; then
        AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null)
        if [ -z "$AWS_ACCOUNT_ID" ]; then
            echo_error "Could not determine AWS Account ID. Please set AWS_ACCOUNT_ID or configure AWS CLI."
            exit 1
        fi
        echo_info "Detected AWS Account ID: $AWS_ACCOUNT_ID"
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        echo_error "AWS credentials not configured. Please run 'aws configure'"
        exit 1
    fi
    
    echo_info "Prerequisites check passed!"
}

# Create ECR repositories
create_ecr_repos() {
    echo_info "Creating ECR repositories..."
    
    repos=("onnx-api" "torch-api" "coordinator-api")
    
    for repo in "${repos[@]}"; do
        full_repo_name="${ECR_REPO_PREFIX}-${repo}"
        
        if aws ecr describe-repositories --repository-names "$full_repo_name" --region "$AWS_REGION" &> /dev/null; then
            echo_warn "Repository $full_repo_name already exists, skipping..."
        else
            echo_info "Creating repository: $full_repo_name"
            aws ecr create-repository \
                --repository-name "$full_repo_name" \
                --region "$AWS_REGION" \
                --image-scanning-configuration scanOnPush=true \
                --image-tag-mutability MUTABLE
        fi
    done
    
    echo_info "ECR repositories ready!"
}

# Build and push Docker images
build_and_push() {
    echo_info "Building and pushing Docker images..."
    
    ECR_BASE="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
    
    # Login to ECR
    echo_info "Logging in to ECR..."
    aws ecr get-login-password --region "$AWS_REGION" | \
        docker login --username AWS --password-stdin "$ECR_BASE"
    
    # Build args for GPU
    BUILD_ARGS=""
    if [ "$USE_GPU" = "true" ]; then
        BUILD_ARGS="--build-arg INSTALL_GPU_DEPS=true"
        echo_info "Building with GPU support enabled"
    else
        echo_info "Building with CPU-only support"
    fi
    
    # Build and push ONNX API
    echo_info "Building ONNX API image..."
    docker build $BUILD_ARGS \
        -f dockerfiles/Dockerfile.onnx \
        -t "${ECR_REPO_PREFIX}-onnx-api:latest" \
        -t "${ECR_BASE}/${ECR_REPO_PREFIX}-onnx-api:latest" \
        .
    
    echo_info "Pushing ONNX API image..."
    docker push "${ECR_BASE}/${ECR_REPO_PREFIX}-onnx-api:latest"
    
    # Build and push Torch API
    echo_info "Building Torch API image..."
    docker build $BUILD_ARGS \
        -f dockerfiles/Dockerfile.torch \
        -t "${ECR_REPO_PREFIX}-torch-api:latest" \
        -t "${ECR_BASE}/${ECR_REPO_PREFIX}-torch-api:latest" \
        .
    
    echo_info "Pushing Torch API image..."
    docker push "${ECR_BASE}/${ECR_REPO_PREFIX}-torch-api:latest"
    
    # Build and push Coordinator API
    echo_info "Building Coordinator API image..."
    docker build \
        -f dockerfiles/Dockerfile.coordinator \
        -t "${ECR_REPO_PREFIX}-coordinator-api:latest" \
        -t "${ECR_BASE}/${ECR_REPO_PREFIX}-coordinator-api:latest" \
        .
    
    echo_info "Pushing Coordinator API image..."
    docker push "${ECR_BASE}/${ECR_REPO_PREFIX}-coordinator-api:latest"
    
    echo_info "All images built and pushed successfully!"
}

# Main execution
main() {
    echo_info "Starting AWS deployment process..."
    echo_info "Region: $AWS_REGION"
    echo_info "Account ID: $AWS_ACCOUNT_ID"
    echo_info "GPU Support: $USE_GPU"
    echo ""
    
    check_prerequisites
    create_ecr_repos
    build_and_push
    
    echo_info "Deployment preparation complete!"
    echo_warn "Next steps:"
    echo "  1. Create ECS cluster and task definitions"
    echo "  2. Configure load balancer and networking"
    echo "  3. Set up environment variables and secrets"
    echo "  4. Deploy services using ECS or EKS"
    echo ""
    echo "Image URIs:"
    echo "  ONNX: ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_PREFIX}-onnx-api:latest"
    echo "  Torch: ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_PREFIX}-torch-api:latest"
    echo "  Coordinator: ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_PREFIX}-coordinator-api:latest"
}

# Run main function
main "$@"

