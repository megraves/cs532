#!/bin/bash

# Test script for containerized APIs
# Make sure Docker containers are running first: docker-compose up -d

echo "=========================================="
echo "Testing Containerized Model APIs"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if containers are running
echo -e "${YELLOW}Checking if containers are running...${NC}"
if ! docker ps | grep -q "onnx-int8-api\|torch-api\|coordinator-api"; then
    echo -e "${RED}Error: Containers are not running. Please start them with: docker-compose up -d${NC}"
    exit 1
fi
echo -e "${GREEN}Containers are running!${NC}"
echo ""

# Test Coordinator API Health
echo -e "${YELLOW}1. Testing Coordinator API Health...${NC}"
COORDINATOR_HEALTH=$(curl -s http://localhost:8002/health)
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Coordinator API is healthy${NC}"
    echo "Response: $COORDINATOR_HEALTH"
else
    echo -e "${RED}✗ Coordinator API health check failed${NC}"
fi
echo ""

# Test Model Services Health
echo -e "${YELLOW}2. Testing Model Services Health...${NC}"

echo "  - ONNX INT8 API..."
ONNX_INT8_HEALTH=$(curl -s http://localhost:8000/health)
if [ $? -eq 0 ]; then
    echo -e "  ${GREEN}✓ ONNX INT8 API is healthy${NC}"
else
    echo -e "  ${RED}✗ ONNX INT8 API health check failed${NC}"
fi

echo "  - ONNX INT32 API..."
ONNX_INT32_HEALTH=$(curl -s http://localhost:8003/health)
if [ $? -eq 0 ]; then
    echo -e "  ${GREEN}✓ ONNX INT32 API is healthy${NC}"
else
    echo -e "  ${RED}✗ ONNX INT32 API health check failed${NC}"
fi

echo "  - PyTorch API..."
TORCH_HEALTH=$(curl -s http://localhost:8001/health)
if [ $? -eq 0 ]; then
    echo -e "  ${GREEN}✓ PyTorch API is healthy${NC}"
else
    echo -e "  ${RED}✗ PyTorch API health check failed${NC}"
fi
echo ""

# List available models
echo -e "${YELLOW}3. Listing available models...${NC}"
MODELS=$(curl -s http://localhost:8002/models)
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Models listed successfully${NC}"
    echo "$MODELS" | python3 -m json.tool 2>/dev/null || echo "$MODELS"
else
    echo -e "${RED}✗ Failed to list models${NC}"
fi
echo ""

# Test prediction with a sample image (if available)
echo -e "${YELLOW}4. Testing predictions...${NC}"

# Find a sample image from the dataset
SAMPLE_IMAGE=$(find data/imagenette2/val -name "*.JPEG" | head -1)

if [ -z "$SAMPLE_IMAGE" ]; then
    echo -e "${YELLOW}  No sample image found. Skipping prediction test.${NC}"
    echo -e "${YELLOW}  To test predictions, use:${NC}"
    echo -e "${YELLOW}    curl -X POST http://localhost:8002/predict/torch -F \"file=@path/to/image.jpg\"${NC}"
else
    echo -e "  Using sample image: $SAMPLE_IMAGE"
    echo ""
    
    echo -e "  Testing PyTorch model prediction..."
    TORCH_PRED=$(curl -s -X POST http://localhost:8002/predict/torch -F "file=@$SAMPLE_IMAGE")
    if [ $? -eq 0 ]; then
        echo -e "  ${GREEN}✓ PyTorch prediction successful${NC}"
        echo "$TORCH_PRED" | python3 -m json.tool 2>/dev/null || echo "$TORCH_PRED"
    else
        echo -e "  ${RED}✗ PyTorch prediction failed${NC}"
    fi
    echo ""
    
    echo -e "  Testing ONNX INT8 model prediction..."
    ONNX_INT8_PRED=$(curl -s -X POST http://localhost:8002/predict/onnx-int8 -F "file=@$SAMPLE_IMAGE")
    if [ $? -eq 0 ]; then
        echo -e "  ${GREEN}✓ ONNX INT8 prediction successful${NC}"
        echo "$ONNX_INT8_PRED" | python3 -m json.tool 2>/dev/null || echo "$ONNX_INT8_PRED"
    else
        echo -e "  ${RED}✗ ONNX INT8 prediction failed${NC}"
    fi
    echo ""
    
    echo -e "  Testing ONNX INT32 model prediction..."
    ONNX_INT32_PRED=$(curl -s -X POST http://localhost:8002/predict/onnx-int32 -F "file=@$SAMPLE_IMAGE")
    if [ $? -eq 0 ]; then
        echo -e "  ${GREEN}✓ ONNX INT32 prediction successful${NC}"
        echo "$ONNX_INT32_PRED" | python3 -m json.tool 2>/dev/null || echo "$ONNX_INT32_PRED"
    else
        echo -e "  ${RED}✗ ONNX INT32 prediction failed${NC}"
    fi
fi

echo ""
echo -e "${GREEN}=========================================="
echo "Testing Complete!"
echo "==========================================${NC}"
echo ""
echo "API Endpoints:"
echo "  - Coordinator: http://localhost:8002"
echo "  - ONNX INT8:   http://localhost:8000"
echo "  - ONNX INT32:  http://localhost:8003"
echo "  - PyTorch:     http://localhost:8001"
echo ""
echo "View API docs:"
echo "  - Coordinator: http://localhost:8002/docs"
echo "  - ONNX INT8:   http://localhost:8000/docs"
echo "  - PyTorch:     http://localhost:8001/docs"

