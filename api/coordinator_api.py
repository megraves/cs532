"""
Coordinator API that routes requests to appropriate model containers.
"""
import os
import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
import uvicorn
from typing import Optional

app = FastAPI(title="Model Coordinator API", version="1.0.0")

# Model service URLs (can be overridden by environment variables)
ONNX_INT8_URL = os.getenv("ONNX_INT8_URL", "http://onnx-int8-api:8000")
ONNX_INT32_URL = os.getenv("ONNX_INT32_URL", "http://onnx-int32-api:8000")
TORCH_URL = os.getenv("TORCH_URL", "http://torch-api:8001")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Model Coordinator API",
        "available_models": ["onnx-int8", "onnx-int32", "torch"]
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/models")
async def list_models():
    """List available models."""
    return {
        "models": [
            {
                "id": "onnx-int8",
                "name": "SqueezeNet ONNX INT8",
                "description": "Quantized INT8 ONNX model",
                "endpoint": "/predict/onnx-int8"
            },
            {
                "id": "onnx-int32",
                "name": "SqueezeNet ONNX INT32",
                "description": "Standard INT32 ONNX model",
                "endpoint": "/predict/onnx-int32"
            },
            {
                "id": "torch",
                "name": "SqueezeNet PyTorch",
                "description": "Native PyTorch model",
                "endpoint": "/predict/torch"
            }
        ]
    }


@app.get("/health/{model_id}")
async def check_model_health(model_id: str):
    """Check health of a specific model service."""
    url_map = {
        "onnx-int8": ONNX_INT8_URL,
        "onnx-int32": ONNX_INT32_URL,
        "torch": TORCH_URL
    }
    
    if model_id not in url_map:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{url_map[model_id]}/health")
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model service unavailable: {str(e)}")


@app.post("/predict/{model_id}")
async def predict(
    model_id: str,
    file: UploadFile = File(...)
):
    """
    Route prediction request to the specified model.
    
    Args:
        model_id: Model identifier (onnx-int8, onnx-int32, or torch)
        file: Image file to predict
    
    Returns:
        JSON response with prediction results
    """
    url_map = {
        "onnx-int8": ONNX_INT8_URL,
        "onnx-int32": ONNX_INT32_URL,
        "torch": TORCH_URL
    }
    
    if model_id not in url_map:
        raise HTTPException(
            status_code=404,
            detail=f"Model {model_id} not found. Available models: {list(url_map.keys())}"
        )
    
    try:
        # Read file content
        file_content = await file.read()
        files = {"file": (file.filename, file_content, file.content_type)}
        
        # Forward request to model service
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{url_map[model_id]}/predict",
                files=files
            )
            response.raise_for_status()
            return response.json()
            
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error routing request: {str(e)}")


@app.post("/predict/{model_id}/batch")
async def predict_batch(
    model_id: str,
    files: list[UploadFile] = File(...)
):
    """
    Route batch prediction request to the specified model.
    
    Args:
        model_id: Model identifier (onnx-int8, onnx-int32, or torch)
        files: List of image files to predict
    
    Returns:
        JSON response with batch prediction results
    """
    url_map = {
        "onnx-int8": ONNX_INT8_URL,
        "onnx-int32": ONNX_INT32_URL,
        "torch": TORCH_URL
    }
    
    if model_id not in url_map:
        raise HTTPException(
            status_code=404,
            detail=f"Model {model_id} not found. Available models: {list(url_map.keys())}"
        )
    
    try:
        # Prepare files for forwarding
        file_list = []
        for file in files:
            file_content = await file.read()
            file_list.append(("files", (file.filename, file_content, file.content_type)))
        
        # Forward request to model service
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{url_map[model_id]}/predict/batch",
                files=file_list
            )
            response.raise_for_status()
            return response.json()
            
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error routing request: {str(e)}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)

