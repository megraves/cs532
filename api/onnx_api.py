"""
FastAPI server for ONNX model inference.
Supports both INT8 and INT32 ONNX models.
"""
import os
import sys
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
from typing import Optional
import uvicorn

# Add scripts directory to path
scripts_path = os.path.join(os.path.dirname(__file__), '..', 'scripts')
sys.path.insert(0, scripts_path)

from inference_utils import create_onnx_session, get_model_input_details, run_onnx_inference
from utils import load_class_mapping, get_class_name

app = FastAPI(title="ONNX Model Inference API", version="1.0.0")

# Global variables for model and session
session = None
input_name = None
model_format = None
C_model = None
H_model = None
W_model = None
class_mapping = None


def preprocess_image_for_api(image_bytes: bytes, target_height: int = 224, target_width: int = 224) -> np.ndarray:
    """Preprocess image from bytes for ONNX inference."""
    from torchvision import transforms
    from PIL import Image
    
    # Load image from bytes
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Apply transforms
    transform = transforms.Compose([
        transforms.Resize((target_height, target_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image)
    # Convert to numpy array (NCHW format)
    image_np = image_tensor.numpy().astype(np.float32)
    return image_np


@app.on_event("startup")
async def load_model():
    """Load ONNX model on startup."""
    global session, input_name, model_format, C_model, H_model, W_model, class_mapping
    
    # Get model path from environment variable
    model_path = os.getenv("MODEL_PATH", "models/squeezenet.onnx")
    use_gpu = os.getenv("USE_GPU", "false").lower() == "true"
    class_mapping_file = os.getenv("CLASS_MAPPING", "data/imagenette2/index_to_class.txt")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading ONNX model from: {model_path}")
    print(f"Using GPU: {use_gpu}")
    
    # Create ONNX session
    session = create_onnx_session(model_path, use_gpu=use_gpu)
    input_name, batch_dim, C_model, H_model, W_model = get_model_input_details(session)
    
    # Determine model format
    model_format = "NHWC" if W_model == 3 or H_model == 3 else "NCHW"
    
    # Load class mapping
    if os.path.exists(class_mapping_file):
        class_mapping = load_class_mapping(class_mapping_file)
    else:
        print(f"Warning: Class mapping file not found: {class_mapping_file}")
        class_mapping = {}
    
    print(f"Model loaded successfully!")
    print(f"Input: name='{input_name}', shape=(N={batch_dim or 'dynamic'}, C={C_model}, H={H_model}, W={W_model})")
    print(f"Format: {model_format}")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "ONNX Model Inference API",
        "model_loaded": session is not None
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": session is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict class for an uploaded image.
    
    Args:
        file: Image file (JPEG, PNG, etc.)
    
    Returns:
        JSON response with prediction results
    """
    if session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Preprocess image
        image_np = preprocess_image_for_api(image_bytes)
        
        # Add batch dimension if needed
        if image_np.ndim == 3:
            image_np = np.expand_dims(image_np, axis=0)
        
        # Ensure contiguous array
        batch_np = np.ascontiguousarray(image_np, dtype=np.float32)
        
        # Handle format conversion
        N, D1, D2, D3 = batch_np.shape
        if model_format == "NCHW":
            if (D1, D2, D3) != (C_model, H_model, W_model):
                # Convert NHWC -> NCHW
                batch_np = batch_np.transpose(0, 3, 1, 2)
        elif model_format == "NHWC":
            if (D1, D2, D3) != (H_model, W_model, C_model):
                # Convert NCHW -> NHWC
                batch_np = batch_np.transpose(0, 2, 3, 1)
        
        # Run inference
        output = run_onnx_inference(session, batch_np, input_name=input_name)
        
        # Get prediction
        pred_index = int(np.argmax(output[0]))
        confidence = float(np.max(output[0]))
        class_name = get_class_name(pred_index, class_mapping) if class_mapping else f"Class_{pred_index}"
        
        # Get top 5 predictions
        top5_indices = np.argsort(output[0])[-5:][::-1]
        top5_predictions = [
            {
                "class_index": int(idx),
                "class_name": get_class_name(int(idx), class_mapping) if class_mapping else f"Class_{idx}",
                "confidence": float(output[0][idx])
            }
            for idx in top5_indices
        ]
        
        return JSONResponse({
            "predicted_class_index": pred_index,
            "predicted_class_name": class_name,
            "confidence": confidence,
            "top5_predictions": top5_predictions
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Predict classes for multiple uploaded images.
    
    Args:
        files: List of image files
    
    Returns:
        JSON response with predictions for each image
    """
    if session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        
        for file in files:
            # Read image bytes
            image_bytes = await file.read()
            
            # Preprocess image
            image_np = preprocess_image_for_api(image_bytes)
            
            # Add batch dimension if needed
            if image_np.ndim == 3:
                image_np = np.expand_dims(image_np, axis=0)
            
            # Ensure contiguous array
            batch_np = np.ascontiguousarray(image_np, dtype=np.float32)
            
            # Handle format conversion
            N, D1, D2, D3 = batch_np.shape
            if model_format == "NCHW":
                if (D1, D2, D3) != (C_model, H_model, W_model):
                    batch_np = batch_np.transpose(0, 3, 1, 2)
            elif model_format == "NHWC":
                if (D1, D2, D3) != (H_model, W_model, C_model):
                    batch_np = batch_np.transpose(0, 2, 3, 1)
            
            # Run inference
            output = run_onnx_inference(session, batch_np, input_name=input_name)
            
            # Get prediction
            pred_index = int(np.argmax(output[0]))
            confidence = float(np.max(output[0]))
            class_name = get_class_name(pred_index, class_mapping) if class_mapping else f"Class_{pred_index}"
            
            results.append({
                "filename": file.filename,
                "predicted_class_index": pred_index,
                "predicted_class_name": class_name,
                "confidence": confidence
            })
        
        return JSONResponse({"predictions": results})
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

