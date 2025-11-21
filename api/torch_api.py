"""
FastAPI server for PyTorch model inference.
"""
import os
import sys
import numpy as np
import torch
import torchvision.models as models
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
from typing import Optional
import uvicorn

# Add scripts directory to path
scripts_path = os.path.join(os.path.dirname(__file__), '..', 'scripts')
sys.path.insert(0, scripts_path)

from utils import load_class_mapping, get_class_name

app = FastAPI(title="PyTorch Model Inference API", version="1.0.0")

# Global variables for model
model = None
device = None
class_mapping = None


def preprocess_image_for_api(image_bytes: bytes, target_height: int = 224, target_width: int = 224) -> torch.Tensor:
    """Preprocess image from bytes for PyTorch inference."""
    from torchvision import transforms
    
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
    return image_tensor


@app.on_event("startup")
async def load_model():
    """Load PyTorch model on startup."""
    global model, device, class_mapping
    
    # Get configuration from environment variables
    model_name = os.getenv("MODEL_NAME", "squeezenet1_1")
    use_gpu = os.getenv("USE_GPU", "false").lower() == "true"
    class_mapping_file = os.getenv("CLASS_MAPPING", "data/imagenette2/index_to_class.txt")
    
    print(f"Loading PyTorch model: {model_name}")
    print(f"Using GPU: {use_gpu}")
    
    # Set device
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    if model_name == "squeezenet1_1":
        model = models.squeezenet1_1(pretrained=True)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    model.eval()
    model.to(device)
    
    # Load class mapping
    if os.path.exists(class_mapping_file):
        class_mapping = load_class_mapping(class_mapping_file)
    else:
        print(f"Warning: Class mapping file not found: {class_mapping_file}")
        class_mapping = {}
    
    print("Model loaded successfully!")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "PyTorch Model Inference API",
        "model_loaded": model is not None,
        "device": str(device) if device else None
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else None
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict class for an uploaded image.
    
    Args:
        file: Image file (JPEG, PNG, etc.)
    
    Returns:
        JSON response with prediction results
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Preprocess image
        image_tensor = preprocess_image_for_api(image_bytes)
        
        # Add batch dimension if needed
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Move to device
        image_tensor = image_tensor.to(device)
        
        # Run inference
        with torch.no_grad():
            output = model(image_tensor)
            output_np = output.cpu().numpy()
        
        # Get prediction
        pred_index = int(np.argmax(output_np[0]))
        confidence = float(np.max(output_np[0]))
        class_name = get_class_name(pred_index, class_mapping) if class_mapping else f"Class_{pred_index}"
        
        # Get top 5 predictions
        top5_indices = np.argsort(output_np[0])[-5:][::-1]
        top5_predictions = [
            {
                "class_index": int(idx),
                "class_name": get_class_name(int(idx), class_mapping) if class_mapping else f"Class_{idx}",
                "confidence": float(output_np[0][idx])
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
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        
        for file in files:
            # Read image bytes
            image_bytes = await file.read()
            
            # Preprocess image
            image_tensor = preprocess_image_for_api(image_bytes)
            
            # Add batch dimension if needed
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            # Move to device
            image_tensor = image_tensor.to(device)
            
            # Run inference
            with torch.no_grad():
                output = model(image_tensor)
                output_np = output.cpu().numpy()
            
            # Get prediction
            pred_index = int(np.argmax(output_np[0]))
            confidence = float(np.max(output_np[0]))
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
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)

