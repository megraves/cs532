"""
Tests for PyTorch API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import torch
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.torch_api import app


class TestTorchAPIHealth:
    """Tests for health check endpoints."""
    
    @patch('api.torch_api.model', None)
    def test_root_endpoint_no_model(self):
        """Test root endpoint when model is not loaded."""
        client = TestClient(app)
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "PyTorch Model Inference API"
        assert data["model_loaded"] is False
    
    @patch('api.torch_api.model', MagicMock())
    @patch('api.torch_api.device', torch.device('cpu'))
    def test_root_endpoint_with_model(self):
        """Test root endpoint when model is loaded."""
        client = TestClient(app)
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_loaded"] is True
    
    @patch('api.torch_api.model', None)
    def test_health_endpoint_no_model(self):
        """Test health endpoint when model is not loaded."""
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is False


class TestTorchAPIPredict:
    """Tests for prediction endpoints."""
    
    @patch('api.torch_api.model', None)
    def test_predict_no_model_loaded(self, sample_image_bytes):
        """Test predict endpoint when model is not loaded."""
        client = TestClient(app)
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]
    
    @patch('api.torch_api.preprocess_image_for_api')
    @patch('api.torch_api.model')
    @patch('api.torch_api.device', torch.device('cpu'))
    @patch('api.torch_api.class_mapping', {0: "tench", 1: "English springer"})
    def test_predict_single_image(self, mock_model, mock_preprocess, sample_image_bytes):
        """Test single image prediction."""
        # Setup mocks
        mock_preprocess.return_value = torch.randn(3, 224, 224)
        
        # Mock model output
        mock_output = torch.randn(1, 1000)
        mock_output[0, 0] = 10.0  # Make class 0 the highest
        mock_model.return_value = mock_output
        mock_model.eval = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        
        client = TestClient(app)
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "predicted_class_index" in data
        assert "predicted_class_name" in data
        assert "confidence" in data
        assert "top5_predictions" in data
        assert isinstance(data["top5_predictions"], list)
        assert len(data["top5_predictions"]) == 5
    
    @patch('api.torch_api.preprocess_image_for_api')
    @patch('api.torch_api.model')
    @patch('api.torch_api.device', torch.device('cpu'))
    def test_predict_invalid_image(self, mock_model, mock_preprocess):
        """Test predict with invalid image data."""
        mock_preprocess.side_effect = Exception("Invalid image")
        
        client = TestClient(app)
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", b"invalid image data", "image/jpeg")}
        )
        
        assert response.status_code == 500
        assert "error" in response.json()["detail"].lower()
    
    @patch('api.torch_api.model')
    def test_predict_missing_file(self, mock_model):
        """Test predict endpoint without file."""
        client = TestClient(app)
        response = client.post("/predict")
        
        assert response.status_code == 422  # Validation error


class TestTorchAPIBatchPredict:
    """Tests for batch prediction endpoints."""
    
    @patch('api.torch_api.model', None)
    def test_batch_predict_no_model_loaded(self, sample_image_bytes):
        """Test batch predict when model is not loaded."""
        client = TestClient(app)
        response = client.post(
            "/predict/batch",
            files=[
                ("files", ("test1.jpg", sample_image_bytes, "image/jpeg")),
                ("files", ("test2.jpg", sample_image_bytes, "image/jpeg"))
            ]
        )
        
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]
    
    @patch('api.torch_api.preprocess_image_for_api')
    @patch('api.torch_api.model')
    @patch('api.torch_api.device', torch.device('cpu'))
    @patch('api.torch_api.class_mapping', {0: "tench"})
    def test_batch_predict_multiple_images(self, mock_model, mock_preprocess, sample_image_bytes):
        """Test batch prediction with multiple images."""
        # Setup mocks
        mock_preprocess.return_value = torch.randn(3, 224, 224)
        mock_output = torch.randn(1, 1000)
        mock_output[0, 0] = 10.0
        mock_model.return_value = mock_output
        mock_model.eval = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        
        client = TestClient(app)
        response = client.post(
            "/predict/batch",
            files=[
                ("files", ("test1.jpg", sample_image_bytes, "image/jpeg")),
                ("files", ("test2.jpg", sample_image_bytes, "image/jpeg"))
            ]
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert isinstance(data["predictions"], list)
        assert len(data["predictions"]) == 2
        assert "filename" in data["predictions"][0]
        assert "predicted_class_index" in data["predictions"][0]

