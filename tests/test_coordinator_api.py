"""
Tests for Coordinator API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import httpx
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.coordinator_api import app


class TestCoordinatorAPIHealth:
    """Tests for health check endpoints."""
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        client = TestClient(app)
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "Model Coordinator API"
        assert "available_models" in data
        assert isinstance(data["available_models"], list)
    
    def test_health_endpoint(self):
        """Test health endpoint."""
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestCoordinatorAPIListModels:
    """Tests for model listing endpoints."""
    
    def test_list_models(self):
        """Test listing available models."""
        client = TestClient(app)
        response = client.get("/models")
        
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)
        assert len(data["models"]) == 3
        
        model_ids = [m["id"] for m in data["models"]]
        assert "onnx-int8" in model_ids
        assert "onnx-int32" in model_ids
        assert "torch" in model_ids
        
        # Check structure of each model entry
        for model in data["models"]:
            assert "id" in model
            assert "name" in model
            assert "description" in model
            assert "endpoint" in model


class TestCoordinatorAPICheckModelHealth:
    """Tests for model health check endpoints."""
    
    @patch('httpx.AsyncClient')
    def test_check_model_health_success(self, mock_client_class):
        """Test checking health of a specific model."""
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {"status": "healthy", "model_loaded": True}
        mock_response.status_code = 200
        
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client
        
        client = TestClient(app)
        response = client.get("/health/torch")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_check_model_health_invalid_model(self):
        """Test checking health of invalid model ID."""
        client = TestClient(app)
        response = client.get("/health/invalid-model")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    @patch('httpx.AsyncClient')
    def test_check_model_health_service_unavailable(self, mock_client_class):
        """Test checking health when model service is unavailable."""
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(side_effect=httpx.RequestError("Connection failed"))
        mock_client_class.return_value = mock_client
        
        client = TestClient(app)
        response = client.get("/health/torch")
        
        assert response.status_code == 503
        assert "unavailable" in response.json()["detail"].lower()


class TestCoordinatorAPIPredict:
    """Tests for prediction routing endpoints."""
    
    def test_predict_invalid_model_id(self, sample_image_bytes):
        """Test predict with invalid model ID."""
        client = TestClient(app)
        response = client.post(
            "/predict/invalid-model",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    @patch('httpx.AsyncClient')
    def test_predict_success(self, mock_client_class, sample_image_bytes):
        """Test successful prediction routing."""
        # Mock successful response from model service
        mock_response = Mock()
        mock_response.json.return_value = {
            "predicted_class_index": 0,
            "predicted_class_name": "tench",
            "confidence": 0.95,
            "top5_predictions": []
        }
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client
        
        client = TestClient(app)
        response = client.post(
            "/predict/torch",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "predicted_class_index" in data
        assert "predicted_class_name" in data
    
    @patch('httpx.AsyncClient')
    def test_predict_model_service_error(self, mock_client_class, sample_image_bytes):
        """Test prediction when model service returns error."""
        mock_response = Mock()
        mock_response.status_code = 500
        error = httpx.HTTPStatusError(
            "Server Error",
            request=Mock(),
            response=mock_response
        )
        mock_response.raise_for_status.side_effect = error
        
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client
        
        client = TestClient(app)
        response = client.post(
            "/predict/torch",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 500
    
    def test_predict_missing_file(self):
        """Test predict endpoint without file."""
        client = TestClient(app)
        response = client.post("/predict/torch")
        
        assert response.status_code == 422  # Validation error


class TestCoordinatorAPIBatchPredict:
    """Tests for batch prediction routing endpoints."""
    
    def test_batch_predict_invalid_model_id(self, sample_image_bytes):
        """Test batch predict with invalid model ID."""
        client = TestClient(app)
        response = client.post(
            "/predict/invalid-model/batch",
            files=[
                ("files", ("test1.jpg", sample_image_bytes, "image/jpeg")),
                ("files", ("test2.jpg", sample_image_bytes, "image/jpeg"))
            ]
        )
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    @patch('httpx.AsyncClient')
    def test_batch_predict_success(self, mock_client_class, sample_image_bytes):
        """Test successful batch prediction routing."""
        # Mock successful response from model service
        mock_response = Mock()
        mock_response.json.return_value = {
            "predictions": [
                {"filename": "test1.jpg", "predicted_class_index": 0, "confidence": 0.95},
                {"filename": "test2.jpg", "predicted_class_index": 1, "confidence": 0.90}
            ]
        }
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client
        
        client = TestClient(app)
        response = client.post(
            "/predict/torch/batch",
            files=[
                ("files", ("test1.jpg", sample_image_bytes, "image/jpeg")),
                ("files", ("test2.jpg", sample_image_bytes, "image/jpeg"))
            ]
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2
    
    @patch('httpx.AsyncClient')
    def test_batch_predict_model_service_error(self, mock_client_class, sample_image_bytes):
        """Test batch prediction when model service returns error."""
        mock_response = Mock()
        mock_response.status_code = 503
        error = httpx.HTTPStatusError(
            "Service Unavailable",
            request=Mock(),
            response=mock_response
        )
        mock_response.raise_for_status.side_effect = error
        
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client
        
        client = TestClient(app)
        response = client.post(
            "/predict/torch/batch",
            files=[
                ("files", ("test1.jpg", sample_image_bytes, "image/jpeg"))
            ]
        )
        
        assert response.status_code == 503

