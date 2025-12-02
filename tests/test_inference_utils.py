"""
Unit tests for inference utility functions.
"""
import pytest
import numpy as np
import onnxruntime as ort
from pathlib import Path
from scripts.inference_utils import (
    create_onnx_session,
    get_model_input_details,
    run_onnx_inference
)


class TestCreateOnnxSession:
    """Tests for create_onnx_session function."""
    
    def test_create_session_cpu(self, tmp_path):
        """Test creating an ONNX session with CPU provider."""
        # This test requires a real ONNX model file
        # For now, we'll skip if model doesn't exist
        model_path = Path("models/squeezenet.onnx")
        
        if not model_path.exists():
            pytest.skip("Model file not found")
        
        session = create_onnx_session(str(model_path), use_gpu=False)
        
        assert isinstance(session, ort.InferenceSession)
        # Check that CPU provider is available
        providers = session.get_providers()
        assert 'CPUExecutionProvider' in providers
    
    def test_create_session_nonexistent_file(self):
        """Test creating session with non-existent model file."""
        with pytest.raises(Exception):  # Should raise FileNotFoundError or similar
            create_onnx_session("nonexistent.onnx", use_gpu=False)


class TestGetModelInputDetails:
    """Tests for get_model_input_details function."""
    
    def test_get_input_details(self, tmp_path):
        """Test getting input details from a model."""
        model_path = Path("models/squeezenet.onnx")
        
        if not model_path.exists():
            pytest.skip("Model file not found")
        
        session = create_onnx_session(str(model_path), use_gpu=False)
        input_name, batch_dim, C, H, W = get_model_input_details(session)
        
        assert isinstance(input_name, str)
        assert isinstance(C, int)
        assert isinstance(H, int)
        assert isinstance(W, int)
        assert C == 3  # RGB channels
        assert H > 0
        assert W > 0
        # Batch dimension can be None (dynamic) or an int
        assert batch_dim is None or isinstance(batch_dim, int)


class TestRunOnnxInference:
    """Tests for run_onnx_inference function."""
    
    def test_run_inference_numpy_array(self, tmp_path):
        """Test running inference with numpy array."""
        model_path = Path("models/squeezenet.onnx")
        
        if not model_path.exists():
            pytest.skip("Model file not found")
        
        session = create_onnx_session(str(model_path), use_gpu=False)
        input_name, _, C, H, W = get_model_input_details(session)
        
        # Create dummy input batch
        batch = np.random.randn(1, C, H, W).astype(np.float32)
        
        output = run_onnx_inference(session, batch, input_name=input_name)
        
        assert isinstance(output, np.ndarray)
        assert output.dtype == np.float32
        assert len(output.shape) == 2  # [batch_size, num_classes]
        assert output.shape[0] == 1  # Batch size
    
    def test_run_inference_without_input_name(self, tmp_path):
        """Test running inference without specifying input name."""
        model_path = Path("models/squeezenet.onnx")
        
        if not model_path.exists():
            pytest.skip("Model file not found")
        
        session = create_onnx_session(str(model_path), use_gpu=False)
        _, _, C, H, W = get_model_input_details(session)
        
        batch = np.random.randn(1, C, H, W).astype(np.float32)
        
        output = run_onnx_inference(session, batch, input_name=None)
        
        assert isinstance(output, np.ndarray)
        assert output.shape[0] == 1
    
    def test_run_inference_batch_size_2(self, tmp_path):
        """Test running inference with batch size 2."""
        model_path = Path("models/squeezenet.onnx")
        
        if not model_path.exists():
            pytest.skip("Model file not found")
        
        session = create_onnx_session(str(model_path), use_gpu=False)
        input_name, batch_dim, C, H, W = get_model_input_details(session)
        
        # Skip if model has fixed batch size of 1
        if batch_dim == 1:
            pytest.skip("Model has fixed batch size of 1, cannot test batch size 2")
        
        batch = np.random.randn(2, C, H, W).astype(np.float32)
        
        try:
            output = run_onnx_inference(session, batch, input_name=input_name)
            assert output.shape[0] == 2  # Batch size
        except Exception as e:
            # Some models may not support dynamic batch sizes
            if "invalid dimensions" in str(e).lower() or "batch" in str(e).lower():
                pytest.skip(f"Model does not support batch size 2: {e}")
            raise
    
    def test_run_inference_contiguous_array(self, tmp_path):
        """Test that function handles non-contiguous arrays."""
        model_path = Path("models/squeezenet.onnx")
        
        if not model_path.exists():
            pytest.skip("Model file not found")
        
        session = create_onnx_session(str(model_path), use_gpu=False)
        input_name, _, C, H, W = get_model_input_details(session)
        
        # Create non-contiguous array (transpose makes it non-contiguous)
        batch = np.random.randn(H, W, C).astype(np.float32)
        batch = batch.transpose(2, 0, 1)  # Make it non-contiguous
        batch = np.expand_dims(batch, axis=0)
        
        # Should still work (function makes it contiguous)
        output = run_onnx_inference(session, batch, input_name=input_name)
        assert isinstance(output, np.ndarray)

