"""
Unit tests for image preprocessing functions.
"""
import pytest
from pathlib import Path
from PIL import Image
import torch
from scripts.image_preprocessing import preprocess_image


class TestPreprocessImage:
    """Tests for preprocess_image function."""
    
    def test_preprocess_valid_image(self, sample_image_path):
        """Test preprocessing a valid image file."""
        tensor = preprocess_image(sample_image_path)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)  # C, H, W
        assert tensor.dtype == torch.float32
    
    def test_preprocess_default_size(self, sample_image_path):
        """Test preprocessing with default size (224x224)."""
        tensor = preprocess_image(sample_image_path)
        assert tensor.shape[1] == 224
        assert tensor.shape[2] == 224
    
    def test_preprocess_custom_size(self, sample_image_path):
        """Test preprocessing with custom size."""
        tensor = preprocess_image(sample_image_path, target_height=256, target_width=256)
        assert tensor.shape[1] == 256
        assert tensor.shape[2] == 256
    
    def test_preprocess_nonexistent_file(self):
        """Test preprocessing a non-existent file."""
        with pytest.raises(FileNotFoundError):
            preprocess_image("nonexistent.jpg")
    
    def test_preprocess_different_image_sizes(self, tmp_path):
        """Test preprocessing images of different original sizes."""
        # Create images of different sizes
        sizes = [(100, 100), (500, 300), (224, 224)]
        
        for width, height in sizes:
            img = Image.new('RGB', (width, height), color='green')
            img_path = tmp_path / f"test_{width}x{height}.jpg"
            img.save(img_path)
            
            tensor = preprocess_image(str(img_path))
            # All should be resized to 224x224
            assert tensor.shape == (3, 224, 224)
    
    def test_preprocess_normalization(self, sample_image_path):
        """Test that preprocessing applies normalization."""
        tensor = preprocess_image(sample_image_path)
        
        # Check that values are in reasonable range (normalized)
        # Normalized values should be roughly in [-2, 2] range
        assert tensor.min() >= -3.0
        assert tensor.max() <= 3.0
    
    def test_preprocess_rgb_conversion(self, tmp_path):
        """Test that images are converted to RGB."""
        # Create a grayscale image
        img = Image.new('L', (224, 224), color=128)
        img_path = tmp_path / "grayscale.jpg"
        img.save(img_path)
        
        tensor = preprocess_image(str(img_path))
        # Should still be 3 channels (RGB)
        assert tensor.shape[0] == 3

