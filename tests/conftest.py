"""
Pytest configuration and shared fixtures.
"""
import pytest
import os
import sys
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np
import io

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

@pytest.fixture
def sample_image_bytes():
    """Create a sample image in memory as bytes."""
    # Create a simple test image
    img = Image.new('RGB', (224, 224), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes.read()

@pytest.fixture
def sample_image_path(tmp_path):
    """Create a temporary image file."""
    img = Image.new('RGB', (224, 224), color='blue')
    img_path = tmp_path / "test_image.jpg"
    img.save(img_path)
    return str(img_path)

@pytest.fixture
def sample_class_mapping_file(tmp_path):
    """Create a temporary class mapping file."""
    mapping_file = tmp_path / "class_mapping.txt"
    content = """{
0: 'tench',
1: 'English springer',
2: 'cassette player',
3: 'chain saw',
4: 'church',
5: 'French horn',
6: 'garbage truck',
7: 'gas pump',
8: 'golf ball',
9: 'parachute'
}"""
    mapping_file.write_text(content)
    return str(mapping_file)

@pytest.fixture
def sample_config_file(tmp_path):
    """Create a temporary YAML config file."""
    config_file = tmp_path / "test_config.yml"
    content = """model:
  name: squeezenet1_1
  path: models/squeezenet.onnx
  use_gpu: false
api:
  port: 8000
  host: 0.0.0.0
"""
    config_file.write_text(content)
    return str(config_file)

@pytest.fixture
def sample_csv_file(tmp_path):
    """Create a temporary CSV file with image paths."""
    csv_file = tmp_path / "test_images.csv"
    content = """path,label
train/n01440764/ILSVRC2012_val_00000293.JPEG,0
train/n02102040/ILSVRC2012_val_00002138.JPEG,1
train/n02979186/ILSVRC2012_val_00003028.JPEG,2"""
    csv_file.write_text(content)
    return str(csv_file)

@pytest.fixture
def mock_onnx_session(monkeypatch):
    """Mock ONNX session for testing."""
    class MockSession:
        def __init__(self, *args, **kwargs):
            pass
        
        def get_inputs(self):
            class MockInput:
                def __init__(self):
                    self.name = "input"
                    self.shape = [1, 3, 224, 224]
            return [MockInput()]
        
        def run(self, output_names, input_dict):
            # Return mock output (1000 classes)
            return [np.random.randn(1, 1000).astype(np.float32)]
    
    import onnxruntime as ort
    monkeypatch.setattr(ort, "InferenceSession", MockSession)
    return MockSession

