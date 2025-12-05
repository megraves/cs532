"""
Unit tests for utility functions in scripts/utils.py
"""
import pytest
import tempfile
import os
from pathlib import Path
import pandas as pd
from scripts.utils import load_config, get_random_images_from_csv, load_class_mapping, get_class_name


class TestLoadConfig:
    """Tests for load_config function."""
    
    def test_load_valid_yaml(self, sample_config_file):
        """Test loading a valid YAML config file."""
        config = load_config(sample_config_file)
        assert isinstance(config, dict)
        assert "model" in config
        assert "api" in config
        assert config["model"]["name"] == "squeezenet1_1"
    
    def test_load_nonexistent_file(self):
        """Test loading a non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yml")
    
    def test_load_invalid_yaml(self, tmp_path):
        """Test loading an invalid YAML file."""
        invalid_file = tmp_path / "invalid.yml"
        invalid_file.write_text("invalid: yaml: content: [")
        
        with pytest.raises(Exception):  # Should raise YAML parsing error
            load_config(str(invalid_file))


class TestGetRandomImagesFromCSV:
    """Tests for get_random_images_from_csv function."""
    
    def test_get_single_image(self, sample_csv_file, tmp_path):
        """Test getting a single random image."""
        data_root = str(tmp_path)
        images = get_random_images_from_csv(sample_csv_file, data_root, num_images=1)
        
        assert len(images) == 1
        assert isinstance(images[0], str)
        assert images[0].startswith(data_root)
    
    def test_get_multiple_images(self, sample_csv_file, tmp_path):
        """Test getting multiple random images."""
        data_root = str(tmp_path)
        images = get_random_images_from_csv(sample_csv_file, data_root, num_images=2)
        
        assert len(images) == 2
        assert all(isinstance(img, str) for img in images)
    
    def test_get_more_than_available(self, sample_csv_file, tmp_path):
        """Test getting more images than available in CSV."""
        data_root = str(tmp_path)
        # CSV has 3 rows, try to get 5
        with pytest.raises(ValueError):
            get_random_images_from_csv(sample_csv_file, data_root, num_images=5)
    
    def test_nonexistent_csv(self):
        """Test with non-existent CSV file."""
        with pytest.raises(FileNotFoundError):
            get_random_images_from_csv("nonexistent.csv", "/tmp", num_images=1)


class TestLoadClassMapping:
    """Tests for load_class_mapping function."""
    
    def test_load_valid_mapping(self, sample_class_mapping_file):
        """Test loading a valid class mapping file."""
        mapping = load_class_mapping(sample_class_mapping_file)
        
        assert isinstance(mapping, dict)
        assert len(mapping) > 0
        assert 0 in mapping
        assert mapping[0] == "tench"
        assert isinstance(mapping[0], str)
    
    def test_load_nonexistent_file(self):
        """Test loading a non-existent mapping file."""
        with pytest.raises(FileNotFoundError):
            load_class_mapping("nonexistent.txt")
    
    def test_load_empty_file(self, tmp_path):
        """Test loading an empty mapping file."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")
        
        mapping = load_class_mapping(str(empty_file))
        assert mapping == {}
    
    def test_load_file_with_git_lfs_pointer(self, tmp_path):
        """Test loading a file that contains Git LFS pointer (should skip)."""
        lfs_file = tmp_path / "lfs.txt"
        content = """version https://git-lfs.github.com/spec/v1
oid sha256:abc123
size 12345
{
0: 'tench',
1: 'English springer'
}"""
        lfs_file.write_text(content)
        
        mapping = load_class_mapping(str(lfs_file))
        # Should skip LFS pointer lines and still parse the mapping
        assert len(mapping) >= 2
    
    def test_load_file_with_invalid_lines(self, tmp_path):
        """Test loading a file with some invalid lines."""
        invalid_file = tmp_path / "invalid.txt"
        content = """{
0: 'tench',
invalid line without colon
1: 'English springer',
another invalid line
2: 'cassette player'
}"""
        invalid_file.write_text(content)
        
        mapping = load_class_mapping(str(invalid_file))
        # Should parse valid lines and skip invalid ones
        assert 0 in mapping
        assert 1 in mapping
        assert 2 in mapping


class TestGetClassName:
    """Tests for get_class_name function."""
    
    def test_get_existing_class(self, sample_class_mapping_file):
        """Test getting name for an existing class index."""
        mapping = load_class_mapping(sample_class_mapping_file)
        class_name = get_class_name(0, mapping)
        
        assert class_name == "tench"
        assert isinstance(class_name, str)
    
    def test_get_nonexistent_class(self, sample_class_mapping_file):
        """Test getting name for a non-existent class index."""
        mapping = load_class_mapping(sample_class_mapping_file)
        class_name = get_class_name(999, mapping)
        
        assert class_name == "Unknown"
    
    def test_get_class_with_empty_mapping(self):
        """Test getting class name with empty mapping."""
        class_name = get_class_name(0, {})
        assert class_name == "Unknown"
    
    def test_get_class_with_none_mapping(self):
        """Test getting class name with None mapping."""
        # Should handle None gracefully or raise AttributeError
        with pytest.raises((AttributeError, TypeError)):
            get_class_name(0, None)

