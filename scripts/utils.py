import yaml
import pandas as pd
import os
from typing import Dict

def load_config(config_path: str) -> dict:
    """
    Load a YAML configuration file.
    Args:
        config_path (str): Path to the YAML configuration file.
    Returns:
        dict: Loaded configuration as a dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_random_images_from_csv(csv_path: str, data_root: str, num_images: int = 1) -> list[str]:
    """
    Select random image paths from a CSV file.
    Args:
        csv_path (str): Path to the CSV file containing image paths.
        data_root (str): Root directory where images are stored.
        num_images (int): Number of random images to select.
    Returns:
        list[str]: List of selected image paths.
    """
    df = pd.read_csv(csv_path)
    sampled = df.sample(n=num_images)
    image_paths = [os.path.join(data_root, row["path"]) for _, row in sampled.iterrows()]
    return image_paths

def load_class_mapping(mapping_file: str) -> Dict[int, str]:
    """
    Load class index to name mapping from a txt file containing a Python dictionary-like format.

    Args:
        mapping_file (str): Path to the txt file

    Returns:
        dict[int, str]: Mapping from index to class name
    """
    mapping = {}
    with open(mapping_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("{") or line.startswith("}"):
                continue 

            key_part, value_part = line.split(":", 1)
            key = int(key_part.strip())
            value = value_part.strip().rstrip(",").strip("'\"")
            mapping[key] = value
    return mapping

def get_class_name(index: int, mapping: Dict[int, str]) -> str:
    """
    Retrieve class name given an index.

    Args:
        index (int): Class index
        mapping (dict[int, str]): Index-to-class mapping

    Returns:
        str: Class name
    """
    return mapping.get(index, "Unknown")