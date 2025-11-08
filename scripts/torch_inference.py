import torch
import torchvision.models as models
import numpy as np
import sys
from dataset import get_dataloader
from utils import load_config, get_random_images_from_csv, get_class_name, load_class_mapping


def run_inference(config_path: str) -> None:
    """
    Run inference on random images using a pretrained PyTorch model.

    Args:
        config_path (str): Path to the configuration YAML file.
    """
    # Load config
    config = load_config(config_path)

    # Device
    use_gpu = config["model"].get("use_gpu", False)
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    # Model
    model_name = config["model"].get("name", "squeezenet1_1")
    if model_name == "squeezenet1_1":
        model = models.squeezenet1_1(pretrained=True)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    model.eval()
    model.to(device)

    # Data paths
    csv_path = config["data"]["csv_path"]
    data_root = config["data"].get("root", "data/imagenette2")
    num_random_images = config["data"].get("num_random_images", 1)
    batch_size = config["data"].get("batch_size", 4)
    num_workers = config["data"].get("num_workers", 4)
    target_height = config["data"].get("target_height", 224)
    target_width = config["data"].get("target_width", 224)
    class_mapping_file = config["data"].get("class_mapping", "data/imagenette2/index_to_class.txt")

    # Class mapping
    class_mapping = load_class_mapping(class_mapping_file)

    # Select random images
    image_paths = get_random_images_from_csv(csv_path, data_root, num_images=num_random_images)
    print("Selected random images:")
    for path in image_paths:
        print("  -", path)

    # Create dataloader using your existing dataset utilities
    dataloader = get_dataloader(
        image_paths=image_paths,
        batch_size=batch_size,
        num_workers=num_workers,
        target_height=target_height,
        target_width=target_width
    )

    # Run inference
    outputs = []
    for batch in dataloader:
        batch = batch.to(device)

        with torch.no_grad():
            preds = model(batch)
            preds = preds.cpu().numpy()
            outputs.extend(preds)

    # Display results
    for i, output in enumerate(outputs):
        pred_index = int(np.argmax(output))
        class_name = get_class_name(pred_index, class_mapping)
        print(f"Image: {image_paths[i]}")
        print(f"Predicted class index: {pred_index}")
        print(f"Predicted class name: {class_name}\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_inference_torch.py <path_to_config.yml>")
        sys.exit(1)

    config_path = sys.argv[1]
    run_inference(config_path)
