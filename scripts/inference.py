import numpy as np
import sys
from inference_utils import create_onnx_session, get_model_input_details, run_onnx_inference
from dataset import get_dataloader
from utils import load_config, get_random_images_from_csv, get_class_name, load_class_mapping


def run_inference(config_path: str) -> None:
    """Run inference on random images from a dataset using an ONNX model.
    
    Args:
        config_path (str): Path to the configuration YAML file.
    """
    config = load_config(config_path)

    model_path = config["model"]["path"]
    use_gpu = config["model"].get("use_gpu", False)

    csv_path = config["data"]["csv_path"]
    data_root = config["data"].get("root", "data/imagenette2")
    num_random_images = config["data"].get("num_random_images", 1)
    batch_size = config["data"].get("batch_size", 4)
    num_workers = config["data"].get("num_workers", 4)
    class_mapping_file = config["data"].get("class_mapping", "data/imagenette2/index_to_class.txt")

    class_mapping = load_class_mapping(class_mapping_file)
    image_paths = get_random_images_from_csv(csv_path, data_root, num_images=num_random_images)

    print("Selected random images:")
    for path in image_paths:
        print("  -", path)

    session = create_onnx_session(model_path, use_gpu=use_gpu)
    input_name, batch_dim, C, H, W = get_model_input_details(session)
    print(f"Model input: name='{input_name}', shape=(N={batch_dim or 'dynamic'}, C={C}, H={H}, W={W})")

    if batch_dim is not None:
        print(f"Model has fixed batch size {batch_dim}. Overriding DataLoader batch size.")
        batch_size = batch_dim

    dataloader = get_dataloader(
        image_paths=image_paths,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    outputs = run_onnx_inference(session, dataloader)

    for i, output in enumerate(outputs):
        pred_index = int(np.argmax(output))
        class_name = get_class_name(pred_index, class_mapping)
        print(f"Image: {image_paths[i]}")
        print(f"Predicted class index: {pred_index}")
        print(f"Predicted class name: {class_name}\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/run_inference.py <path_to_config.yml>")
        sys.exit(1)

    config_path = sys.argv[1]
    run_inference(config_path)
