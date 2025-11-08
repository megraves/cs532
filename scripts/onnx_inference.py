import numpy as np
import sys
from inference_utils import create_onnx_session, get_model_input_details, run_onnx_inference
from dataset import get_dataloader
from utils import load_config, get_random_images_from_csv, get_class_name, load_class_mapping


def run_inference(config_path: str) -> None:
    """Run inference on random images from a dataset using an ONNX model."""
    config = load_config(config_path)

    # Load model config
    model_path = config["model"]["path"]
    use_gpu = config["model"].get("use_gpu", False)

    # Load data config
    csv_path = config["data"]["csv_path"]
    data_root = config["data"].get("root", "data/imagenette2")
    num_random_images = config["data"].get("num_random_images", 1)
    batch_size = config["data"].get("batch_size", 4)
    num_workers = config["data"].get("num_workers", 4)
    class_mapping_file = config["data"].get("class_mapping", "data/imagenette2/index_to_class.txt")

    # Load class mapping and select random images
    class_mapping = load_class_mapping(class_mapping_file)
    image_paths = get_random_images_from_csv(csv_path, data_root, num_images=num_random_images)

    print("Selected random images:")
    for path in image_paths:
        print("  -", path)

    # Create ONNX session
    session = create_onnx_session(model_path, use_gpu=use_gpu)
    input_name, batch_dim, C_model, H_model, W_model = get_model_input_details(session)
    print(f"Model input: name='{input_name}', shape=(N={batch_dim or 'dynamic'}, C={C_model}, H={H_model}, W={W_model})")

    # Determine model format: NHWC or NCHW
    # If C is last dimension, model expects NHWC
    model_format = "NHWC" if W_model == 3 or H_model == 3 else "NCHW"

    # Override batch size if model has fixed batch
    if batch_dim is not None:
        print(f"Model has fixed batch size {batch_dim}. Overriding DataLoader batch size.")
        batch_size = batch_dim

    # Prepare dataloader
    dataloader = get_dataloader(
        image_paths=image_paths,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    outputs = []
    for batch in dataloader:
        # Ensure batch is 4D
        if batch.ndim == 3:
            batch = np.expand_dims(batch, axis=0)

        batch_np = np.ascontiguousarray(batch, dtype=np.float32)

        # Dynamically reshape batch to match model input
        N, D1, D2, D3 = batch_np.shape
        if model_format == "NCHW":
            if (D1, D2, D3) != (C_model, H_model, W_model):
                # Convert NHWC -> NCHW
                batch_np = batch_np.transpose(0, 3, 1, 2)
                print(f"Transposed batch NHWC -> NCHW: {batch_np.shape}")
        elif model_format == "NHWC":
            if (D1, D2, D3) != (H_model, W_model, C_model):
                # Convert NCHW -> NHWC
                batch_np = batch_np.transpose(0, 2, 3, 1)
                print(f"Transposed batch NCHW -> NHWC: {batch_np.shape}")
        else:
            raise ValueError("Cannot determine model input format (NCHW or NHWC).")

        # Run inference
        batch_outputs = run_onnx_inference(session, batch_np, input_name=input_name)
        outputs.extend(batch_outputs)

    # Display results
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
