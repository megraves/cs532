import numpy as np
import cv2
from inference_utils import create_onnx_session, get_model_input_details, run_onnx_inference
from dataset import get_dataloader
from utils import load_config, get_random_images_from_csv, get_class_name, load_class_mapping


def interactive_inference(config_path: str) -> None:
    """ Run interactive inference on random images from a dataset.
    
    Args:
        config_path (str): Path to the configuration YAML file.
    """
    config = load_config(config_path)

    model_path = config["model"]["path"]
    use_gpu = config["model"].get("use_gpu", False)

    csv_path = config["data"]["csv_path"]
    data_root = config["data"].get("root", "data/imagenette2")
    num_random_images = config["data"].get("num_random_images", 5)
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

    if batch_dim is not None:
        batch_size = batch_dim

    dataloader = get_dataloader(
        image_paths=image_paths,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    outputs = run_onnx_inference(session, dataloader)

    # Create a resizable window and set its initial size
    cv2.namedWindow("Interactive Inference", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Interactive Inference", 1200, 800)  # Adjust width and height as needed

    idx = 0
    while True:
        output = outputs[idx]
        pred_index = int(np.argmax(output))
        class_name = get_class_name(pred_index, class_mapping)

        img = cv2.imread(image_paths[idx])
        display_img = img.copy()

        # Reduced font size and thickness further
        cv2.putText(display_img,
                    f"Predicted: {class_name} ({pred_index})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("Interactive Inference", display_img)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == 81:  # Left arrow
            idx = (idx - 1) % len(image_paths)
        elif key == 83:  # Right arrow
            idx = (idx + 1) % len(image_paths)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python interactive_inference.py <path_to_config.yml>")
        sys.exit(1)

    interactive_inference(sys.argv[1])
