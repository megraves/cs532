import torch
import torchvision.models as models
import numpy as np
import cv2
from dataset import get_dataloader
from utils import load_config, get_random_images_from_csv, get_class_name, load_class_mapping


def interactive_torch_inference(config_path: str) -> None:
    """
    Run inference on random images using a pretrained PyTorch model.

    Args:
        config_path (str): Path to the configuration YAML file.
    """
    config = load_config(config_path)

    use_gpu = config["model"].get("use_gpu", False)
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    model_name = config["model"].get("name", "squeezenet1_1")
    if model_name == "squeezenet1_1":
        model = models.squeezenet1_1(pretrained=True)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    model.eval()
    model.to(device)

    csv_path = config["data"]["csv_path"]
    data_root = config["data"].get("root", "data/imagenette2")
    num_random_images = config["data"].get("num_random_images", 5)
    batch_size = config["data"].get("batch_size", 4)
    num_workers = config["data"].get("num_workers", 4)
    target_height = config["data"].get("target_height", 224)
    target_width = config["data"].get("target_width", 224)
    class_mapping_file = config["data"].get("class_mapping", "data/imagenette2/index_to_class.txt")

    class_mapping = load_class_mapping(class_mapping_file)
    image_paths = get_random_images_from_csv(csv_path, data_root, num_images=num_random_images)
    print("Selected random images:")
    for path in image_paths:
        print("  -", path)

    dataloader = get_dataloader(
        image_paths=image_paths,
        batch_size=batch_size,
        num_workers=num_workers,
        target_height=target_height,
        target_width=target_width
    )

    outputs = []
    for batch in dataloader:
        batch = batch.to(device)
        with torch.no_grad():
            preds = model(batch)
            outputs.extend(preds.cpu().numpy())

    cv2.namedWindow("Interactive Inference", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Interactive Inference", 1200, 800)

    idx = 0
    while True:
        output = outputs[idx]
        pred_index = int(np.argmax(output))
        class_name = get_class_name(pred_index, class_mapping)

        img = cv2.imread(image_paths[idx])
        display_img = img.copy()

        cv2.putText(
            display_img,
            f"Predicted: {class_name} ({pred_index})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Interactive Inference", display_img)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == 81:
            idx = (idx - 1) % len(image_paths)
        elif key == 83:
            idx = (idx + 1) % len(image_paths)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python interactive_torch_inference.py <path_to_config.yml>")
        sys.exit(1)

    interactive_torch_inference(sys.argv[1])
