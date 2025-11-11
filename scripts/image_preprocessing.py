from PIL import Image
from torchvision import transforms
from torch import Tensor


def preprocess_image(image_path: str, target_height: int=224, target_width: int=224) -> Tensor: 
    """ Preprocess the input image for model inference. 
    Args: 
        image_path (str): Path to the input image. 
        target_height (int): Target height for resizing. 
        target_width (int): Target width for resizing. 

    Returns: 
        torch.Tensor: Preprocessed image tensor. 
    """
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((target_height, target_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image)
    return image_tensor
