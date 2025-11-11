from torch.utils.data import Dataset, DataLoader
from image_preprocessing import preprocess_image 
from torch import Tensor
from typing import List


class InferenceDataset(Dataset):
    def __init__(self, image_paths: List[str], target_height: int=224, target_width: int=224) -> None:
        """ Dataset for loading and preprocessing images for ONNX model inference. 
        Args: 
            image_paths (list[str]): List of paths to input images. 
            target_height (int): Target height for resizing. 
            target_width (int): Target width for resizing. 
        """
        self.image_paths = image_paths
        self.target_height = target_height
        self.target_width = target_width

    def __len__(self) -> int:
        """ Returns the number of images in the dataset. """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tensor:
        """ Get the preprocessed image tensor at the specified index. 
        Args: 
            idx (int): Index of the image to retrieve. 
        Returns: 
            torch.Tensor: Preprocessed image tensor. 
        """
        img_tensor = preprocess_image(
            self.image_paths[idx], 
            self.target_height, 
            self.target_width
        )
        return img_tensor

def get_dataloader(image_paths: List[str], batch_size: int=16, num_workers: int=4, target_height: int=224, target_width: int=224) -> DataLoader:
    """ Create a DataLoader for the InferenceDataset. 
    Args: 
        image_paths (list[str]): List of paths to input images. 
        batch_size (int): Number of images per batch. 
        num_workers (int): Number of subprocesses for data loading. 
    Returns: 
        DataLoader: DataLoader for the InferenceDataset. 
    """
    dataset = InferenceDataset(
        image_paths,
        target_height=target_height,
        target_width=target_width
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader
