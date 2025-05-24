from dataclasses import dataclass
from typing import Tuple

import numpy as np

from torch.utils.data import (
    DataLoader,
    Dataset
)
from torchvision.transforms import v2 as transforms


@dataclass
class ZeroImageInfo():
    batch_size: int
    num_workers: int
    transform: transforms
    image_size: int = 256
    dataset_size: int = 1000
    n_classes: int = 10


class ZeroImageDataset(Dataset):
    def __init__(self, dataset_size, image_size, n_classes, transform=None):
        self.transform = transform
        self.dataset_size = dataset_size
        self.image_size = image_size
        self.n_classes = n_classes

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        image = np.zeros((self.image_size, self.image_size, 3))  # HWC
        label = index % self.n_classes
        if self.transform:
            image = self.transform(image)  # HWC, ndarray --> CWH, tensor
        return image, label


def zero_images(
        zero_image_info: ZeroImageInfo
) -> Tuple[DataLoader, DataLoader, int]:

    zero_image_dataset_dict = {
        "dataset_size": zero_image_info.dataset_size,
        "image_size": zero_image_info.image_size,
        "transform": zero_image_info.transform,
        "n_classes": zero_image_info.n_classes
    }

    train_loader = DataLoader(
        ZeroImageDataset(**zero_image_dataset_dict),
        batch_size=zero_image_info.batch_size,
        num_workers=zero_image_info.num_workers
    )
    val_loader = DataLoader(
        ZeroImageDataset(**zero_image_dataset_dict),
        batch_size=zero_image_info.batch_size,
        num_workers=zero_image_info.num_workers
    )

    return train_loader, val_loader, zero_image_info.n_classes
