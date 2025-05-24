import os
from dataclasses import dataclass
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2 as transforms


@dataclass
class ImageFolderInfo():
    root: str
    train_dir: str
    val_dir: str
    batch_size: int
    num_workers: int
    train_transform: transforms
    val_transform: transforms


def image_folder(
        image_folder_info: ImageFolderInfo
) -> Tuple[DataLoader, DataLoader, int]:
    """creating dataloaders for images in folders by torchvision

    Args:
        image_folder_info (ImageFolderInfo): information for dataloaders

    Returns:
        DataLoader: train dataloader
        DataLoader: val dataloader
        int: number of classes
    """

    root_train_dir = os.path.join(image_folder_info.root, image_folder_info.train_dir)
    root_val_dir = os.path.join(image_folder_info.root, image_folder_info.val_dir)
    assert os.path.exists(root_train_dir) and os.path.isdir(root_train_dir)
    assert os.path.exists(root_val_dir) and os.path.isdir(root_val_dir)

    train_dataset = ImageFolder(
        root=root_train_dir,
        transform=image_folder_info.train_transform)
    val_dataset = ImageFolder(
        root=root_val_dir,
        transform=image_folder_info.val_transform)

    assert sorted(train_dataset.classes) == sorted(val_dataset.classes)
    assert len(train_dataset.classes) == len(val_dataset.classes)
    n_classes = len(train_dataset.classes)

    train_loader = DataLoader(
        train_dataset,
        batch_size=image_folder_info.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=image_folder_info.num_workers)
    val_loader = DataLoader(
        val_dataset,
        batch_size=image_folder_info.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=image_folder_info.num_workers)

    return train_loader, val_loader, n_classes
