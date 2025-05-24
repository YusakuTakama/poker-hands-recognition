from dataclasses import dataclass

from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
from torchvision.datasets import CIFAR10


@dataclass
class Cifar10Info():
    root: str
    batch_size: int
    num_workers: int
    train_transform: transforms
    val_transform: transforms


def cifar10(
        cifar10_info: Cifar10Info
):

    train_dataset = CIFAR10(
        root=cifar10_info.root,
        train=True,
        download=True,
        transform=cifar10_info.train_transform)
    val_dataset = CIFAR10(
        root=cifar10_info.root,
        train=False,
        download=True,
        transform=cifar10_info.val_transform)
    n_classes = 10

    train_loader = DataLoader(
        train_dataset,
        batch_size=cifar10_info.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cifar10_info.num_workers)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cifar10_info.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cifar10_info.num_workers)

    return train_loader, val_loader, n_classes
