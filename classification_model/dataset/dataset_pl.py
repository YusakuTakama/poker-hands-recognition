import argparse

import lightning.pytorch as pl

from dataset import configure_dataloader


class TrainValDataModule(pl.LightningDataModule):
    def __init__(
        self,
        command_line_args: argparse.Namespace,
        dataset_name: str,
    ):
        super().__init__()
        self.args = command_line_args

        self.dataloaders_info = \
            configure_dataloader(
                command_line_args=command_line_args,
                dataset_name=dataset_name  # type: ignore[arg-type]
            )

    def train_dataloader(self):
        return self.dataloaders_info.train_loader

    def val_dataloader(self):
        return self.dataloaders_info.val_loader

    @property
    def n_classes(self):
        return self.dataloaders_info.n_classes
