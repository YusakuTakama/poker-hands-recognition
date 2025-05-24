import torch
from typing import Tuple, Literal

from utils.mixin import (
    GetMetricsDictMixin,
)


class AverageMeter:
    """
    Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    https://github.com/machine-perception-robotics-group/attention_branch_network/blob/ced1d97303792ac6d56442571d71bb0572b3efd8/utils/misc.py#L59
    """

    def __init__(self):
        """average meter
        """
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(
        self,
        value: float | torch.Tensor,
        n: int = 1
    ):
        """update the statistics

        Args:
            value (float or torch.Tensor): a value used for averaging
            n (int, optional): multiplier of the curent value for averaging. Defaults to 1.
        """
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


class AvgMeterLossTopk(GetMetricsDictMixin):
    """average meter set for loss and top1/top5

    """

    def __init__(
        self,
        mode_name: Literal['train', 'val'],
        topk: Tuple[int, ...] = (1, 5)
    ):
        """a set of average meters for loss and topk

        Args:
            mode_name (Literal['train', 'val']): prefix
            topk (Tuple[int], optional): specifying (1, 5) logs top1 and top5. Defaults to (1, 5).
        """
        self.mode_name = mode_name
        self.topk = topk
        self.loss_meter = AverageMeter()
        self.topk_meters = [AverageMeter() for _ in topk]

    def update(
        self,
        loss: float | torch.Tensor,
        topk_values: Tuple[float, ...] | Tuple[torch.Tensor, ...],
        batch_size: int = 1
    ):
        """update average meters with statistics of a single batch

        Args:
            loss (float | torch.Tensor): a batch loss
            topk_values (Tuple[float, ...] | Tuple[torch.Tensor, ...]): a batch topk values
            batch_size (int, optional): batch size for the loss. Defaults to 1.
        """
        self.loss_meter.update(loss, batch_size)
        for meter, value in zip(self.topk_meters, topk_values):
            meter.update(value, batch_size)  # type: ignore[arg-type]

    def get_meters(self):
        return self.loss_meter, self.topk_meters, self.topk

    def get_mode_name(self):
        return self.mode_name
