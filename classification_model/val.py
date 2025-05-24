from dataclasses import dataclass

import comet_ml
import torch
from torch.utils.data import DataLoader

from utils import (
    compute_topk_accuracy,
    AvgMeterLossTopk,
    TqdmLossTopK,
)
from detection_model import ClassificationBaseModel, get_device


@dataclass
class ValidationOutput:
    loss: float
    top1: float
    val_step: int


def validation(
    model: ClassificationBaseModel,
    val_loader: DataLoader,
    current_val_step: int,
    current_epoch: int,
    logger: comet_ml.Experiment,
) -> ValidationOutput:
    """validation for the current model

    Args:
        model(ClassificationBaseModel): CNN model
        loader(DataLoader): validation dataset loader
        current_val_step(int): current step for validation
        current_epoch(int): current epoch
        logger(comet_ml.Experiment): comet logger

    Returns:
        ValidationOutput: val loss, val top1, steps for validation
    """

    val_meter = AvgMeterLossTopk("val")

    model.eval()
    device = get_device(model)

    with torch.no_grad(), TqdmLossTopK(
        val_loader,
        total=len(val_loader),
        leave=False,
        unit='step',
    ) as progress_bar_step:
        progress_bar_step.set_description("[val      ]")

        for batch in progress_bar_step:
            data, labels = batch  # (BCHW, B) or {'video': BCTHW, 'label': B}

            data = data.to(device)
            labels = labels.to(device)
            batch_size = data.size(0)

            outputs = model(data, labels=labels)
            loss = outputs.loss

            val_topk = compute_topk_accuracy(outputs.logits, labels, topk=(1, 5))
            val_meter.update(loss, val_topk, batch_size)

            progress_bar_step.set_postfix_str_loss_topk(
                current_val_step, loss, val_topk
            )
            logger.log_metrics(
                val_meter.get_step_metrics_dict(),
                step=current_val_step,
                epoch=current_epoch,
            )
            current_val_step += 1

    logger.log_metrics(
        val_meter.get_epoch_metrics_dict(),
        step=current_val_step,
        epoch=current_epoch,
    )

    return ValidationOutput(
        loss=val_meter.loss_meter.avg,
        top1=val_meter.topk_meters[0].avg,  # top1: topk[0] should be 1
        val_step=current_val_step
    )
