from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    LRScheduler,
    StepLR,
    ConstantLR,
)


def configure_scheduler(
    optimizer: Optimizer,
    use_scheduler: bool,
) -> LRScheduler:
    """scheduler factory

    Args:
        optimizer (Optimizer): optimizer
        use_scheduler (bool): flag if scheduler is used.
            Use StepLR if True, or dummy_scheduler (no lr scheduling) if False.

    Returns:
        LRScheduler: learning rate scheduler
    """
    if use_scheduler:
        return StepLR(
            optimizer,
            step_size=10,  # every 10 epoch
            gamma=0.1  # lr = lr * 0.1
        )

    return dummy_scheduler(optimizer)


def dummy_scheduler(optimizer) -> LRScheduler:
    """A dummy scheduler that doesn't change lr because of factor=1.0
    """
    return ConstantLR(
        optimizer,
        factor=1.0,
        total_iters=65535,  # dummy max
    )
