from typing import Iterator, Literal

from torch.nn.parameter import Parameter
from torch.optim import Optimizer
from torch.optim import SGD, Adam

SupportedOptimizers = Literal["SGD", "Adam"]


def configure_optimizer(
    optimizer_name: SupportedOptimizers,
    model_params: Iterator[Parameter],
    lr: float,
    weight_decay: float,
    momentum: float = 0.9,
) -> Optimizer:
    """optimizer factory

    Args:
        optimizer_name (SupportedOptimizers): optimizer name (str).
            ["SGD", "Adam"]
        model_params (Iterator[Parameter]): model parameters.
            Typically "model.parameters()"
        lr (float): learning rate.
        weight_decay (float): weight decay
        momentum (float, optional): momentum. Defaults to 0.9.

    Raises:
        ValueError: invalide optimizer name given by command line

    Returns:
        Optimizer: optimizer
    """

    if optimizer_name == "SGD":
        return SGD(
            model_params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    if optimizer_name == "Adam":
        return Adam(
            model_params,
            lr=lr,
            weight_decay=weight_decay,
        )

    raise ValueError("invalid optimizer_name")
