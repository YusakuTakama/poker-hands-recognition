from typing import Tuple

from tqdm import tqdm


class TqdmLossTopK(tqdm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.postfix_str = ""

    def set_postfix_str_loss_topk(
        self,
        global_step: int,
        loss: float,
        topk_values: Tuple[float, ...],
        topk: Tuple[int, ...] = (1, 5),
    ) -> str:
        assert len(topk) == len(topk_values)

        self.init_postfix_str()
        self.add_step_to_postfix_str(global_step)
        self.add_loss_to_postfix_str(loss)
        self.add_topk_to_postfix_str(topk_values, topk)

        self.set_postfix_str(self.postfix_str)
        return self.postfix_str  # type: ignore[no-any-return]

    def init_postfix_str(self) -> None:
        self.postfix_str = ""

    def add_step_to_postfix_str(self, step: int) -> None:
        self.postfix_str += f"step={step:d}, "

    def add_loss_to_postfix_str(self, loss: float) -> None:
        self.postfix_str += f"loss={loss:6.4e} "

    def add_topk_to_postfix_str(
        self,
        topk_values: Tuple[float, ...],
        topk: Tuple[int, ...]
    ) -> None:
        for k, value in zip(topk, topk_values):
            self.postfix_str += f"top{k}={value:6.2f} "
