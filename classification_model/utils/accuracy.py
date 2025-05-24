from typing import Tuple

import torch


def compute_topk_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    topk: Tuple[int, ...] = (1,)
) -> Tuple[float, ...]:
    """Computes the accuracy over top-k predictions for the specified values of k
    https://github.com/pytorch/examples/blob/cedca7729fef11c91e28099a0e45d7e98d03b66d/imagenet/main.py#L411

    Args:
        logits (torch.Tensor): model logits of the batch.
            The shape is (B, L) for batchsize B and number of labels L
        labels (torch.Tensor): labels of the batch
            The shape is (B, )
        topk (tuple of int, optional):
            k for computing top-k accuracy. Defaults to (1,).
                topk=(1,) returns (top1,)
                topk=(1,5) returns (top1, top5)

    Returns:
        Tuple[float]: top1 accuracy, or list of top-k accuracy values
    """
    assert topk[0] == 1, "topk[0] should be top1"
    assert len(topk) >= 1

    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)

        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.item() * 100.0 / batch_size)

        return tuple(res)
