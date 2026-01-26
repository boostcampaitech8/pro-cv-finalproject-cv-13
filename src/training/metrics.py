from __future__ import annotations

import torch


def _get_tp_fp_fn(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> tuple[list[int], list[int], list[int]]:
    tp = []
    fp = []
    fn = []
    for cls in range(1, num_classes):
        pred_c = pred == cls
        target_c = target == cls
        tp.append(int((pred_c & target_c).sum().item()))
        fp.append(int((pred_c & ~target_c).sum().item()))
        fn.append(int((~pred_c & target_c).sum().item()))
    return tp, fp, fn


def dice_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    if isinstance(target, (list, tuple)):
        target = target[0]

    pred = pred.argmax(1)
    if pred.ndim != target.ndim:
        target = target.squeeze(1)

    tp, fp, fn = _get_tp_fp_fn(pred, target, num_classes)
    dice = []
    for t, f, n in zip(tp, fp, fn):
        denom = (2 * t + f + n)
        if denom == 0:
            continue
        dice.append(2 * t / denom)
    if not dice:
        return 0.0
    return float(sum(dice) / len(dice))


def pick_highest_res_output(outputs, targets):
    if not isinstance(outputs, (list, tuple)):
        return outputs, targets
    if not isinstance(targets, (list, tuple)):
        targets = [targets] * len(outputs)
    best_idx = max(range(len(outputs)), key=lambda i: outputs[i].shape[2])
    return outputs[best_idx], targets[best_idx]
