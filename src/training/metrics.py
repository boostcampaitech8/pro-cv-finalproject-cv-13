from __future__ import annotations

import torch


def dice_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    if isinstance(target, (list, tuple)):
        target = target[0]
    pred = pred.argmax(1)
    dice_scores = []
    for cls in range(1, num_classes):
        pred_c = pred == cls
        target_c = target == cls
        inter = (pred_c & target_c).sum().item()
        denom = pred_c.sum().item() + target_c.sum().item()
        if denom == 0:
            continue
        dice_scores.append(2.0 * inter / (denom + 1e-6))
    if not dice_scores:
        return 0.0
    return float(sum(dice_scores) / len(dice_scores))


def pick_highest_res_output(outputs, targets):
    if not isinstance(outputs, (list, tuple)):
        return outputs, targets
    if not isinstance(targets, (list, tuple)):
        targets = [targets] * len(outputs)
    best_idx = max(range(len(outputs)), key=lambda i: outputs[i].shape[2])
    return outputs[best_idx], targets[best_idx]
