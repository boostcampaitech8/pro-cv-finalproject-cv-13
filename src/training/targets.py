from __future__ import annotations

from typing import Tuple

import torch


def resize_target(target: torch.Tensor, shape: Tuple[int, int, int]) -> torch.Tensor:
    resized = torch.nn.functional.interpolate(target[:, None].float(), size=shape, mode="nearest")
    return resized[:, 0].long()


def prepare_targets(outputs, target):
    if not isinstance(outputs, (list, tuple)):
        return target
    targets = []
    for out in outputs:
        if out.shape[2:] == target.shape[1:]:
            targets.append(target)
        else:
            targets.append(resize_target(target, out.shape[2:]))
    return targets
