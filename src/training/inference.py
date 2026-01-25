from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn


def _get_sliding_window_starts(size: int, patch: int, stride: int) -> List[int]:
    if size <= patch:
        return [0]
    starts = list(range(0, size - patch + 1, stride))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


def sliding_window_inference(
    volume: torch.Tensor,
    model: nn.Module,
    patch_size: Tuple[int, int, int],
    overlap: float,
    device: torch.device,
) -> torch.Tensor:
    _, d, h, w = volume.shape
    pd = max(0, patch_size[0] - d)
    ph = max(0, patch_size[1] - h)
    pw = max(0, patch_size[2] - w)

    if pd or ph or pw:
        volume = torch.nn.functional.pad(volume, (0, pw, 0, ph, 0, pd))

    _, d_p, h_p, w_p = volume.shape
    stride_d = max(1, int(patch_size[0] * (1 - overlap)))
    stride_h = max(1, int(patch_size[1] * (1 - overlap)))
    stride_w = max(1, int(patch_size[2] * (1 - overlap)))

    starts_d = _get_sliding_window_starts(d_p, patch_size[0], stride_d)
    starts_h = _get_sliding_window_starts(h_p, patch_size[1], stride_h)
    starts_w = _get_sliding_window_starts(w_p, patch_size[2], stride_w)

    output_sum = None
    count = torch.zeros((1, d_p, h_p, w_p), dtype=torch.float32)

    for sd in starts_d:
        for sh in starts_h:
            for sw in starts_w:
                patch = volume[
                    :,
                    sd : sd + patch_size[0],
                    sh : sh + patch_size[1],
                    sw : sw + patch_size[2],
                ]
                patch = patch.unsqueeze(0).to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    logits = model(patch)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                logits = logits.detach().cpu()[0]

                if output_sum is None:
                    output_sum = torch.zeros(
                        (logits.shape[0], d_p, h_p, w_p),
                        dtype=torch.float32,
                    )
                output_sum[
                    :,
                    sd : sd + patch_size[0],
                    sh : sh + patch_size[1],
                    sw : sw + patch_size[2],
                ] += logits
                count[
                    :,
                    sd : sd + patch_size[0],
                    sh : sh + patch_size[1],
                    sw : sw + patch_size[2],
                ] += 1

    output_sum = output_sum / count
    return output_sum[:, :d, :h, :w]
