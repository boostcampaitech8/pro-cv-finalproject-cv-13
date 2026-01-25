from __future__ import annotations

from typing import List, Tuple, Optional

import torch
from torch import nn


def _compute_steps_for_sliding_window(
    image_size: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    tile_step_size: float,
) -> List[List[int]]:
    if any(i < j for i, j in zip(image_size, patch_size)):
        raise ValueError("image size must be >= patch size in all dims")
    if not (0 < tile_step_size <= 1):
        raise ValueError("tile_step_size must be in (0, 1]")

    target_step_sizes = [i * tile_step_size for i in patch_size]
    num_steps = [
        int(torch.ceil(torch.tensor((i - k) / j)).item()) + 1
        for i, j, k in zip(image_size, target_step_sizes, patch_size)
    ]

    steps = []
    for dim in range(len(patch_size)):
        max_step_value = image_size[dim] - patch_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 1e9
        steps.append([int(round(actual_step_size * i)) for i in range(num_steps[dim])])
    return steps


def _compute_gaussian(
    patch_size: Tuple[int, int, int],
    sigma_scale: float = 1.0 / 8,
    value_scaling_factor: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    coords = [torch.arange(s, device=device, dtype=dtype) for s in patch_size]
    grids = torch.meshgrid(*coords, indexing="ij")
    center = [(s - 1) / 2.0 for s in patch_size]
    sigmas = [s * sigma_scale for s in patch_size]
    dist = torch.zeros(patch_size, device=device, dtype=dtype)
    for g, c, s in zip(grids, center, sigmas):
        dist += ((g - c) ** 2) / (2 * (s**2 + 1e-8))
    gaussian = torch.exp(-dist)
    gaussian = gaussian / (gaussian.max() / value_scaling_factor)
    gaussian = gaussian.to(device=device, dtype=dtype)
    mask = gaussian == 0
    if torch.any(mask):
        gaussian[mask] = torch.min(gaussian[~mask])
    return gaussian


def _maybe_mirror_and_predict(
    patch: torch.Tensor,
    model: nn.Module,
    mirror_axes: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    logits = model(patch)
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    if not mirror_axes:
        return logits

    axes = [ax + 2 for ax in mirror_axes]
    combos = []
    for i in range(len(axes)):
        for c in torch.combinations(torch.tensor(axes), r=i + 1):
            combos.append(tuple(int(x) for x in c))

    pred = logits.clone()
    for ax in combos:
        flipped = torch.flip(patch, dims=ax)
        out = model(flipped)
        if isinstance(out, (list, tuple)):
            out = out[0]
        pred += torch.flip(out, dims=ax)
    pred = pred / (len(combos) + 1)
    return pred


def sliding_window_inference(
    volume: torch.Tensor,
    model: nn.Module,
    patch_size: Tuple[int, int, int],
    overlap: float,
    device: torch.device,
    use_gaussian: bool = True,
    mirror_axes: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    _, d, h, w = volume.shape
    pd = max(0, patch_size[0] - d)
    ph = max(0, patch_size[1] - h)
    pw = max(0, patch_size[2] - w)

    if pd or ph or pw:
        volume = torch.nn.functional.pad(volume, (0, pw, 0, ph, 0, pd))

    _, d_p, h_p, w_p = volume.shape
    tile_step_size = 1.0 - overlap
    steps = _compute_steps_for_sliding_window((d_p, h_p, w_p), patch_size, tile_step_size)

    output_sum = None
    count = torch.zeros((1, d_p, h_p, w_p), dtype=torch.float32)
    gaussian = _compute_gaussian(patch_size, device=device) if use_gaussian else None

    for sd in steps[0]:
        for sh in steps[1]:
            for sw in steps[2]:
                patch = volume[
                    :,
                    sd : sd + patch_size[0],
                    sh : sh + patch_size[1],
                    sw : sw + patch_size[2],
                ]
                patch = patch.unsqueeze(0).to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    logits = _maybe_mirror_and_predict(patch, model, mirror_axes=mirror_axes)
                logits = logits.detach().cpu()[0]

                if output_sum is None:
                    output_sum = torch.zeros(
                        (logits.shape[0], d_p, h_p, w_p),
                        dtype=torch.float32,
                    )
                weight = gaussian.detach().cpu() if gaussian is not None else 1.0
                output_sum[
                    :,
                    sd : sd + patch_size[0],
                    sh : sh + patch_size[1],
                    sw : sw + patch_size[2],
                ] += logits * weight
                count[
                    :,
                    sd : sd + patch_size[0],
                    sh : sh + patch_size[1],
                    sw : sw + patch_size[2],
                ] += weight

    output_sum = output_sum / count
    return output_sum[:, :d, :h, :w]
