# made by LDH
import torch
from torch import nn
import torch.nn.functional as F


def _compute_chamfer_distance_gpu(mask: torch.Tensor, is_3d: bool, max_iter: int = 50) -> torch.Tensor:
    """
    Approximate Euclidean distance transform using iterative max pooling.
    Fast GPU implementation.
    """
    if is_3d:
        x = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, D)
        kernel_size = 3
        padding = 1
        pool = F.max_pool3d
    else:
        x = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        kernel_size = 3
        padding = 1
        pool = F.max_pool2d

    dist = torch.zeros_like(x)
    current = x.clone()

    for i in range(1, max_iter + 1):
        dilated = pool(current, kernel_size=kernel_size, stride=1, padding=padding)
        new_pixels = (dilated > 0) & (current == 0)
        dist[new_pixels] = float(i)
        current = dilated
        if not new_pixels.any():
            break

    return dist.squeeze(0).squeeze(0)


def _signed_distance_map(mask: torch.Tensor, max_iter: int = 50) -> torch.Tensor:
    """
    Signed distance map for a binary mask.
    Positive outside, negative inside, zero on the boundary.
    GPU approximation via chamfer distance.
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()

    # If empty or full, distance maps are not meaningful for a band-based loss.
    if mask.sum() == 0 or mask.sum() == mask.numel():
        return torch.zeros_like(mask, dtype=torch.float32)

    is_3d = mask.dim() == 3
    mask_f = mask.float()

    dist_out = _compute_chamfer_distance_gpu(mask_f, is_3d, max_iter=max_iter)
    dist_in = _compute_chamfer_distance_gpu(1.0 - mask_f, is_3d, max_iter=max_iter)
    return (dist_out - dist_in).float()

def _one_hot_from_labels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    labels: [B, ...] int
    returns: [B, C, ...] bool
    """
    shape = (labels.shape[0], num_classes, *labels.shape[1:])
    one_hot = torch.zeros(shape, device=labels.device, dtype=torch.bool)
    one_hot.scatter_(1, labels.unsqueeze(1).long(), True)
    return one_hot


class GeneralizedSurfaceLoss(nn.Module):
    """
        Practical surface-aware loss (GSL-like) for multi-class segmentation.

    Uses:
      - signed distance map of GT (computed per class)
      - band mask near surface
      - error term |p - y| to avoid "probability-only" shrinkage
    """
    def __init__(
        self,
        include_background: bool = False,
        ignore_label: int | None = None,
        eps: float = 1e-6,
        band_width: float = 3.0,
        min_band_voxels: int = 32, # 16 ~ 64
        error_power: float = 1.0,  # 1.0 => L1, 2.0 => L2 (더 강하게)
        class_indices: list[int] | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum'")
        if error_power not in (1.0, 2.0):
            raise ValueError("error_power must be 1.0 (L1) or 2.0 (L2)")

        self.include_background = include_background
        self.ignore_label = ignore_label
        self.eps = eps
        self.band_width = float(band_width)
        self.min_band_voxels = int(min_band_voxels)
        self.error_power = float(error_power)
        self.class_indices = class_indices
        self.reduction = reduction

    def forward(self, probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        probs: [B, C, ...] float probs after softmax
        target: [B, ...] int label map or [B, C, ...] one-hot/soft target
        """
        if probs.dim() < 3:
            raise ValueError(f"Expected probs [B, C, ...], got {probs.shape}")

        B, C = probs.shape[0], probs.shape[1]
        probs = probs.float()

        # Prepare target_one_hot and valid mask for ignore_label
        # If target is [B,1,...] (label map with channel dim) but probs has C>1,
        # treat it as label map by squeezing channel.
        if target.dim() == probs.dim() and target.shape[1] == 1 and C > 1:
            target = target[:, 0]

        if target.dim() == probs.dim() - 1:
            # label map case: [B, ...]
            if self.ignore_label is not None:
                valid = (target != self.ignore_label)
                # for one-hot creation, set ignored to background label (0),
                # but we'll mask them out later using `valid`.
                target_for_oh = target.clone()
                target_for_oh[~valid] = 0
            else:
                valid = torch.ones_like(target, dtype=torch.bool, device=target.device)
                target_for_oh = target

            target_one_hot = _one_hot_from_labels(target_for_oh, C)  # bool [B,C,...]

        elif target.dim() == probs.dim():
            # one-hot/soft target: [B, C, ...]
            # In this mode we can't reliably apply ignore_label (no label map).
            valid = None
            target_one_hot = target > 0.5

        else:
            raise ValueError(f"Expected target [B, ...] or [B, C, ...], got {target.shape}")

        # classes to include
        if self.class_indices is not None:
            class_range = [c for c in self.class_indices if 0 <= c < C]
        else:
            if (not self.include_background) and C > 1:
                class_range = range(1, C)
            else:
                class_range = range(C)

        if len(class_range) == 0:
            return probs.new_zeros(())

        loss_sum = probs.new_zeros(())
        count = 0

        for c in class_range:
            for b in range(B):
                # Compute signed distance map from GT for this (b,c)
                with torch.no_grad():
                    dist_t = _signed_distance_map(target_one_hot[b, c])
                d = dist_t.abs().to(dtype=probs.dtype)

                # band near surface
                band = (d <= self.band_width)

                # apply ignore mask if label-map mode
                if valid is not None:
                    band = band & valid[b]

                band_f = band.to(probs.dtype)
                denom = band_f.sum()

                # if band is empty (or too small), skip to avoid noisy gradients
                if denom.item() < self.min_band_voxels:
                    continue

                # error term: |p - y| (or squared)
                y = target_one_hot[b, c].to(probs.dtype)
                err = (probs[b, c] - y).abs()
                if self.error_power == 2.0:
                    err = err * err

                loss_bc = (err * d * band_f).sum() / (denom + self.eps)
                loss_sum = loss_sum + loss_bc
                count += 1

        if self.reduction == "sum":
            return loss_sum

        return loss_sum / (count + self.eps)


class BoundaryDiceLoss(nn.Module):
    """
    Boundary Dice loss for multi-class segmentation.
    Uses soft boundary maps (via max-pooling) and computes Dice on boundaries.
    """
    def __init__(
        self,
        include_background: bool = False,
        ignore_label: int | None = None,
        eps: float = 1e-6,
        boundary_width: int = 1,
    ):
        super().__init__()
        self.include_background = include_background
        self.ignore_label = ignore_label
        self.eps = eps
        self.boundary_width = int(boundary_width)

    @staticmethod
    def _make_boundary(x: torch.Tensor, width: int) -> torch.Tensor:
        """
        x: [B, C, ...] float in [0,1], returns soft boundary map.
        """
        if x.dim() == 4:
            pool = torch.nn.functional.max_pool2d
        elif x.dim() == 5:
            pool = torch.nn.functional.max_pool3d
        else:
            raise ValueError(f"Expected 2D/3D tensor, got {x.shape}")

        # Use dilation - erosion to approximate boundary
        pad = width
        kernel = 2 * width + 1
        dilated = pool(x, kernel_size=kernel, stride=1, padding=pad)
        eroded = -pool(-x, kernel_size=kernel, stride=1, padding=pad)
        boundary = (dilated - eroded).clamp_min(0)
        return boundary

    def forward(self, probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if probs.dim() < 4:
            raise ValueError(f"Expected probs [B, C, ...], got {probs.shape}")

        B, C = probs.shape[0], probs.shape[1]
        probs = probs.float()

        # Handle label map vs one-hot
        if target.dim() == probs.dim() and target.shape[1] == 1 and C > 1:
            target = target[:, 0]

        if target.dim() == probs.dim() - 1:
            if self.ignore_label is not None:
                valid = (target != self.ignore_label)
                target_for_oh = target.clone()
                target_for_oh[~valid] = 0
            else:
                valid = None
                target_for_oh = target
            target_one_hot = _one_hot_from_labels(target_for_oh, C).float()
        elif target.dim() == probs.dim():
            valid = None
            target_one_hot = (target > 0.5).float()
        else:
            raise ValueError(f"Expected target [B, ...] or [B, C, ...], got {target.shape}")

        if (not self.include_background) and C > 1:
            class_range = range(1, C)
        else:
            class_range = range(C)

        if len(class_range) == 0:
            return probs.new_zeros(())

        pred_b = self._make_boundary(probs, self.boundary_width)
        gt_b = self._make_boundary(target_one_hot, self.boundary_width)

        loss_sum = probs.new_zeros(())
        count = 0
        for c in class_range:
            pb = pred_b[:, c]
            gb = gt_b[:, c]
            if valid is not None:
                pb = pb * valid.float()
                gb = gb * valid.float()

            inter = (pb * gb).sum()
            denom = pb.sum() + gb.sum()
            dice = (2 * inter + self.eps) / (denom + self.eps)
            loss_sum = loss_sum + (1.0 - dice)
            count += 1

        return loss_sum / (count + self.eps)
