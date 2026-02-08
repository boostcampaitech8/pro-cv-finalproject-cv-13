#m ade by JDH
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def compute_chamfer_distance_gpu(mask: torch.Tensor, is_3d: bool, max_iter: int = 50) -> torch.Tensor:
    """
    Approximate Euclidean distance transform using iterative max pooling.
    Fast GPU implementation.

    Args:
        mask: (H, W[, D]) binary mask on GPU
        is_3d: whether the mask is 3D
        max_iter: maximum number of iterations
    Returns:
        dist: (H, W[, D]) distance map on GPU
    """
    device = mask.device

    # Add batch and channel dimensions for conv operations
    if is_3d:
        x = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, D)
        kernel_size = 3
        padding = 1
    else:
        x = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        kernel_size = 3
        padding = 1

    # Initialize distance map
    dist = torch.zeros_like(x)

    # Iteratively dilate to compute distances
    current = x.clone()
    for i in range(1, max_iter + 1):
        if is_3d:
            dilated = F.max_pool3d(current, kernel_size, stride=1, padding=padding)
        else:
            dilated = F.max_pool2d(current, kernel_size, stride=1, padding=padding)

        # Find newly added pixels
        new_pixels = (dilated > 0) & (current == 0)
        dist[new_pixels] = i

        current = dilated

        # Early stopping if no new pixels
        if not new_pixels.any():
            break

    return dist.squeeze(0).squeeze(0)


def compute_distance_transform_gpu(seg: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    GPU-based approximate distance transform using max pooling.
    Fast but slightly less accurate than scipy.

    Args:
        seg: (H, W[, D]) integer label map on GPU
        num_classes: total number of classes
    Returns:
        dist_maps: (num_classes, H, W[, D]) tensor on GPU
    """
    device = seg.device
    is_3d = seg.ndim == 3

    # Initialize distance maps
    dist_maps = torch.zeros((num_classes, *seg.shape), device=device, dtype=torch.float32)

    for c in range(num_classes):
        mask = (seg == c).float()

        # Skip if empty or full
        if mask.sum() == 0 or mask.sum() == mask.numel():
            continue

        # Compute approximate distance using iterative dilation
        # Exterior distance (from boundary outward)
        exterior = compute_chamfer_distance_gpu(mask, is_3d, max_iter=50)

        # Interior distance (from boundary inward)
        interior = compute_chamfer_distance_gpu(1 - mask, is_3d, max_iter=50)

        # Signed distance: positive outside, negative inside
        dist_maps[c] = exterior - interior

    return dist_maps


def compute_distance_transform(seg: np.ndarray, num_classes: int) -> np.ndarray:
    """
    GPU-accelerated distance transform with numpy interface.
    Converts numpy input to torch, computes on GPU, returns numpy.

    Args:
        seg: (H, W[, D]) integer label map (numpy)
        num_classes: total number of classes including background
    Returns:
        dist_maps: (num_classes, H, W[, D]) float32 (numpy)
    """
    # Convert to torch and move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seg_tensor = torch.from_numpy(seg).to(device)

    # Compute on GPU
    dist_maps_tensor = compute_distance_transform_gpu(seg_tensor, num_classes)

    # Convert back to numpy
    return dist_maps_tensor.cpu().numpy()


def compute_class_weights(dataset_class_voxel_counts: dict) -> dict:
    """
    Compute class weights: w_k = (1/N_k) / sum_j(1/N_j)

    Args:
        dataset_class_voxel_counts: {class_idx: total_voxel_count}
    Returns:
        {class_idx: weight}
    """
    inv_counts = {k: 1.0 / v for k, v in dataset_class_voxel_counts.items() if v > 0}
    total = sum(inv_counts.values())
    return {k: v / total for k, v in inv_counts.items()}


class GeneralizedSurfaceLoss(nn.Module):
    """
    Generalized Surface Loss (Celaya et al., 2023) with GPU-accelerated distance transform.
    ℒ_gsl = 1 - [sum_k w_k sum_i (D_i^k * (1 - (T_i^k + P_i^k)))^2]
                / [sum_k w_k sum_i (D_i^k)^2]

    Output is bounded in [0, 1].

    Args:
        apply_nonlin: nonlinearity to apply to net_output (e.g. softmax)
        class_weights: dict {class_idx: weight} or None (uniform)
        apply_to_classes: list of class indices to apply GSL to, or None for all
    """
    def __init__(self, apply_nonlin=None, class_weights=None, apply_to_classes=None):
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.class_weights = class_weights
        self.apply_to_classes = apply_to_classes

    def forward(self, net_output: torch.Tensor, target_onehot: torch.Tensor,
                dist_maps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            net_output: (B, C, ...) logits
            target_onehot: (B, C, ...) one-hot encoded GT
            dist_maps: (B, C, ...) signed distance transform maps
        Returns:
            scalar loss in [0, 1]
        """
        if self.apply_nonlin is not None:
            pred = self.apply_nonlin(net_output)
        else:
            pred = net_output

        # Select specific classes if specified
        if self.apply_to_classes is not None:
            cls_idx = self.apply_to_classes
            pred = pred[:, cls_idx]
            target_onehot = target_onehot[:, cls_idx]
            dist_maps = dist_maps[:, cls_idx]

        # D_i^k * (1 - (T_i^k + P_i^k))
        residual = dist_maps * (1.0 - (target_onehot + pred))
        numerator = (residual ** 2)  # (B, C, ...)
        denominator = (dist_maps ** 2)  # (B, C, ...)

        # Apply class weights
        if self.class_weights is not None and self.apply_to_classes is not None:
            weight_tensor = torch.tensor(
                [self.class_weights.get(c, 1.0) for c in self.apply_to_classes],
                dtype=pred.dtype, device=pred.device
            )
            # reshape for broadcasting: (1, C, 1, 1, ...)
            shape = [1, len(self.apply_to_classes)] + [1] * (pred.ndim - 2)
            weight_tensor = weight_tensor.view(shape)
            numerator = numerator * weight_tensor
            denominator = denominator * weight_tensor

        num_sum = numerator.sum()
        den_sum = denominator.sum()

        if den_sum < 1e-8:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        gsl = 1.0 - (num_sum / den_sum)
        return gsl
