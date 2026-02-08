# made by JDH
"""
nnUNetTrainer with Generalized Surface Loss (Celaya et al., 2023)
for specific classes.

Usage:
    nnUNetv2_train DATASET_ID CONFIG FOLD -tr nnUNetTrainerGSL

Modify GSL_CLASSES, GSL_WEIGHT, and GSL_SCHEDULE below as needed.
"""

import numpy as np
import torch
from typing import List
from torch import autocast

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
'''
from nnunetv2.training.loss.generalized_surface_loss import (
    GeneralizedSurfaceLoss,
    compute_distance_transform,
)

# 기존 (CPU 버전)
from nnunetv2.training.loss.generalized_surface_loss import (
    GeneralizedSurfaceLoss, compute_distance_transform
)
'''
# 변경 옵션 1: PyTorch GPU 버전 (추천)
from nnunetv2.training.loss.generalized_surface_loss_gpu import (
    GeneralizedSurfaceLoss, compute_distance_transform
)

'''
# 변경 옵션 2: MONAI 버전
from nnunetv2.training.loss.generalized_surface_loss_monai import (
    GeneralizedSurfaceLoss, compute_distance_transform
)
'''
from nnunetv2.utilities.helpers import softmax_helper_dim1, dummy_context


class nnUNetTrainerGSL(nnUNetTrainer):
    """
    nnUNetTrainer + Generalized Surface Loss on selected classes.

    Configure these class attributes before training:
        GSL_CLASSES: list of class indices to apply GSL (e.g. [1, 2])
                     None = all foreground classes
        GSL_WEIGHT:  max weight (1-alpha) for the surface loss term
        GSL_SCHEDULE: 'linear', 'cosine', or 'step'
    """
    # ============ USER CONFIG ============
    GSL_CLASSES = [1,2,3,12,13]       # None → all foreground classes; or e.g. [1, 3]
    GSL_WEIGHT = 0.5         # max surface loss weight
    GSL_SCHEDULE = 'linear'  # 'linear', 'cosine', or 'step'
    # =====================================

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.gsl_classes = self.GSL_CLASSES
        self.gsl_max_weight = self.GSL_WEIGHT
        self.gsl_schedule = self.GSL_SCHEDULE
        self.num_iterations_per_epoch = 250  # 250 → 10 (for faster training)

    def _build_loss(self):
        # Build standard DC+CE loss with deep supervision as usual
        loss = super()._build_loss()
        # GSL is computed separately in train_step, so just return base loss
        return loss

    def _get_gsl_alpha(self) -> float:
        """
        Returns (1-alpha): the weight for the GSL term.
        alpha starts at 1 (no GSL) and decreases → GSL weight increases.
        """
        t = self.current_epoch
        T = self.num_epochs

        if self.gsl_schedule == 'linear':
            alpha = 1.0 - t / T
        elif self.gsl_schedule == 'cosine':
            alpha = 0.5 * (1.0 + np.cos(np.pi * t / T))
        elif self.gsl_schedule == 'step':
            num_steps = 10
            step_size = T / num_steps
            alpha = 1.0 - (t // step_size) / num_steps
        else:
            alpha = 1.0 - t / T

        gsl_weight = (1.0 - alpha) * self.gsl_max_weight
        return gsl_weight

    def _compute_batch_dist_maps(self, target_tensor: torch.Tensor,
                                  num_classes: int) -> torch.Tensor:
        """
        Compute signed distance transform maps for a batch.

        Args:
            target_tensor: (B, 1, ...) label map on CPU
            num_classes: number of classes
        Returns:
            dist_maps: (B, num_classes, ...) tensor
        """
        target_np = target_tensor.cpu().numpy()
        B = target_np.shape[0]
        seg = target_np[:, 0]  # (B, H, W[, D])

        all_maps = []
        for b in range(B):
            dm = compute_distance_transform(seg[b].astype(int), num_classes)
            all_maps.append(dm)

        dist_maps = np.stack(all_maps, axis=0)  # (B, C, ...)
        return torch.from_numpy(dist_maps).float()

    def _make_onehot(self, target: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        Convert (B, 1, ...) label map to (B, C, ...) one-hot.
        """
        seg = target[:, 0].long()  # (B, ...)
        onehot = torch.zeros(seg.shape[0], num_classes, *seg.shape[1:],
                             dtype=torch.float32, device=target.device)
        onehot.scatter_(1, seg.unsqueeze(1), 1.0)
        return onehot

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)

            # Standard loss (DC+CE with deep supervision)
            l = self.loss(output, target)

            # GSL: apply only on full-resolution output
            gsl_weight = self._get_gsl_alpha()
            if gsl_weight > 0:
                # Get full-resolution output and target
                if isinstance(output, (list, tuple)):
                    full_output = output[0]
                else:
                    full_output = output

                if isinstance(target, (list, tuple)):
                    full_target = target[0]
                else:
                    full_target = target

                num_classes = full_output.shape[1]

                # Compute distance maps (on CPU, then move to device)
                dist_maps = self._compute_batch_dist_maps(full_target, num_classes)
                dist_maps = dist_maps.to(self.device, non_blocking=True)

                # One-hot encode target
                target_onehot = self._make_onehot(full_target, num_classes)

                # Compute GSL
                gsl_fn = GeneralizedSurfaceLoss(
                    apply_nonlin=softmax_helper_dim1,
                    apply_to_classes=self.gsl_classes,
                )
                gsl_loss = gsl_fn(full_output, target_onehot, dist_maps)

                l = l + gsl_weight * gsl_loss

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': l.detach().cpu().numpy()}


class nnUNetTrainerGSL_100epochs(nnUNetTrainerGSL):
    """nnUNetTrainerGSL with 100 epochs for quick experiments."""
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
