from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


@dataclass
class AugmentConfig:
    p_flip: float = 0.5
    flip_axes: Tuple[int, int, int] = (0, 1, 2)
    p_rotation: float = 0.2
    rotation_range: Tuple[float, float] = (-3.14159, 3.14159)
    p_scale: float = 0.2
    scale_range: Tuple[float, float] = (0.7, 1.4)
    p_elastic: float = 0.2
    elastic_alpha: float = 90.0
    elastic_sigma: float = 9.0
    p_noise: float = 0.1
    noise_variance: Tuple[float, float] = (0.0, 0.1)
    per_channel_noise: bool = True
    p_blur: float = 0.2
    blur_sigma: Tuple[float, float] = (0.5, 1.0)
    p_brightness: float = 0.15
    brightness_range: Tuple[float, float] = (0.75, 1.25)
    p_brightness_add: float = 0.1
    brightness_add_range: Tuple[float, float] = (-0.1, 0.1)
    p_contrast: float = 0.15
    contrast_range: Tuple[float, float] = (0.75, 1.25)
    p_gamma: float = 0.3
    gamma_range: Tuple[float, float] = (0.7, 1.5)
    p_lowres: float = 0.25
    lowres_scale: Tuple[float, float] = (0.5, 1.0)


class NnUNetAugment3D:
    def __init__(self, config: AugmentConfig):
        self.cfg = config

    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 5:
            raise ValueError(f"Expected images shape (B, C, D, H, W), got {images.shape}")
        if labels.ndim != 4:
            raise ValueError(f"Expected labels shape (B, D, H, W), got {labels.shape}")

        batch = images.shape[0]
        out_images = []
        out_labels = []
        for i in range(batch):
            img = images[i]
            lab = labels[i]

            img, lab = self._random_flip(img, lab)
            img, lab = self._random_spatial(img, lab)
            img, lab = self._random_elastic(img, lab)
            img = self._random_noise(img)
            img = self._random_blur(img)
            img = self._random_brightness(img)
            img = self._random_brightness_add(img)
            img = self._random_contrast(img)
            img = self._random_gamma(img)
            img = self._random_lowres(img)

            out_images.append(img)
            out_labels.append(lab)

        return torch.stack(out_images, dim=0), torch.stack(out_labels, dim=0)

    def _random_flip(self, image: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() > self.cfg.p_flip:
            return image, label
        img_dims = []
        lbl_dims = []
        for axis in self.cfg.flip_axes:
            if torch.rand(1).item() < 0.5:
                img_dims.append(axis + 1)
                lbl_dims.append(axis)
        if img_dims:
            image = torch.flip(image, dims=img_dims)
            label = torch.flip(label, dims=lbl_dims)
        return image, label

    def _random_spatial(self, image: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        do_rot = torch.rand(1).item() < self.cfg.p_rotation
        do_scale = torch.rand(1).item() < self.cfg.p_scale
        if not (do_rot or do_scale):
            return image, label

        angle_x = angle_y = angle_z = 0.0
        if do_rot:
            angle_x = float(torch.empty(1, device=image.device).uniform_(*self.cfg.rotation_range).item())
            angle_y = float(torch.empty(1, device=image.device).uniform_(*self.cfg.rotation_range).item())
            angle_z = float(torch.empty(1, device=image.device).uniform_(*self.cfg.rotation_range).item())

        scale = 1.0
        if do_scale:
            scale = float(torch.empty(1, device=image.device).uniform_(*self.cfg.scale_range).item())

        rot_x = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, torch.cos(torch.tensor(angle_x)), -torch.sin(torch.tensor(angle_x))],
                [0.0, torch.sin(torch.tensor(angle_x)), torch.cos(torch.tensor(angle_x))],
            ],
            device=image.device,
            dtype=image.dtype,
        )
        rot_y = torch.tensor(
            [
                [torch.cos(torch.tensor(angle_y)), 0.0, torch.sin(torch.tensor(angle_y))],
                [0.0, 1.0, 0.0],
                [-torch.sin(torch.tensor(angle_y)), 0.0, torch.cos(torch.tensor(angle_y))],
            ],
            device=image.device,
            dtype=image.dtype,
        )
        rot_z = torch.tensor(
            [
                [torch.cos(torch.tensor(angle_z)), -torch.sin(torch.tensor(angle_z)), 0.0],
                [torch.sin(torch.tensor(angle_z)), torch.cos(torch.tensor(angle_z)), 0.0],
                [0.0, 0.0, 1.0],
            ],
            device=image.device,
            dtype=image.dtype,
        )
        rot = rot_z @ rot_y @ rot_x
        rot = rot / scale

        theta = torch.zeros((1, 3, 4), device=image.device, dtype=image.dtype)
        theta[:, :3, :3] = rot

        img_5d = image.unsqueeze(0)
        lab_5d = label.unsqueeze(0).unsqueeze(0).float()
        grid = F.affine_grid(theta, img_5d.size(), align_corners=False)
        img_out = F.grid_sample(img_5d, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        lab_out = F.grid_sample(lab_5d, grid, mode="nearest", padding_mode="zeros", align_corners=False)
        return img_out.squeeze(0), lab_out.squeeze(0).squeeze(0).long()

    def _random_elastic(self, image: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() > self.cfg.p_elastic:
            return image, label
        _, d, h, w = image.shape
        device = image.device

        displacement = torch.randn((1, 3, d, h, w), device=device, dtype=image.dtype)
        kernel = int(max(3, round(self.cfg.elastic_sigma)))
        if kernel % 2 == 0:
            kernel += 1
        pad = kernel // 2
        displacement = F.avg_pool3d(
            F.pad(displacement, (pad, pad, pad, pad, pad, pad), mode="reflect"),
            kernel_size=kernel,
            stride=1,
        )
        displacement = displacement * self.cfg.elastic_alpha / max(d, h, w)

        zz, yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, d, device=device, dtype=image.dtype),
            torch.linspace(-1, 1, h, device=device, dtype=image.dtype),
            torch.linspace(-1, 1, w, device=device, dtype=image.dtype),
            indexing="ij",
        )
        grid = torch.stack((xx, yy, zz), dim=-1).unsqueeze(0)
        grid = grid + displacement.permute(0, 2, 3, 4, 1)

        img_5d = image.unsqueeze(0)
        lab_5d = label.unsqueeze(0).unsqueeze(0).float()
        img_out = F.grid_sample(img_5d, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        lab_out = F.grid_sample(lab_5d, grid, mode="nearest", padding_mode="zeros", align_corners=True)
        return img_out.squeeze(0), lab_out.squeeze(0).squeeze(0).long()

    def _random_noise(self, image: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.cfg.p_noise:
            return image
        variance = torch.empty(1, device=image.device).uniform_(
            self.cfg.noise_variance[0], self.cfg.noise_variance[1]
        ).item()
        if self.cfg.per_channel_noise:
            noise = torch.randn_like(image)
        else:
            noise = torch.randn(1, *image.shape[1:], device=image.device, dtype=image.dtype)
        return image + noise * variance**0.5

    def _random_blur(self, image: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.cfg.p_blur:
            return image
        sigma = torch.empty(1, device=image.device).uniform_(
            self.cfg.blur_sigma[0], self.cfg.blur_sigma[1]
        ).item()
        kernel_size = int(4 * sigma + 0.5) * 2 + 1
        kernel_size = max(3, kernel_size)
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        padding = kernel_size // 2

        x = torch.arange(kernel_size, device=image.device, dtype=image.dtype) - padding
        kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        c = image.shape[0]
        img_5d = image.unsqueeze(0)
        kx = kernel_1d.view(1, 1, 1, 1, -1).repeat(c, 1, 1, 1, 1)
        ky = kernel_1d.view(1, 1, 1, -1, 1).repeat(c, 1, 1, 1, 1)
        kz = kernel_1d.view(1, 1, -1, 1, 1).repeat(c, 1, 1, 1, 1)

        img_5d = F.pad(img_5d, (padding, padding, 0, 0, 0, 0), mode="reflect")
        img_5d = F.conv3d(img_5d, kx, groups=c)
        img_5d = F.pad(img_5d, (0, 0, padding, padding, 0, 0), mode="reflect")
        img_5d = F.conv3d(img_5d, ky, groups=c)
        img_5d = F.pad(img_5d, (0, 0, 0, 0, padding, padding), mode="reflect")
        img_5d = F.conv3d(img_5d, kz, groups=c)
        return img_5d.squeeze(0)

    def _random_brightness(self, image: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.cfg.p_brightness:
            return image
        multiplier = torch.empty(1, device=image.device).uniform_(
            self.cfg.brightness_range[0], self.cfg.brightness_range[1]
        ).item()
        return image * multiplier

    def _random_brightness_add(self, image: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.cfg.p_brightness_add:
            return image
        add_val = torch.empty(1, device=image.device).uniform_(
            self.cfg.brightness_add_range[0], self.cfg.brightness_add_range[1]
        ).item()
        return image + add_val

    def _random_contrast(self, image: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.cfg.p_contrast:
            return image
        factor = torch.empty(1, device=image.device).uniform_(
            self.cfg.contrast_range[0], self.cfg.contrast_range[1]
        ).item()
        mean = image.mean()
        out = (image - mean) * factor + mean
        return out.clamp(image.min(), image.max())

    def _random_gamma(self, image: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.cfg.p_gamma:
            return image
        gamma = torch.empty(1, device=image.device).uniform_(
            self.cfg.gamma_range[0], self.cfg.gamma_range[1]
        ).item()
        minm = image.min()
        rnge = image.max() - minm
        out = torch.pow((image - minm) / rnge.clamp(min=1e-7), gamma) * rnge + minm
        return out

    def _random_lowres(self, image: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.cfg.p_lowres:
            return image
        scale = torch.empty(1, device=image.device).uniform_(
            self.cfg.lowres_scale[0], self.cfg.lowres_scale[1]
        ).item()
        if scale >= 0.999:
            return image
        _, d, h, w = image.shape
        new_d = max(1, int(d * scale))
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))
        img_5d = image.unsqueeze(0)
        down = F.interpolate(img_5d, size=(new_d, new_h, new_w), mode="nearest")
        up = F.interpolate(down, size=(d, h, w), mode="trilinear", align_corners=False)
        return up.squeeze(0)
