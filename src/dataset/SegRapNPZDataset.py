from torch.utils.data import Dataset
from typing import List, Tuple
from pathlib import Path
import numpy as np
import torch
import random

class SegRapNPZDataset(Dataset):
    def __init__(
        self,
        files: List[Path],
        patch_size: Tuple[int, int, int],
        is_train: bool,
    ):
        self.files = files
        self.patch_size = patch_size
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        with np.load(path) as data:
            image = data["image"].astype(np.float32)
            label = data["label"].astype(np.int64)

        if self.is_train:
            image, label = _random_crop(image, label, self.patch_size)
        else:
            image, label = _center_crop(image, label, self.patch_size)

        return torch.from_numpy(image), torch.from_numpy(label)


def _center_crop(
    image: np.ndarray,
    label: np.ndarray,
    patch_size: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    image, label = _pad_to_patch(image, label, patch_size)
    _, d, h, w = image.shape
    sd = max(0, (d - patch_size[0]) // 2)
    sh = max(0, (h - patch_size[1]) // 2)
    sw = max(0, (w - patch_size[2]) // 2)
    crop = (
        slice(sd, sd + patch_size[0]),
        slice(sh, sh + patch_size[1]),
        slice(sw, sw + patch_size[2]),
    )
    return image[:, crop[0], crop[1], crop[2]], label[crop[0], crop[1], crop[2]]


def _pad_to_patch(
    image: np.ndarray,
    label: np.ndarray,
    patch_size: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    _, d, h, w = image.shape
    pd = max(0, patch_size[0] - d)
    ph = max(0, patch_size[1] - h)
    pw = max(0, patch_size[2] - w)

    if pd == 0 and ph == 0 and pw == 0:
        return image, label

    pad_d = (pd // 2, pd - pd // 2)
    pad_h = (ph // 2, ph - ph // 2)
    pad_w = (pw // 2, pw - pw // 2)

    image = np.pad(
        image,
        ((0, 0), pad_d, pad_h, pad_w),
        mode="constant",
        constant_values=0,
    )
    label = np.pad(
        label,
        (pad_d, pad_h, pad_w),
        mode="constant",
        constant_values=0,
    )
    return image, label


def _random_crop(
    image: np.ndarray,
    label: np.ndarray,
    patch_size: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    image, label = _pad_to_patch(image, label, patch_size)
    _, d, h, w = image.shape
    sd = random.randint(0, d - patch_size[0])
    sh = random.randint(0, h - patch_size[1])
    sw = random.randint(0, w - patch_size[2])
    crop = (
        slice(sd, sd + patch_size[0]),
        slice(sh, sh + patch_size[1]),
        slice(sw, sw + patch_size[2]),
    )
    return image[:, crop[0], crop[1], crop[2]], label[crop[0], crop[1], crop[2]]