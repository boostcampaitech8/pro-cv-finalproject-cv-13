from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import numpy as np
import torch
import random

class NPZDataset(Dataset):
    def __init__(
        self,
        files: List[Path],
        patch_size: Tuple[int, int, int],
        is_train: bool,
        oversample_foreground_percent: float = 0.0,
    ):
        self.files = files
        self.patch_size = patch_size
        self.is_train = is_train
        self.oversample_foreground_percent = oversample_foreground_percent

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        with np.load(path) as data:
            image = data["image"].astype(np.float32)
            label = data["label"].astype(np.int64)

        if self.is_train:
            if self.oversample_foreground_percent > 0 and random.random() < self.oversample_foreground_percent:
                image, label = _foreground_crop(image, label, self.patch_size)
            else:
                image, label = _random_crop(image, label, self.patch_size)
        else:
            image, label = _center_crop(image, label, self.patch_size)

        return torch.from_numpy(image), torch.from_numpy(label)


def _load_nifti(path: Path) -> np.ndarray:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("nibabel is required to load .nii.gz files") from exc

    img = nib.load(str(path))
    return np.asarray(img.dataobj, dtype=np.float32)


def _zscore(volume: np.ndarray) -> np.ndarray:
    mean = volume.mean()
    std = volume.std()
    return (volume - mean) / (std + 1e-8)


def _resolve_modality_path(case_dir: Path, modality: str) -> Path:
    if "{case_id}" in modality:
        return case_dir / modality.format(case_id=case_dir.name)
    if modality.startswith("__"):
        return case_dir / f"{case_dir.name}{modality}"
    return case_dir / modality


class RAMDataset(Dataset):
    def __init__(
        self,
        case_dirs: List[Path],
        labels_dir: Path,
        modalities: List[str],
        patch_size: Tuple[int, int, int],
        is_train: bool,
        oversample_foreground_percent: float = 0.0,
        cache_in_ram: bool = True,
        cached_cases: Optional[Dict[str, dict]] = None,
    ):
        self.case_dirs = case_dirs
        self.case_ids = [p.name for p in case_dirs]
        self.labels_dir = labels_dir
        self.modalities = modalities
        self.patch_size = patch_size
        self.is_train = is_train
        self.oversample_foreground_percent = oversample_foreground_percent
        self.cache_in_ram = cache_in_ram or cached_cases is not None
        self.cached_cases = cached_cases

        if self.cache_in_ram and self.cached_cases is None:
            self.cached_cases = self.cache_cases(case_dirs, labels_dir, modalities)

    @staticmethod
    def cache_cases(
        case_dirs: List[Path],
        labels_dir: Path,
        modalities: List[str],
    ) -> dict:
        cached = {}
        total_bytes = 0
        for case_dir in case_dirs:
            case_id = case_dir.name
            image, label = RAMDataset._load_case(case_dir, labels_dir, modalities)
            total_bytes += image.nbytes + label.nbytes
            cached[case_id] = {"image": image, "label": label}
        total_gb = total_bytes / (1024 ** 3)
        print(f"RAMDataset: cached {len(cached)} cases in RAM ({total_gb:.2f} GB)")
        return cached

    @staticmethod
    def _load_case(
        case_dir: Path,
        labels_dir: Path,
        modalities: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        images = []
        for modality in modalities:
            img_path = _resolve_modality_path(case_dir, modality)
            if not img_path.exists():
                raise FileNotFoundError(f"Missing modality file: {img_path}")
            img = _load_nifti(img_path)
            images.append(_zscore(img))

        image = np.stack(images, axis=0).astype(np.float32)

        label_path = labels_dir / f"{case_dir.name}.nii.gz"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label file: {label_path}")
        label = _load_nifti(label_path).astype(np.int16)
        return image, label

    def _get_case(self, case_id: str) -> Tuple[np.ndarray, np.ndarray]:
        if self.cache_in_ram and self.cached_cases is not None:
            cached = self.cached_cases[case_id]
            return cached["image"], cached["label"]

        case_dir = self.case_dirs[self.case_ids.index(case_id)]
        return self._load_case(case_dir, self.labels_dir, self.modalities)

    def get_full_case(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        case_id = self.case_ids[idx]
        image, label = self._get_case(case_id)
        return image, label

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int):
        case_id = self.case_ids[idx]
        image, label = self._get_case(case_id)

        if self.is_train:
            if self.oversample_foreground_percent > 0 and random.random() < self.oversample_foreground_percent:
                image, label = _foreground_crop(image, label, self.patch_size)
            else:
                image, label = _random_crop(image, label, self.patch_size)
        else:
            image, label = _center_crop(image, label, self.patch_size)

        image = np.ascontiguousarray(image)
        label = np.ascontiguousarray(label)
        return torch.from_numpy(image), torch.from_numpy(label)


class RawDataset(Dataset):
    def __init__(
        self,
        case_dirs: List[Path],
        labels_dir: Path,
        modalities: List[str],
        patch_size: Tuple[int, int, int],
        is_train: bool,
        oversample_foreground_percent: float = 0.0,
    ):
        self.case_dirs = case_dirs
        self.case_ids = [p.name for p in case_dirs]
        self.labels_dir = labels_dir
        self.modalities = modalities
        self.patch_size = patch_size
        self.is_train = is_train
        self.oversample_foreground_percent = oversample_foreground_percent

    @staticmethod
    def _load_case(
        case_dir: Path,
        labels_dir: Path,
        modalities: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        images = []
        for modality in modalities:
            img_path = _resolve_modality_path(case_dir, modality)
            if not img_path.exists():
                raise FileNotFoundError(f"Missing modality file: {img_path}")
            img = _load_nifti(img_path)
            images.append(_zscore(img))

        image = np.stack(images, axis=0).astype(np.float32)

        label_path = labels_dir / f"{case_dir.name}.nii.gz"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label file: {label_path}")
        label = _load_nifti(label_path).astype(np.int16)
        return image, label

    def get_full_case(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        case_dir = self.case_dirs[idx]
        return self._load_case(case_dir, self.labels_dir, self.modalities)

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int):
        case_dir = self.case_dirs[idx]
        image, label = self._load_case(case_dir, self.labels_dir, self.modalities)

        if self.is_train:
            if self.oversample_foreground_percent > 0 and random.random() < self.oversample_foreground_percent:
                image, label = _foreground_crop(image, label, self.patch_size)
            else:
                image, label = _random_crop(image, label, self.patch_size)
        else:
            image, label = _center_crop(image, label, self.patch_size)

        image = np.ascontiguousarray(image)
        label = np.ascontiguousarray(label)
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


def _foreground_crop(
    image: np.ndarray,
    label: np.ndarray,
    patch_size: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    image, label = _pad_to_patch(image, label, patch_size)
    foreground = np.argwhere(label > 0)
    if foreground.size == 0:
        return _random_crop(image, label, patch_size)

    center_d, center_h, center_w = foreground[random.randrange(len(foreground))]
    _, d, h, w = image.shape

    sd = int(center_d - patch_size[0] // 2)
    sh = int(center_h - patch_size[1] // 2)
    sw = int(center_w - patch_size[2] // 2)

    sd = max(0, min(sd, d - patch_size[0]))
    sh = max(0, min(sh, h - patch_size[1]))
    sw = max(0, min(sw, w - patch_size[2]))

    crop = (
        slice(sd, sd + patch_size[0]),
        slice(sh, sh + patch_size[1]),
        slice(sw, sw + patch_size[2]),
    )
    return image[:, crop[0], crop[1], crop[2]], label[crop[0], crop[1], crop[2]]
