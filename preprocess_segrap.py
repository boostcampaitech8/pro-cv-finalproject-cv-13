from __future__ import annotations

import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm


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


def _iter_cases(raw_dir: Path):
    for case_dir in sorted(raw_dir.iterdir()):
        if case_dir.is_dir():
            yield case_dir


def _save_case(
    case_dir: Path,
    labels_dir: Path,
    modalities: list[str],
    output_dir: Path,
):
    case_id = case_dir.name
    images = []
    for modality in modalities:
        img_path = case_dir / modality
        if not img_path.exists():
            raise FileNotFoundError(f"Missing modality file: {img_path}")
        img = _load_nifti(img_path)
        images.append(_zscore(img))

    image = np.stack(images, axis=0).astype(np.float32)

    label_path = labels_dir / f"{case_id}.nii.gz"
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label file: {label_path}")
    label = _load_nifti(label_path).astype(np.int16)

    case_out = output_dir / case_id
    case_out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(case_out / f"{case_id}.npz", image=image, label=label)


def _save_case_star(args):
    return _save_case(*args)


def preprocess_segrap_dataset(
    raw_dir: Path,
    labels_dir: Path,
    modalities: list[str],
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = list(_iter_cases(raw_dir))
    args = [(c, labels_dir, modalities, output_dir) for c in cases]

    num_workers = 4
    with Pool(num_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(_save_case_star, args, chunksize=1),
            total=len(args),
            desc="Preprocessing",
            unit="case",
        ):
            pass

    print(f"Preprocessing complete. Saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess SegRap 3D NIfTI volumes into npz files."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("segrap/SegRap2023_Training_Set_120cases-002/SegRap2023_Training_Set_120cases"),
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("segrap/SegRap2023_Training_Set_120cases_OneHot_Labels/Task001"),
    )
    parser.add_argument(
        "--modalities",
        type=str,
        nargs="+",
        default=["image.nii.gz", "image_contrast.nii.gz"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preprocessed_segrap"),
    )
    args = parser.parse_args()

    preprocess_segrap_dataset(
        raw_dir=args.raw_dir,
        labels_dir=args.labels_dir,
        modalities=args.modalities,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
