from __future__ import annotations

import argparse
import json
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm


def strip_nii_gz(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def load_label_volume(path: Path, threshold: float) -> tuple[np.ndarray, "nib.Nifti1Image", int]:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("nibabel is required: pip install nibabel") from exc

    img = nib.load(str(path))
    data = np.asarray(img.get_fdata())
    if data.ndim == 4:
        labels = np.zeros(data.shape[:3], dtype=np.int16)
        overlap = 0
        for idx in range(data.shape[-1]):
            mask = data[..., idx] > threshold
            if np.any(mask):
                overlap += int(np.count_nonzero(labels[mask]))
                labels[mask] = idx + 1
        return labels, img, overlap
    return data.astype(np.int16), img, 0


def build_unified_labels(
    task1_path: Path,
    task2_path: Path,
    offset: int,
    threshold: float,
) -> tuple[np.ndarray, "nib.Nifti1Image", int]:
    task1_labels, task1_img, overlap1 = load_label_volume(task1_path, threshold)
    task2_labels, task2_img, overlap2 = load_label_volume(task2_path, threshold)

    if task1_labels.shape != task2_labels.shape:
        raise ValueError(f"Shape mismatch: {task1_path} {task1_labels.shape} vs {task2_path} {task2_labels.shape}")

    combined = task1_labels.copy()
    task2_mask = task2_labels > 0
    combined[task2_mask] = (task2_labels[task2_mask] + offset).astype(np.int16)
    return combined, task1_img, overlap1 + overlap2


def collect_cases(labels_dir: Path) -> dict[str, Path]:
    return {strip_nii_gz(p): p for p in labels_dir.glob("*.nii.gz")}


def build_unified_label_map(task1_json: Path, task2_json: Path) -> dict[str, str]:
    task1 = json.loads(task1_json.read_text(encoding="utf-8"))
    task2 = json.loads(task2_json.read_text(encoding="utf-8"))

    labels1 = {int(k): v for k, v in task1.get("labels", {}).items()}
    labels2 = {int(k): v for k, v in task2.get("labels", {}).items()}

    offset = max(labels1.keys()) if labels1 else 0
    unified = {str(k): v for k, v in labels1.items()}
    for k, v in labels2.items():
        unified[str(k + offset)] = v
    return unified


def write_dataset_template(
    task1_json: Path,
    task2_json: Path,
    output_path: Path,
) -> int:
    task1 = json.loads(task1_json.read_text(encoding="utf-8"))
    labels1 = {int(k): v for k, v in task1.get("labels", {}).items()}
    offset = max(labels1.keys()) if labels1 else 0

    unified = {
        "name": "SegRap2023(Task001+Task002)",
        "description": "Unified labels from Task001 and Task002",
        "reference": task1.get("reference", ""),
        "licence": task1.get("licence", ""),
        "release": task1.get("release", ""),
        "tensorImageSize": task1.get("tensorImageSize", "3D"),
        "modality": task1.get("modality", {"0": "CT", "1": "CT"}),
        "labels": build_unified_label_map(task1_json, task2_json),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(unified, indent=4), encoding="utf-8")
    return offset


def _save_unified_case(args: tuple[Path, Path, Path, int, float]) -> int:
    task1_path, task2_path, output_path, offset, threshold = args
    combined, ref_img, overlaps = build_unified_labels(
        task1_path,
        task2_path,
        offset=offset,
        threshold=threshold,
    )

    header = ref_img.header.copy()
    header.set_data_dtype(np.int16)
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("nibabel is required: pip install nibabel") from exc
    nib.save(nib.Nifti1Image(combined, ref_img.affine, header), str(output_path))
    return overlaps


def main() -> None:
    parser = argparse.ArgumentParser(description="Unify Task001/Task002 labels into a single label volume.")
    parser.add_argument(
        "--task1-dir",
        type=Path,
        default=Path("segrap/SegRap2023_Training_Set_120cases_OneHot_Labels/Task001"),
    )
    parser.add_argument(
        "--task2-dir",
        type=Path,
        default=Path("segrap/SegRap2023_Training_Set_120cases_OneHot_Labels/Task002"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./segrap_unified/labelsTr"),
    )
    parser.add_argument(
        "--task1-json",
        type=Path,
        default=Path("segrap/dataset_task001.json"),
    )
    parser.add_argument(
        "--task2-json",
        type=Path,
        default=Path("segrap/dataset_task002.json"),
    )
    parser.add_argument(
        "--dataset-template-out",
        type=Path,
        default=Path("./segrap/dataset_unified.json"),
        help="Path to write the unified dataset template json.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for one-hot channels.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, cpu_count() // 2),
        help="Number of parallel worker processes.",
    )
    args = parser.parse_args()

    offset = write_dataset_template(args.task1_json, args.task2_json, args.dataset_template_out)

    task1_cases = collect_cases(args.task1_dir)
    task2_cases = collect_cases(args.task2_dir)
    common_cases = sorted(set(task1_cases) & set(task2_cases))
    if not common_cases:
        raise RuntimeError("No overlapping cases found between Task001 and Task002.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    job_args = [
        (
            task1_cases[case_id],
            task2_cases[case_id],
            args.output_dir / f"{case_id}.nii.gz",
            offset,
            args.threshold,
        )
        for case_id in common_cases
    ]

    total_overlaps = 0
    with Pool(processes=max(1, args.workers)) as pool:
        for overlaps in tqdm(
            pool.imap_unordered(_save_unified_case, job_args, chunksize=1),
            total=len(job_args),
            desc="Unifying",
            unit="case",
        ):
            total_overlaps += overlaps

    if total_overlaps > 0:
        print(f"Warning: {total_overlaps} overlapping voxels detected in one-hot channels.")

    print(f"Unified labels written to: {args.output_dir}")
    print(f"Dataset template written to: {args.dataset_template_out}")


if __name__ == "__main__":
    main()
