from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def log_time(step: str, seconds: float) -> None:
    print(f"[TIMING] {step}: {seconds:.2f} seconds")

try:
    import SimpleITK as sitk
except ImportError as exc:
    raise ImportError("SimpleITK is required for Dice computation") from exc


def _infer_case_id(ct_path: Path) -> str:
    name = ct_path.name
    if name.endswith("__CT.nii.gz"):
        return name[: -len("__CT.nii.gz")]
    if name.endswith(".nii.gz"):
        stem = name[: -len(".nii.gz")]
        if stem.endswith("_0000"):
            stem = stem[: -len("_0000")]
        return stem
    return ct_path.stem


def _find_ct_pt(input_path: Path, pt_path: Path | None) -> tuple[Path, Path | None]:
    if input_path.is_dir():
        ct_candidates = sorted(input_path.glob("*__CT.nii.gz"))
        if len(ct_candidates) != 1:
            raise FileNotFoundError(f"Expected 1 CT file in {input_path}, found {len(ct_candidates)}")
        ct_path = ct_candidates[0]
        if pt_path is None:
            pt_candidates = sorted(input_path.glob("*__PT.nii.gz"))
            if len(pt_candidates) == 1:
                pt_path = pt_candidates[0]
        return ct_path, pt_path

    if not input_path.is_file():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    ct_path = input_path
    if pt_path is None:
        pt_candidates = sorted(ct_path.parent.glob("*__PT.nii.gz"))
        if len(pt_candidates) == 1:
            pt_path = pt_candidates[0]
    return ct_path, pt_path


def _find_label(input_path: Path) -> Path | None:
    folder = input_path if input_path.is_dir() else input_path.parent
    candidates = [
        p
        for p in folder.glob("*.nii.gz")
        if not p.name.endswith("__CT.nii.gz") and not p.name.endswith("__PT.nii.gz")
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _ensure_empty_output(output_dir: Path) -> None:
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise RuntimeError(f"Output folder must be empty: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)


def _align_to_shape(volume: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    if volume.shape == target_shape:
        return volume

    d, h, w = volume.shape
    td, th, tw = target_shape

    pad_d = max(0, td - d)
    pad_h = max(0, th - h)
    pad_w = max(0, tw - w)
    if pad_d or pad_h or pad_w:
        pad = (
            pad_d // 2,
            pad_d - pad_d // 2,
            pad_h // 2,
            pad_h - pad_h // 2,
            pad_w // 2,
            pad_w - pad_w // 2,
        )
        volume = np.pad(volume, pad, mode="constant", constant_values=0)
        d, h, w = volume.shape

    sd = max(0, (d - td) // 2)
    sh = max(0, (h - th) // 2)
    sw = max(0, (w - tw) // 2)
    return volume[sd : sd + td, sh : sh + th, sw : sw + tw]


def _dice_per_label(pred: np.ndarray, gt: np.ndarray, labels: list[int]) -> dict[int, float]:
    scores: dict[int, float] = {}
    for label in labels:
        pred_mask = pred == label
        gt_mask = gt == label
        intersection = int(np.count_nonzero(pred_mask & gt_mask))
        denom = int(np.count_nonzero(pred_mask)) + int(np.count_nonzero(gt_mask))
        scores[label] = 1.0 if denom == 0 else (2.0 * intersection / denom)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run preprocessing + nnUNetv2_predict + restore for a single case."
    )
    parser.add_argument("input_path", type=Path, help="CT file path or case folder.")
    parser.add_argument("output_dir", type=Path, help="Output folder.")
    parser.add_argument(
        "--model-folder",
        required=True,
        type=Path,
        help="Trained model folder (e.g., nnUNet_results/.../Trainer__Plans__3d_fullres).",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Checkpoint filename or path (default: checkpoints/checkpoint_best.pth).",
    )
    parser.add_argument("--pt-path", default="", help="PT file path (optional).")
    parser.add_argument(
        "--label-path",
        default="",
        help="Ground-truth label path (optional). If omitted, auto-detect in input folder.",
    )
    parser.add_argument(
        "--folds",
        default="0",
        help='Folds to use (e.g., "0", "0 1 2 3 4", or "all"). Default: 0.',
    )
    parser.add_argument(
        "--nnunet-predict",
        default="",
        help="Path to nnUNetv2_predict_from_modelfolder executable (optional).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    default_checkpoint = script_dir / "checkpoints" / "checkpoint_best.pth"
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
    else:
        checkpoint = default_checkpoint

    if not args.model_folder.is_dir():
        raise FileNotFoundError(f"Model folder not found: {args.model_folder}")

    pt_path = Path(args.pt_path) if args.pt_path else None
    label_path = Path(args.label_path) if args.label_path else None
    ct_path, pt_path = _find_ct_pt(args.input_path, pt_path)
    if label_path is None:
        label_path = _find_label(args.input_path)

    _ensure_empty_output(args.output_dir)

    nnunet_predict = args.nnunet_predict or shutil.which("nnUNetv2_predict_from_modelfolder")
    if not nnunet_predict:
        raise FileNotFoundError(
            "nnUNetv2_predict_from_modelfolder not found in PATH. Set --nnunet-predict if needed."
        )

    case_id = _infer_case_id(ct_path)

    pipeline_start = time.time()
    print("[TIMING] ========== 종양 분할 시작 ==========")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        prep_dir = tmp_path / "imagesTs"
        roi_dir = tmp_path / "roi"
        pred_dir = tmp_path / "pred"

        preprocess_py = script_dir / "preprocess_case.py"
        restore_py = script_dir / "restore_case.py"

        # Step 1: Preprocess
        preprocess_start = time.time()
        preprocess_cmd = [
            sys.executable,
            str(preprocess_py),
            "--ct",
            str(ct_path),
            "--case-id",
            case_id,
            "--output-images",
            str(prep_dir),
            "--roi-dir",
            str(roi_dir),
            "--crop",
        ]
        if pt_path is not None:
            preprocess_cmd += ["--pt", str(pt_path)]
        subprocess.run(preprocess_cmd, check=True)
        log_time("Tumor-Preprocess", time.time() - preprocess_start)

        # Step 2: nnUNet Predict
        predict_start = time.time()
        predict_cmd = [
            str(nnunet_predict),
            "-m",
            str(args.model_folder),
            "-i",
            str(prep_dir),
            "-o",
            str(pred_dir),
        ]
        if checkpoint:
            predict_cmd += ["-chk", str(checkpoint.name)]
        if args.folds:
            predict_cmd += ["-f"] + args.folds.split()
        subprocess.run(predict_cmd, check=True)
        log_time("Tumor-nnUNet", time.time() - predict_start)

        pred_path = pred_dir / f"{case_id}.nii.gz"
        if not pred_path.exists():
            raise FileNotFoundError(f"Prediction not found: {pred_path}")

        # Step 3: Restore
        restore_start = time.time()
        restore_cmd = [
            sys.executable,
            str(restore_py),
            "--ct",
            str(ct_path),
            "--pred",
            str(pred_path),
            "--roi",
            str(roi_dir / f"{case_id}_ROI.npz"),
            "--output",
            str(args.output_dir / f"{case_id}.nii.gz"),
        ]
        if pt_path is not None:
            restore_cmd += ["--pt", str(pt_path)]
        subprocess.run(restore_cmd, check=True)
        log_time("Tumor-Restore", time.time() - restore_start)

        if label_path is not None and label_path.exists():
            pred_img = sitk.ReadImage(str(args.output_dir / f"{case_id}.nii.gz"))
            gt_img = sitk.ReadImage(str(label_path))
            pred = sitk.GetArrayFromImage(pred_img).astype(np.int16)
            gt = sitk.GetArrayFromImage(gt_img).astype(np.int16)
            pred = _align_to_shape(pred, gt.shape)

            labels = sorted([int(x) for x in np.unique(gt) if int(x) != 0])
            if not labels:
                labels = [0]
            scores = _dice_per_label(pred, gt, labels)

            mean_dice = float(np.mean(list(scores.values())))
            print("Dice scores:")
            for k in sorted(scores):
                print(f"  label {k}: {scores[k]:.4f}")
            print(f"  mean: {mean_dice:.4f}")

    log_time("Tumor-Total", time.time() - pipeline_start)
    print("[TIMING] ========== 종양 분할 완료 ==========")


if __name__ == "__main__":
    main()
