"""CT-only tumor segmentation inference wrapper."""
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="CT-only tumor segmentation")
    parser.add_argument("ct_path", type=Path, help="Input CT NIfTI file")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument("--model-folder", type=Path, default=Path("/app/model"))
    parser.add_argument("--checkpoint", default="checkpoint_final.pth")
    parser.add_argument("--folds", default="0")
    args = parser.parse_args()

    ct_path = args.ct_path
    if not ct_path.exists():
        raise FileNotFoundError(f"CT file not found: {ct_path}")

    # 체크포인트 존재 확인
    fold = args.folds.split()[0]
    chk_path = args.model_folder / f"fold_{fold}" / args.checkpoint
    if not chk_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {chk_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # case_id 추출
    name = ct_path.name
    if name.endswith("__CT.nii.gz"):
        case_id = name[:-len("__CT.nii.gz")]
    elif name.endswith(".nii.gz"):
        stem = name[:-len(".nii.gz")]
        case_id = stem[:-len("_0000")] if stem.endswith("_0000") else stem
    else:
        case_id = ct_path.stem

    nnunet_predict = shutil.which("nnUNetv2_predict_from_modelfolder")
    if not nnunet_predict:
        raise FileNotFoundError("nnUNetv2_predict_from_modelfolder not found in PATH")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        pred_dir = Path(tmpdir) / "pred"
        input_dir.mkdir()
        pred_dir.mkdir()

        # nnUNet 입력 포맷: {case_id}_0000.nii.gz
        shutil.copy2(ct_path, input_dir / f"{case_id}_0000.nii.gz")

        cmd = [
            nnunet_predict,
            "-m", str(args.model_folder),
            "-i", str(input_dir),
            "-o", str(pred_dir),
            "-f", *args.folds.split(),
            "-chk", args.checkpoint,
        ]
        print(f"[CT-only] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        pred_file = pred_dir / f"{case_id}.nii.gz"
        if not pred_file.exists():
            raise FileNotFoundError(f"Prediction not found: {pred_file}")

        import nibabel as nib
        import numpy as np
        img = nib.load(str(pred_file))
        data = (img.get_fdata().astype(np.uint8) > 0).astype(np.uint8)
        out_img = nib.Nifti1Image(data, img.affine, img.header)
        dest = args.output_dir / f"{case_id}.nii.gz"
        nib.save(out_img, str(dest))
        print(f"[CT-only] Output: {dest}")


if __name__ == "__main__":
    main()
