"""
Merge TSv2 outputs into nnUNet predictions, then save per-label masks.
"""

import argparse
import shutil
from multiprocessing import Pool, cpu_count
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml


def split_and_save(
    merged_data: np.ndarray,
    reference_img: nib.Nifti1Image,
    case_name: str,
    out_dir: Path,
    label_name_by_value: dict[int, str],
) -> tuple[list[Path], list[str]]:
    """Save per-label binary masks from a merged multi-class mask."""
    split_dir = out_dir

    saved_files: list[Path] = []
    debug_lines: list[str] = []
    unique_labels = sorted(int(v) for v in np.unique(merged_data) if int(v) != 0)

    for label_val in unique_labels:
        label_name = label_name_by_value.get(label_val, f"label_{label_val}")
        mask = (merged_data == label_val).astype(np.uint8)
        if np.count_nonzero(mask) == 0:
            continue

        out_path = split_dir / f"{label_name}.nii.gz"
        out_img = nib.Nifti1Image(mask, reference_img.affine, reference_img.header)
        nib.save(out_img, str(out_path))
        saved_files.append(out_path)
        debug_lines.append(f"  [SPLIT SHAPE] {label_name}: {mask.shape}")

    return saved_files, debug_lines


def process_case(args_tuple):
    """Process one case directory."""
    case_dir, out_dir, label_map, label_name_by_value = args_tuple

    case_name = case_dir.name
    nnunet_name = case_name.rsplit("_", 1)[0] if case_name.endswith("_0000") else case_name
    nnunet_file = out_dir / f"{nnunet_name}.nii.gz"

    if not nnunet_file.exists():
        return f"[SKIP] nnUNet prediction not found: {nnunet_file}"

    result_lines = [f"[MERGE] {case_name}"]

    nnunet_img = nib.load(str(nnunet_file))
    nnunet_data = np.asarray(nnunet_img.dataobj).copy()

    for task_dir in sorted(case_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        for label_file in sorted(task_dir.glob("*.nii.gz")):
            tsv2_name = label_file.stem.replace(".nii", "")

            if tsv2_name not in label_map:
                result_lines.append(f"  [SKIP] {tsv2_name} not in structure_list.yaml")
                continue

            label_val = int(label_map[tsv2_name])
            mask_img = nib.load(str(label_file))
            mask_data = np.asarray(mask_img.dataobj)

            mask_to_add = (mask_data > 0) & (nnunet_data == 0)
            nnunet_data[mask_to_add] = label_val
            result_lines.append(f"  [OK] {tsv2_name} -> label {label_val}")

    out_img = nib.Nifti1Image(nnunet_data, nnunet_img.affine, nnunet_img.header)
    nib.save(out_img, str(nnunet_file))
    result_lines.append(f"  [SAVE] {nnunet_file}")
    result_lines.append(f"  [OUT SHAPE] {nnunet_data.shape}")

    split_files, split_debug_lines = split_and_save(
        merged_data=nnunet_data,
        reference_img=nnunet_img,
        case_name=nnunet_name,
        out_dir=out_dir,
        label_name_by_value=label_name_by_value,
    )
    result_lines.append(f"  [SPLIT SAVE] {len(split_files)} files")
    result_lines.extend(split_debug_lines)

    shutil.rmtree(case_dir)
    result_lines.append(f"  [DEL] {case_dir}")

    return "\n".join(result_lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--structure_list", required=True)
    parser.add_argument(
        "--split_label_yaml",
        default="",
        help="YAML file used for split output names (default: normal_structure.yaml next to structure_list).",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU core count).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    with open(args.structure_list) as f:
        label_map = yaml.safe_load(f)["labels"]

    split_label_yaml = Path(args.split_label_yaml) if args.split_label_yaml else Path(args.structure_list).with_name("normal_structure.yaml")
    label_name_by_value: dict[int, str] = {}
    if split_label_yaml.exists():
        with open(split_label_yaml) as f:
            split_label_map = yaml.safe_load(f)["labels"]
        label_name_by_value = {int(v): k for k, v in split_label_map.items()}
        print(f"Loaded split label names from: {split_label_yaml}")
    else:
        label_name_by_value = {int(v): k for k, v in label_map.items()}
        print(f"[WARN] split label yaml not found, fallback to structure_list labels: {split_label_yaml}")

    print(f"Loaded labels from structure_list.yaml: {label_map}")

    case_dirs = [d for d in sorted(out_dir.iterdir()) if d.is_dir()]

    if not case_dirs:
        print("No case directories to process.")
        return

    n_jobs = args.n_jobs if args.n_jobs else cpu_count()
    print(f"Processing {len(case_dirs)} cases with {n_jobs} workers")

    tasks = [(case_dir, out_dir, label_map, label_name_by_value) for case_dir in case_dirs]

    with Pool(processes=n_jobs) as pool:
        results = pool.map(process_case, tasks)

    for result in results:
        print(result)

    for json_file in out_dir.glob("*.json"):
        json_file.unlink()
        print(f"[DEL JSON] {json_file}")

    print("Done.")


if __name__ == "__main__":
    main()
