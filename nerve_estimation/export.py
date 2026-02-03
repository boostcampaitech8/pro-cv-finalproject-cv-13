"""신경 추정 결과를 3D Slicer 호환 형식으로 내보내기."""

import json
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, Union


def create_tube_mask(
    points: np.ndarray,
    shape: tuple,
    radius_mm: float,
    spacing: np.ndarray,
) -> np.ndarray:
    """경로를 따라 튜브 형태 마스크 생성."""
    mask = np.zeros(shape, dtype=np.uint8)
    if len(points) == 0:
        return mask

    radius_voxels = radius_mm / spacing

    for point in points:
        min_coords = np.maximum(0, np.floor(point - radius_voxels).astype(int))
        max_coords = np.minimum(shape, np.ceil(point + radius_voxels + 1).astype(int))

        x_range = np.arange(min_coords[0], max_coords[0])
        y_range = np.arange(min_coords[1], max_coords[1])
        z_range = np.arange(min_coords[2], max_coords[2])

        if len(x_range) == 0 or len(y_range) == 0 or len(z_range) == 0:
            continue

        xx, yy, zz = np.meshgrid(x_range, y_range, z_range, indexing='ij')

        dist_mm = np.sqrt(
            ((xx - point[0]) * spacing[0]) ** 2 +
            ((yy - point[1]) * spacing[1]) ** 2 +
            ((zz - point[2]) * spacing[2]) ** 2
        )

        local_mask = dist_mm <= radius_mm
        mask[min_coords[0]:max_coords[0],
             min_coords[1]:max_coords[1],
             min_coords[2]:max_coords[2]] |= local_mask.astype(np.uint8)

    return mask


def create_sphere_mask(
    center: np.ndarray,
    shape: tuple,
    radius_mm: float,
    spacing: np.ndarray,
) -> np.ndarray:
    """구형 마스크 생성 (danger zone용)."""
    return create_tube_mask(center.reshape(1, 3), shape, radius_mm, spacing)


def export_from_json(
    nerve_results_path: Union[str, Path],
    reference_nifti: Union[str, Path],
    output_dir: Union[str, Path],
    use_uncertainty_as_radius: bool = True,
    default_radius_mm: float = 3.0,
) -> Dict[str, Path]:
    """JSON 파일에서 신경 결과를 NIfTI 마스크로 내보내기."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(nerve_results_path) as f:
        results = json.load(f)

    ref_nii = nib.load(str(reference_nifti))
    ref_shape = ref_nii.shape[:3]
    affine = ref_nii.affine
    spacing = np.abs(np.diag(affine)[:3])

    saved_files = {}
    combined_mask = np.zeros(ref_shape, dtype=np.uint8)
    nerve_labels = {}
    label_counter = 1

    for nerve_data in results.get('nerves', []):
        nerve = nerve_data['nerve']
        side = nerve_data['side']
        nerve_type = nerve_data['type']
        name = f"{nerve}_{side}"

        radius_mm = nerve_data.get('uncertainty_mm', default_radius_mm) if use_uncertainty_as_radius else default_radius_mm

        if nerve_type == 'pathway':
            points = nerve_data.get('pathway_voxels')
            if not points:
                continue
            mask = create_tube_mask(np.array(points), ref_shape, radius_mm, spacing)
        elif nerve_type == 'danger_zone':
            center = nerve_data.get('center_voxels')
            if center is None:
                continue
            mask = create_sphere_mask(np.array(center), ref_shape, radius_mm, spacing)
        else:
            continue

        output_path = output_dir / f"{name}_mask.nii.gz"
        nii = nib.Nifti1Image(mask, affine)
        nib.save(nii, str(output_path))
        saved_files[name] = output_path
        print(f"Saved: {output_path}")

        combined_mask[mask > 0] = label_counter
        nerve_labels[label_counter] = name
        label_counter += 1

    if label_counter > 1:
        combined_path = output_dir / "all_nerves_mask.nii.gz"
        nii = nib.Nifti1Image(combined_mask, affine)
        nib.save(nii, str(combined_path))
        saved_files["combined"] = combined_path
        print(f"Saved: {combined_path}")

        labels_path = output_dir / "nerve_labels.json"
        with open(labels_path, 'w') as f:
            json.dump({str(k): v for k, v in nerve_labels.items()}, f, indent=2)
        saved_files["labels"] = labels_path
        print(f"Saved: {labels_path}")

    return saved_files


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python -m nerve_estimation.export <nerve_results.json> <reference.nii.gz> <output_dir>")
        sys.exit(1)

    export_from_json(sys.argv[1], sys.argv[2], sys.argv[3])
