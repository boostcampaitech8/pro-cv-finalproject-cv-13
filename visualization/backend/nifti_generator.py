"""
NIfTI Multi-Label Labelmap Generator

여러 binary NIfTI mask를 합쳐서 단일 multi-label NIfTI 파일 생성.
Cornerstone3D의 LABELMAP segmentation representation에서 사용.

3D Slicer와 동일한 파이프라인:
NIfTI labelmap (regular grid) → polySeg marching cubes → surface → slice plane intersection
"""

import json
import logging
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

try:
    from colors import (
        NIFTI_STRUCTURE_COLORS_RGBA,
        normalize_structure_name,
        get_color_for_structure_rgba,
    )
except ImportError:
    from .colors import (
        NIFTI_STRUCTURE_COLORS_RGBA,
        normalize_structure_name,
        get_color_for_structure_rgba,
    )

logger = logging.getLogger(__name__)


def resample_mask_to_reference(
    mask_img: nib.Nifti1Image,
    ref_img: nib.Nifti1Image,
) -> np.ndarray:
    """Resample a mask to match reference image dimensions.

    IMPORTANT: Uses nearest-neighbor interpolation (order=0) to preserve
    integer label values. Linear/cubic interpolation would mix label values
    and create artifacts.

    Args:
        mask_img: Source mask NIfTI image
        ref_img: Reference NIfTI image (target dimensions/affine)

    Returns:
        Resampled mask data as numpy array matching ref_img dimensions
    """
    # Use nibabel's resample_from_to with order=0 (nearest neighbor)
    # This is critical for labelmap - integer labels must not be interpolated
    resampled_img = resample_from_to(mask_img, ref_img, order=0)
    return resampled_img.get_fdata()


def create_multilabel_nifti(
    reference_nifti_path: str,
    mask_paths: Dict[str, str],
    output_path: str,
    priority_prefixes: List[str] = None,
) -> Tuple[str, Dict[str, int], Dict[int, List[int]]]:
    """
    여러 binary mask를 합쳐서 multi-label NIfTI 생성.

    Args:
        reference_nifti_path: 참조 NIfTI 파일 경로 (affine/shape 사용)
        mask_paths: 구조물명 → mask 파일 경로 딕셔너리
        output_path: 출력 NIfTI 경로 (.nii.gz)
        priority_prefixes: 우선순위 높은 구조물 prefix 리스트 (나중에 덮어씀)
                          기본값: ["nerve_", "tumor"]

    Returns:
        (output_path, label_map, color_map)
        - label_map: {"structure_name": label_id}
        - color_map: {label_id: [R, G, B, A]}
    """
    if priority_prefixes is None:
        priority_prefixes = ["nerve_", "tumor"]

    ref_img = nib.load(reference_nifti_path)
    ref_shape = ref_img.shape[:3]
    ref_affine = ref_img.affine

    logger.info(f"Reference NIfTI shape: {ref_shape}, affine:\n{ref_affine}")

    combined = np.zeros(ref_shape, dtype=np.uint8)

    # Label 할당 (1부터 시작, 0은 배경)
    label_map: Dict[str, int] = {}
    color_map: Dict[int, List[int]] = {}

    # 마스크를 우선순위에 따라 정렬 (낮은 우선순위 먼저 → 높은 우선순위가 덮어씀)
    def get_priority(name: str) -> int:
        for i, prefix in enumerate(priority_prefixes):
            if name.lower().startswith(prefix):
                return len(priority_prefixes) - i  # 높은 숫자 = 높은 우선순위
        return 0

    sorted_masks = sorted(mask_paths.items(), key=lambda x: get_priority(x[0]))

    label_id = 1
    for name, path in sorted_masks:
        if not Path(path).exists():
            logger.warning(f"Mask file not found: {path}")
            continue

        if label_id > 255:
            logger.warning(f"Too many segments (>255), skipping: {name}")
            break

        try:
            mask_img = nib.load(path)

            # Resample if shape OR affine differs from reference.
            # Shape-only check misses orientation mismatches: e.g. mask in ('L','P','S')
            # vs reference in ('L','A','S') — same shape but flipped voxel axis.
            affine_match = np.allclose(mask_img.affine, ref_affine, atol=1e-3)
            shape_match = mask_img.shape[:3] == ref_shape

            if affine_match and shape_match:
                mask_data = mask_img.get_fdata()
            else:
                reason = []
                if not shape_match:
                    reason.append(f"shape {mask_img.shape[:3]} vs {ref_shape}")
                if not affine_match:
                    reason.append("affine mismatch")
                logger.info(f"Resampling {name} to reference space ({', '.join(reason)})")
                mask_data = resample_mask_to_reference(mask_img, ref_img)

            mask_bool = mask_data > 0.5
            voxel_count = np.sum(mask_bool)

            if voxel_count == 0:
                logger.warning(f"Empty mask: {name}, skipping")
                continue

            combined[mask_bool] = label_id

            label_map[name] = label_id
            color = get_color_for_structure_rgba(name)
            color_map[label_id] = list(color)

            logger.info(f"Added label {label_id}: {name} ({voxel_count} voxels)")
            label_id += 1

        except Exception as e:
            logger.error(f"Failed to process mask {name}: {e}")
            continue

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    out_img = nib.Nifti1Image(combined, ref_affine)
    out_img.header.set_data_dtype(np.uint8)
    nib.save(out_img, output_path)

    logger.info(f"Saved multi-label NIfTI: {output_path} ({label_id - 1} labels)")

    return output_path, label_map, color_map


def save_label_config(
    output_path: str,
    label_map: Dict[str, int],
    color_map: Dict[int, List[int]],
    metadata: Dict[str, Any] = None,
) -> str:
    """
    Label 설정을 JSON으로 저장.

    Args:
        output_path: 출력 JSON 경로
        label_map: {"structure_name": label_id}
        color_map: {label_id: [R, G, B, A]}
        metadata: 추가 메타데이터 (optional)

    Returns:
        출력 파일 경로
    """
    config = {
        "labels": label_map,
        "colors": {str(k): v for k, v in color_map.items()},  # JSON key는 문자열
        "segments": [
            {
                "segmentIndex": label_id,
                "label": name,
                "color": color_map.get(label_id, [128, 128, 128, 150]),
            }
            for name, label_id in sorted(label_map.items(), key=lambda x: x[1])
        ],
    }

    if metadata:
        config["metadata"] = metadata

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Saved label config: {output_path}")
    return output_path


def generate_labelmap_for_study(
    ct_nifti_path: str,
    segmentation_dir: str,
    nerve_masks_dir: str,
    output_dir: str,
    study_uid: str,
) -> Dict[str, Any]:
    """
    Study에 대한 multi-label NIfTI labelmap 생성.

    Args:
        ct_nifti_path: CT NIfTI 파일 경로 (또는 참조용 mask)
        segmentation_dir: 분할 결과 폴더 (normal_structure/, tumor/ 하위 폴더 포함)
        nerve_masks_dir: 신경 mask 폴더
        output_dir: 출력 디렉토리
        study_uid: Study UID (파일명에 사용)

    Returns:
        {
            "labelmap_path": str,
            "config_path": str,
            "label_map": dict,
            "color_map": dict,
            "num_labels": int,
        }
    """
    seg_dir = Path(segmentation_dir)
    nerve_dir = Path(nerve_masks_dir) if nerve_masks_dir else None
    out_dir = Path(output_dir)

    mask_paths: Dict[str, str] = {}

    # 1. Normal structure masks
    normal_dir = seg_dir / "normal_structure"
    if normal_dir.exists():
        for nii_file in normal_dir.glob("*.nii.gz"):
            name = nii_file.stem.replace(".nii", "")
            mask_paths[name] = str(nii_file)
        for nii_file in normal_dir.glob("*.nii"):
            if not str(nii_file).endswith(".nii.gz"):
                name = nii_file.stem
                mask_paths[name] = str(nii_file)

    # 2. Tumor masks
    tumor_dir = seg_dir / "tumor"
    if tumor_dir.exists():
        for nii_file in tumor_dir.glob("*.nii.gz"):
            name = nii_file.stem.replace(".nii", "")
            mask_paths[f"tumor_{name}"] = str(nii_file)
        for nii_file in tumor_dir.glob("*.nii"):
            if not str(nii_file).endswith(".nii.gz"):
                name = nii_file.stem
                mask_paths[f"tumor_{name}"] = str(nii_file)

    # 3. Nerve masks (우선순위 높음)
    if nerve_dir and nerve_dir.exists():
        for nii_file in nerve_dir.glob("*.nii.gz"):
            name = nii_file.stem.replace(".nii", "")
            mask_paths[f"nerve_{name}"] = str(nii_file)

    if not mask_paths:
        raise ValueError(f"No mask files found in {segmentation_dir}")

    logger.info(f"Found {len(mask_paths)} mask files")

    # 참조 NIfTI 결정 (CT 또는 첫 번째 mask)
    reference_path = ct_nifti_path
    if not Path(reference_path).exists():
        # CT가 없으면 첫 번째 mask 사용
        reference_path = list(mask_paths.values())[0]

    labelmap_path = str(out_dir / f"labelmap.nii.gz")
    config_path = str(out_dir / f"label_config.json")

    labelmap_path, label_map, color_map = create_multilabel_nifti(
        reference_nifti_path=reference_path,
        mask_paths=mask_paths,
        output_path=labelmap_path,
        priority_prefixes=["nerve_", "tumor_"],
    )

    save_label_config(
        output_path=config_path,
        label_map=label_map,
        color_map=color_map,
        metadata={
            "study_uid": study_uid,
            "reference_nifti": reference_path,
        },
    )

    return {
        "labelmap_path": labelmap_path,
        "config_path": config_path,
        "label_map": label_map,
        "color_map": color_map,
        "num_labels": len(label_map),
    }
