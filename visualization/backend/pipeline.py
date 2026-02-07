"""Pipeline orchestration for nerve estimation visualization workflow.

This module provides a unified pipeline that:
1. Runs nerve estimation
2. Converts CT NIfTI to DICOM
3. Converts segmentation masks to DICOM SEG
4. Creates nerve path cylindrical masks and converts to DICOM SEG
5. Uploads everything to Orthanc
6. Returns OHIF viewer URL
"""

import os
import sys
import json
import shutil
import zipfile
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Any, Optional

import nibabel as nib

logger = logging.getLogger(__name__)

sys.path.insert(0, "/app")

from dicom_converter import (
    nifti_to_dicom_series,
    nifti_seg_to_dicom_seg,
    create_dicom_metadata,
    get_structure_color,
    create_multi_segment_dicom_seg,
)
from nerve_to_dicom import (
    nerve_json_to_nifti_masks,
    get_nerve_color,
)
from orthanc_client import OrthancClient
from rtss_generator import (
    nifti_masks_to_rtss,
    find_existing_rtstruct,
    add_nerves_to_existing_rtss,
)
from surface_segmentation_generator import (
    nifti_masks_to_surface_segmentation,
)
try:
    from colors import STRUCTURE_COLORS
except ImportError:
    from .colors import STRUCTURE_COLORS
from nifti_generator import generate_labelmap_for_study


def extract_segmentation_zip(
    zip_path: str,
    output_dir: str,
) -> Dict[str, Path]:
    """Extract segmentation zip file.

    Expected zip structure:
    segmentation/
    ├── normal_structure/
    │   ├── trachea.nii.gz
    │   └── ...
    └── tumor/
        └── tumor.nii.gz

    Args:
        zip_path: Path to segmentation zip file
        output_dir: Output directory for extraction

    Returns:
        Dictionary with paths to each segmentation folder
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    result = {}

    seg_root = output_dir
    if (output_dir / "segmentation").exists():
        seg_root = output_dir / "segmentation"

    for folder_name in ["normal_structure", "tumor"]:
        folder_path = seg_root / folder_name
        if folder_path.exists():
            result[folder_name] = folder_path

    return result


def find_reference_nifti(segmentation_dirs: Dict[str, Path]) -> Optional[str]:
    """Find a reference NIfTI file for geometry.

    Args:
        segmentation_dirs: Dictionary of segmentation folder paths

    Returns:
        Path to a reference NIfTI file, or None if not found
    """
    for folder_name in ["normal_structure", "tumor"]:
        folder = segmentation_dirs.get(folder_name)
        if folder and folder.exists():
            for nii_file in folder.glob("*.nii.gz"):
                return str(nii_file)
    return None


def run_pipeline(
    ct_path: str,
    segmentation_dir: str,
    output_dir: str,
    patient_name: str = "Anonymous",
    orthanc_url: str = None,
    ohif_url: str = "http://localhost:3000",
) -> Dict[str, Any]:
    """Run the complete nerve estimation visualization pipeline.

    Pipeline steps:
    1. Run nerve_estimation on segmentation data
    2. Convert CT NIfTI to DICOM series
    3. Convert all segmentation masks to DICOM SEG
    4. Create nerve path cylindrical masks and convert to DICOM SEG
    5. Upload everything to Orthanc
    6. Return OHIF viewer URL

    Args:
        ct_path: Path to CT NIfTI file
        segmentation_dir: Path to directory with segmentation folders
            (normal_structure/, tumor/)
        output_dir: Output directory for intermediate files
        patient_name: Patient name for DICOM metadata
        orthanc_url: Orthanc server URL (default from env or http://orthanc:8042)
        ohif_url: OHIF viewer base URL

    Returns:
        Dictionary with:
        - study_uid: DICOM StudyInstanceUID
        - ohif_url: URL to open the study in OHIF
        - nerve_results: Nerve estimation results
        - conversion_results: DICOM conversion details
        - upload_results: Orthanc upload details
    """
    orthanc_url = orthanc_url or os.environ.get("ORTHANC_URL", "http://orthanc:8042")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "status": "processing",
        "steps": {},
    }

    segmentation_dir = Path(segmentation_dir)

    seg_folders = {}
    for folder_name in ["normal_structure", "tumor"]:
        folder_path = segmentation_dir / folder_name
        if folder_path.exists():
            seg_folders[folder_name] = folder_path

    # Step 1: Run nerve estimation
    try:
        from nerve_estimation import run_nerve_estimation

        nerve_output_dir = output_dir / "nerve_output"
        nerve_output_dir.mkdir(exist_ok=True)

        nerve_results = run_nerve_estimation(
            normal_structure_dir=str(seg_folders.get("normal_structure")) if "normal_structure" in seg_folders else None,
            tumor_path=str(seg_folders.get("tumor")) if "tumor" in seg_folders else None,
            output_dir=str(nerve_output_dir),
        )

        results["steps"]["nerve_estimation"] = {
            "status": "success",
            "output_dir": str(nerve_output_dir),
        }
        results["nerve_results"] = nerve_results

    except Exception as e:
        results["steps"]["nerve_estimation"] = {
            "status": "error",
            "error": str(e),
        }
        nerve_results = None

    # Step 2: Create nerve path masks
    ref_nifti = find_reference_nifti(seg_folders)
    if ref_nifti is None:
        ref_nifti = ct_path  # Fall back to CT if no seg masks

    nerve_masks_dir = output_dir / "nerve_masks"
    nerve_results_path = output_dir / "nerve_output" / "nerve_results.json"

    if nerve_results_path.exists() and ref_nifti:
        try:
            nerve_mask_files = nerve_json_to_nifti_masks(
                str(nerve_results_path),
                ref_nifti,
                str(nerve_masks_dir),
            )
            results["steps"]["nerve_masks"] = {
                "status": "success",
                "files": nerve_mask_files,
            }
        except Exception as e:
            results["steps"]["nerve_masks"] = {
                "status": "error",
                "error": str(e),
            }
    else:
        results["steps"]["nerve_masks"] = {
            "status": "skipped",
            "reason": "No nerve results or reference NIfTI",
        }

    # Step 3: Convert CT to DICOM
    dicom_dir = output_dir / "dicom"
    ct_dicom_dir = dicom_dir / "ct"

    study_uid = None
    series_uid = None

    try:
        metadata = create_dicom_metadata(
            patient_name=patient_name,
            study_description="Nerve Estimation Study",
            series_description="CT Volume",
        )

        study_uid, series_uid, ct_files = nifti_to_dicom_series(
            ct_path,
            str(ct_dicom_dir),
            metadata=metadata,
        )

        results["steps"]["ct_conversion"] = {
            "status": "success",
            "study_uid": study_uid,
            "series_uid": series_uid,
            "files_count": len(ct_files),
        }
        results["study_uid"] = study_uid

    except Exception as e:
        results["steps"]["ct_conversion"] = {
            "status": "error",
            "error": str(e),
        }

    # Steps 4 + 6 in parallel: RTSS generation + Labelmap generation
    seg_conversion_results = []
    all_masks_dict = {}

    if study_uid and ct_dicom_dir.exists():
        all_segment_masks = []

        for folder_name, folder_path in seg_folders.items():
            for nii_file in folder_path.glob("*.nii.gz"):
                structure_name = nii_file.stem.replace(".nii", "")
                color = get_structure_color(structure_name)
                all_segment_masks.append({
                    'nifti_path': str(nii_file),
                    'label': structure_name.replace("_", " ").title(),
                    'color': color,
                })
                seg_conversion_results.append({
                    "name": structure_name,
                    "color": color,
                    "status": "pending",
                })

        if nerve_masks_dir.exists():
            for nii_file in nerve_masks_dir.glob("*.nii.gz"):
                nerve_name = nii_file.stem.replace(".nii", "")
                color = get_nerve_color(nerve_name)
                all_segment_masks.append({
                    'nifti_path': str(nii_file),
                    'label': f"Nerve: {nerve_name.replace('_', ' ').title()}",
                    'color': color,
                })
                seg_conversion_results.append({
                    "name": f"nerve_{nerve_name}",
                    "color": color,
                    "type": "nerve_path",
                    "status": "pending",
                })

        for folder_name, folder_path in seg_folders.items():
            for nii_file in folder_path.glob("*.nii.gz"):
                structure_name = nii_file.stem.replace(".nii", "")
                all_masks_dict[structure_name] = str(nii_file)

        if nerve_masks_dir.exists():
            for nii_file in nerve_masks_dir.glob("*.nii.gz"):
                nerve_name = nii_file.stem.replace(".nii", "")
                all_masks_dict[f"nerve_{nerve_name}"] = str(nii_file)

    def _gen_rtss():
        if not all_masks_dict:
            return None
        rtss_output = dicom_dir / "rtss" / "rtstruct.dcm"
        path = nifti_masks_to_rtss(
            mask_files=all_masks_dict,
            reference_ct_dir=str(ct_dicom_dir),
            output_path=str(rtss_output),
            colors=STRUCTURE_COLORS,
            study_uid=study_uid,
            patient_name=patient_name,
        )
        print(f"[Pipeline] Generated RTSS with {len(all_masks_dict)} structures")
        return path

    def _gen_labelmap():
        labelmap_output_dir = output_dir / "labelmap"
        labelmap_output_dir.mkdir(parents=True, exist_ok=True)
        r = generate_labelmap_for_study(
            ct_nifti_path=ct_path,
            segmentation_dir=str(segmentation_dir),
            nerve_masks_dir=str(nerve_masks_dir) if nerve_masks_dir.exists() else None,
            output_dir=str(labelmap_output_dir),
            study_uid=study_uid,
        )
        print(f"[Pipeline] Generated NIfTI labelmap with {r['num_labels']} labels")
        return r

    labelmap_result = None
    rtss_path = None

    with ThreadPoolExecutor(max_workers=2) as ex:
        f_rtss = ex.submit(_gen_rtss)
        f_labelmap = ex.submit(_gen_labelmap)

        try:
            rtss_path = f_rtss.result()
            if rtss_path:
                for result in seg_conversion_results:
                    result["status"] = "success"
                    result["file"] = str(rtss_path)
        except Exception as e:
            print(f"[Pipeline] RTSS generation error: {e}")
            for result in seg_conversion_results:
                result["status"] = "error"
                result["error"] = str(e)

        try:
            labelmap_result = f_labelmap.result()
            results["steps"]["labelmap"] = {
                "status": "success",
                "num_labels": labelmap_result.get("num_labels", 0),
                "labelmap_path": labelmap_result.get("labelmap_path"),
            }
        except Exception as e:
            print(f"[Pipeline] NIfTI labelmap generation failed: {e}")
            labelmap_result = {"error": str(e)}
            results["steps"]["labelmap"] = {
                "status": "error",
                "error": str(e),
            }

    results["steps"]["seg_conversion"] = {
        "status": "success" if seg_conversion_results else "skipped",
        "segments": seg_conversion_results,
    }
    results["segments_created"] = len(seg_conversion_results)
    results["labelmap_result"] = labelmap_result

    # Step 5: Upload to Orthanc
    try:
        client = OrthancClient(url=orthanc_url)
        upload_result = client.upload_directory(str(dicom_dir))

        results["steps"]["orthanc_upload"] = {
            "status": "success",
            "uploaded_count": len(upload_result["uploaded"]),
            "failed_count": len(upload_result["failed"]),
            "study_ids": upload_result["study_ids"],
            "series_ids": upload_result["series_ids"],
        }

    except Exception as e:
        results["steps"]["orthanc_upload"] = {
            "status": "error",
            "error": str(e),
        }

    if study_uid:
        results["ohif_url"] = f"{ohif_url}/nerve-assessment?StudyInstanceUIDs={study_uid}"

    all_success = all(
        step.get("status") in ("success", "skipped")
        for step in results["steps"].values()
    )
    results["status"] = "success" if all_success else "partial_failure"

    return results


def analyze_dicom_study(
    original_dicom_dir: str,
    ct_nifti_path: str,
    segmentation_dir: str,
    output_dir: str,
    study_instance_uid: str,
    orthanc_url: str = None,
) -> Dict[str, Any]:
    """DICOM 입력 워크플로우 전용 분석 함수.

    CT는 이미 Orthanc에 있으므로:
    - CT 변환/업로드 안 함
    - SEG만 생성하여 원본 Study에 추가

    Args:
        original_dicom_dir: 다운로드한 원본 CT DICOM 디렉토리
        ct_nifti_path: 변환된 NIfTI (분석용)
        segmentation_dir: 분할 결과 폴더
        output_dir: 출력 디렉토리
        study_instance_uid: 원본 StudyInstanceUID (유지!)
        orthanc_url: Orthanc 서버 URL

    Returns:
        Dictionary with:
        - status: 처리 상태
        - study_uid: 원본 StudyInstanceUID
        - ohif_url: OHIF 뷰어 URL
        - nerve_results: 신경 추정 결과
        - segments_created: 생성된 SEG 수
        - upload_result: 업로드 결과
    """
    orthanc_url = orthanc_url or os.environ.get("ORTHANC_URL", "http://orthanc:8042")

    output_dir = Path(output_dir)
    seg_dir = Path(segmentation_dir)

    # Step 0: 분할 마스크 수집 (normal_structure, tumor 폴더)
    seg_folders = {}
    for folder_name in ["normal_structure", "tumor"]:
        folder_path = seg_dir / folder_name
        if folder_path.exists():
            seg_folders[folder_name] = folder_path

    # Step 1: 신경 추정 실행
    try:
        from nerve_estimation import run_nerve_estimation

        nerve_output_dir = output_dir / "nerve_output"
        nerve_output_dir.mkdir(parents=True, exist_ok=True)

        nerve_results = run_nerve_estimation(
            normal_structure_dir=str(seg_folders.get("normal_structure")) if "normal_structure" in seg_folders else None,
            tumor_path=str(seg_folders.get("tumor")) if "tumor" in seg_folders else None,
            output_dir=str(nerve_output_dir),
        )
    except Exception as e:
        nerve_results = {"error": str(e)}

    # Step 2: 신경 경로 마스크 생성
    nerve_masks_dir = output_dir / "nerve_masks"
    nerve_results_path = output_dir / "nerve_output" / "nerve_results.json"

    # reference NIfTI 찾기 (기하학 정보용)
    ref_nifti = ct_nifti_path
    for folder in seg_folders.values():
        for nii_file in folder.glob("*.nii.gz"):
            ref_nifti = str(nii_file)
            break
        if ref_nifti != ct_nifti_path:
            break

    nerve_mask_files = {}  # Dict로 초기화
    if nerve_results_path.exists():
        try:
            nerve_mask_files = nerve_json_to_nifti_masks(
                str(nerve_results_path),
                ref_nifti,
                str(nerve_masks_dir),
            )
        except Exception as e:
            logger.warning(f"[Pipeline] Nerve mask creation failed: {e}")

    # Step 3: 모든 SEG를 하나의 multi-segment DICOM SEG로 변환 (원본 CT DICOM 참조!)
    seg_output_dir = output_dir / "seg"
    seg_output_dir.mkdir(parents=True, exist_ok=True)

    all_segment_masks = []

    # 3-1: 분할 마스크 수집
    for folder_name, folder_path in seg_folders.items():
        for nii_file in folder_path.glob("*.nii.gz"):
            structure_name = nii_file.stem.replace(".nii", "")
            color = get_structure_color(structure_name)
            all_segment_masks.append({
                'nifti_path': str(nii_file),
                'label': structure_name.replace("_", " ").title(),
                'color': color,
            })

    # 3-2: 신경 마스크 수집
    for nerve_name, nerve_path in nerve_mask_files.items():
        color = get_nerve_color(nerve_name)
        all_segment_masks.append({
            'nifti_path': nerve_path,
            'label': f"Nerve: {nerve_name.replace('_', ' ').title()}",
            'color': color,
        })

    # 3-3: 단일 multi-segment SEG 파일 생성
    segments_created = 0
    if all_segment_masks:
        try:
            create_multi_segment_dicom_seg(
                segment_masks=all_segment_masks,
                reference_dicom_dir=original_dicom_dir,
                output_path=str(seg_output_dir / "all_segments.dcm"),
                study_uid=study_instance_uid,
            )
            segments_created = len(all_segment_masks)
        except Exception as e:
            logger.warning(f"[Pipeline] DICOM SEG creation failed: {e}")

    # Step 4: SEG만 Orthanc에 업로드 (CT는 안 함!)
    client = OrthancClient(url=orthanc_url)
    upload_result = client.upload_directory(str(seg_output_dir))

    return {
        "status": "success",
        "study_uid": study_instance_uid,  # 원본 UID 그대로
        "ohif_url": f"http://localhost:3000/nerve-assessment?StudyInstanceUIDs={study_instance_uid}",
        "nerve_results": nerve_results,
        "segments_created": segments_created,
        "upload_result": upload_result,
    }


def generate_rtss_for_study(
    mask_files: Dict[str, str],
    reference_ct_dir: str,
    output_dir: str,
    study_uid: str,
    patient_name: str = "Anonymous",
) -> str:
    """
    Generate a single RTSS file from all masks.

    Args:
        mask_files: {"name": "/path/to/mask.nii.gz"}
        reference_ct_dir: CT DICOM directory
        output_dir: Output directory
        study_uid: Original StudyInstanceUID
        patient_name: Patient name

    Returns:
        Generated RTSS file path
    """
    output_path = Path(output_dir) / "rtss" / "rtstruct.dcm"

    rtss_path = nifti_masks_to_rtss(
        mask_files=mask_files,
        reference_ct_dir=reference_ct_dir,
        output_path=str(output_path),
        colors=STRUCTURE_COLORS,
        study_uid=study_uid,
        patient_name=patient_name,
    )

    return rtss_path


def generate_surface_segmentation_for_study(
    mask_files: Dict[str, str],
    reference_ct_dir: str,
    output_dir: str,
    study_uid: str,
    patient_name: str = "Anonymous",
    decimate_ratio: float = 0.3,
) -> str:
    """
    Generate a DICOM Surface Segmentation file from all masks.

    Args:
        mask_files: {"name": "/path/to/mask.nii.gz"}
        reference_ct_dir: CT DICOM directory
        output_dir: Output directory
        study_uid: Original StudyInstanceUID
        patient_name: Patient name
        decimate_ratio: Mesh decimation ratio (0.1-1.0)

    Returns:
        Generated Surface Segmentation file path
    """
    output_path = Path(output_dir) / "surface_seg" / "surface_segmentation.dcm"

    surface_seg_path = nifti_masks_to_surface_segmentation(
        mask_files=mask_files,
        reference_ct_dir=reference_ct_dir,
        output_path=str(output_path),
        colors=STRUCTURE_COLORS,
        study_uid=study_uid,
        patient_name=patient_name,
        decimate_ratio=decimate_ratio,
    )

    return surface_seg_path


def analyze_dicom_study_rtss(
    original_dicom_dir: str,
    ct_nifti_path: str,
    segmentation_dir: str,
    output_dir: str,
    study_instance_uid: str,
    orthanc_url: str = None,
    patient_name: str = "Anonymous",
    generate_surface_seg: bool = True,
) -> Dict[str, Any]:
    """DICOM workflow with RTSS + Surface Segmentation output.

    Uses server-side processing to avoid WebAssembly memory limits.
    Generates:
    - DICOM RT Structure Set (2D contours) for slice-by-slice rendering
    - DICOM Surface Segmentation (3D mesh) for 3D viewport rendering

    Args:
        original_dicom_dir: Downloaded original CT DICOM directory
        ct_nifti_path: Converted NIfTI (for analysis only)
        segmentation_dir: Segmentation results folder
        output_dir: Output directory
        study_instance_uid: Original StudyInstanceUID (preserved)
        orthanc_url: Orthanc server URL
        patient_name: Patient name
        generate_surface_seg: Whether to generate Surface Segmentation for 3D view

    Returns:
        Dictionary with:
        - status: Processing status
        - study_uid: Original StudyInstanceUID
        - ohif_url: OHIF viewer URL
        - nerve_results: Nerve estimation results
        - structures_count: Number of structures
        - rtss_path: Path to generated RTSS file
        - surface_seg_path: Path to generated Surface Segmentation file
        - upload_result: Upload result
    """
    orthanc_url = orthanc_url or os.environ.get("ORTHANC_URL", "http://orthanc:8042")

    output_dir = Path(output_dir)
    seg_dir = Path(segmentation_dir)

    # Step 0: Collect segmentation masks (normal_structure, tumor folders)
    seg_folders = {}
    for folder_name in ["normal_structure", "tumor"]:
        folder_path = seg_dir / folder_name
        if folder_path.exists():
            seg_folders[folder_name] = folder_path

    # Step 1: Run nerve estimation
    try:
        from nerve_estimation import run_nerve_estimation

        nerve_output_dir = output_dir / "nerve_output"
        nerve_output_dir.mkdir(parents=True, exist_ok=True)

        nerve_results = run_nerve_estimation(
            normal_structure_dir=str(seg_folders.get("normal_structure")) if "normal_structure" in seg_folders else None,
            tumor_path=str(seg_folders.get("tumor")) if "tumor" in seg_folders else None,
            output_dir=str(nerve_output_dir),
        )
    except Exception as e:
        nerve_results = {"error": str(e)}

    # Step 2: Create nerve path masks
    nerve_masks_dir = output_dir / "nerve_masks"
    nerve_results_path = output_dir / "nerve_output" / "nerve_results.json"

    # Find reference NIfTI (for geometry info)
    ref_nifti = ct_nifti_path
    for folder in seg_folders.values():
        for nii_file in folder.glob("*.nii.gz"):
            ref_nifti = str(nii_file)
            break
        if ref_nifti != ct_nifti_path:
            break

    nerve_mask_files = {}
    if nerve_results_path.exists():
        try:
            nerve_mask_files = nerve_json_to_nifti_masks(
                str(nerve_results_path),
                ref_nifti,
                str(nerve_masks_dir),
            )
        except Exception as e:
            print(f"[RTSS Pipeline] Error creating nerve masks: {e}")

    # Step 3: Check for existing RTSTRUCT in Orthanc and collect masks
    existing_rtstruct_id = find_existing_rtstruct(study_instance_uid, orthanc_url)
    has_existing_rtstruct = existing_rtstruct_id is not None

    if has_existing_rtstruct:
        print(f"[RTSS Pipeline] Found existing RTSTRUCT, will add nerves only")
    else:
        print(f"[RTSS Pipeline] No existing RTSTRUCT, will create new with all structures")

    all_masks = {}
    if not has_existing_rtstruct:
        for folder_name, folder_path in seg_folders.items():
            for nii_file in folder_path.glob("*.nii.gz"):
                structure_name = nii_file.stem.replace(".nii", "")
                all_masks[structure_name] = str(nii_file)

    nerve_only_masks = {}
    for nerve_name, nerve_path in nerve_mask_files.items():
        prefixed_name = f"nerve_{nerve_name}"
        all_masks[prefixed_name] = nerve_path
        nerve_only_masks[prefixed_name] = nerve_path

    rtss_output_dir = output_dir / "rtss"
    rtss_output_dir.mkdir(parents=True, exist_ok=True)
    labelmap_output_dir = output_dir / "labelmap"
    labelmap_output_dir.mkdir(parents=True, exist_ok=True)

    # Steps 2.5, 4, 5 in parallel
    def _gen_labelmap():
        r = generate_labelmap_for_study(
            ct_nifti_path=ct_nifti_path,
            segmentation_dir=str(seg_dir),
            nerve_masks_dir=str(nerve_masks_dir) if nerve_mask_files else None,
            output_dir=str(labelmap_output_dir),
            study_uid=study_instance_uid,
        )
        print(f"[RTSS Pipeline] Generated NIfTI labelmap with {r['num_labels']} labels")
        return r

    def _gen_rtss():
        if has_existing_rtstruct and nerve_only_masks:
            path = add_nerves_to_existing_rtss(
                nerve_mask_files=nerve_only_masks,
                reference_ct_dir=original_dicom_dir,
                output_path=str(rtss_output_dir / "updated_rtss.dcm"),
                study_uid=study_instance_uid,
                orthanc_url=orthanc_url,
            )
            return path, len(nerve_only_masks)
        elif all_masks:
            path = generate_rtss_for_study(
                mask_files=all_masks,
                reference_ct_dir=original_dicom_dir,
                output_dir=str(output_dir),
                study_uid=study_instance_uid,
                patient_name=patient_name,
            )
            return path, len(all_masks)
        return None, 0

    def _gen_surface():
        if has_existing_rtstruct or not generate_surface_seg or not all_masks:
            return None
        return generate_surface_segmentation_for_study(
            mask_files=all_masks,
            reference_ct_dir=original_dicom_dir,
            output_dir=str(output_dir),
            study_uid=study_instance_uid,
            patient_name=patient_name,
            decimate_ratio=0.3,
        )

    labelmap_result = None
    rtss_path = None
    structures_count = 0
    surface_seg_path = None

    with ThreadPoolExecutor(max_workers=3) as ex:
        f_labelmap = ex.submit(_gen_labelmap)
        f_rtss = ex.submit(_gen_rtss)
        f_surface = ex.submit(_gen_surface)

        try:
            labelmap_result = f_labelmap.result()
        except Exception as e:
            print(f"[RTSS Pipeline] NIfTI labelmap generation failed: {e}")
            labelmap_result = {"error": str(e)}

        try:
            rtss_path, structures_count = f_rtss.result()
            if rtss_path:
                print(f"[RTSS Pipeline] RTSS: {structures_count} structures")
        except Exception as e:
            print(f"[RTSS Pipeline] RTSS generation failed: {e}")
            return {"status": "error", "error": str(e), "study_uid": study_instance_uid}

        try:
            surface_seg_path = f_surface.result()
            if surface_seg_path:
                print(f"[RTSS Pipeline] Generated Surface Segmentation: {surface_seg_path}")
        except Exception as e:
            print(f"[RTSS Pipeline] Surface Segmentation generation failed: {e}")

    # Step 6: Upload RTSS and Surface Segmentation to Orthanc
    client = OrthancClient(url=orthanc_url)
    upload_result = {"uploaded": [], "failed": [], "study_ids": [], "series_ids": []}

    if rtss_path:
        try:
            # upload_file returns: {success, orthanc_id, study_id, series_id, error}
            file_result = client.upload_file(rtss_path)

            if file_result.get("success"):
                upload_result["uploaded"].append(rtss_path)
                if file_result.get("study_id"):
                    upload_result["study_ids"].append(file_result["study_id"])
                if file_result.get("series_id"):
                    upload_result["series_ids"].append(file_result["series_id"])
                print(f"[RTSS Pipeline] Uploaded RTSS to Orthanc: {file_result.get('orthanc_id')}")
            else:
                upload_result["failed"].append({
                    "file": rtss_path,
                    "error": file_result.get("error", "Unknown error"),
                })
                print(f"[RTSS Pipeline] RTSS upload failed: {file_result.get('error')}")
        except Exception as e:
            print(f"[RTSS Pipeline] RTSS upload exception: {e}")
            upload_result["failed"].append({
                "file": rtss_path,
                "error": str(e),
            })

    # Upload Surface Segmentation if generated
    if surface_seg_path:
        try:
            file_result = client.upload_file(surface_seg_path)

            if file_result.get("success"):
                upload_result["uploaded"].append(surface_seg_path)
                if file_result.get("study_id"):
                    upload_result["study_ids"].append(file_result["study_id"])
                if file_result.get("series_id"):
                    upload_result["series_ids"].append(file_result["series_id"])
                print(f"[RTSS Pipeline] Uploaded Surface Segmentation to Orthanc: {file_result.get('orthanc_id')}")
            else:
                upload_result["failed"].append({
                    "file": surface_seg_path,
                    "error": file_result.get("error", "Unknown error"),
                })
                print(f"[RTSS Pipeline] Surface Segmentation upload failed: {file_result.get('error')}")
        except Exception as e:
            print(f"[RTSS Pipeline] Surface Segmentation upload exception: {e}")
            upload_result["failed"].append({
                "file": surface_seg_path,
                "error": str(e),
            })

    return {
        "status": "success",
        "study_uid": study_instance_uid,
        "ohif_url": f"http://localhost:3000/nerve-assessment?StudyInstanceUIDs={study_instance_uid}",
        "nerve_results": nerve_results,
        "structures_count": structures_count,
        "rtss_path": rtss_path,
        "surface_seg_path": surface_seg_path,
        "upload_result": upload_result,
        "labelmap_result": labelmap_result,
    }


def run_pipeline_from_files(
    ct_path: str,
    segmentation_zip_path: str,
    output_dir: str,
    patient_name: str = "Anonymous",
    orthanc_url: str = None,
    ohif_url: str = "http://localhost:3000",
    cleanup: bool = False,
) -> Dict[str, Any]:
    """Run pipeline from CT file and segmentation zip.

    This is a convenience wrapper that handles zip extraction before
    running the main pipeline.

    Args:
        ct_path: Path to CT NIfTI file
        segmentation_zip_path: Path to segmentation zip file
        output_dir: Output directory
        patient_name: Patient name for DICOM
        orthanc_url: Orthanc server URL
        ohif_url: OHIF viewer URL
        cleanup: Whether to cleanup intermediate files after completion

    Returns:
        Pipeline results dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract segmentation zip
    seg_extract_dir = output_dir / "segmentation"
    extract_segmentation_zip(segmentation_zip_path, str(seg_extract_dir))

    # Find the actual segmentation folder (might be nested)
    seg_root = seg_extract_dir
    if (seg_extract_dir / "segmentation").exists():
        seg_root = seg_extract_dir / "segmentation"

    # Run pipeline
    results = run_pipeline(
        ct_path=ct_path,
        segmentation_dir=str(seg_root),
        output_dir=str(output_dir),
        patient_name=patient_name,
        orthanc_url=orthanc_url,
        ohif_url=ohif_url,
    )

    # Cleanup if requested
    if cleanup and results.get("status") == "success":
        # Keep the DICOM output and nerve results, remove intermediate files
        for item in ["segmentation", "nerve_masks"]:
            item_path = output_dir / item
            if item_path.exists():
                shutil.rmtree(item_path)

    return results


def run_full_pipeline_with_segmentation(
    ct_path: str,
    output_dir: str,
    patient_name: str = "Anonymous",
    orthanc_url: str = None,
    ohif_url: str = "http://localhost:3000",
    run_tumor_seg: bool = False,
    pt_path: str = None,
    segmentation_timeout: int = 1800,
) -> Dict[str, Any]:
    """Run full pipeline: CT → Segmentation → Nerve Estimation → DICOM → OHIF.

    This is the complete pipeline that takes a CT NIfTI file as input and:
    1. Runs GPU segmentation (TotalSegmentator + nnUNet) via Docker
    2. Optionally runs tumor segmentation (STU-Net + nnUNet) via Docker
    3. Runs nerve estimation on segmentation results
    4. Converts everything to DICOM (CT + RTSTRUCT)
    5. Uploads to Orthanc
    6. Returns OHIF viewer URL

    Args:
        ct_path: Path to CT NIfTI file
        output_dir: Output directory for all intermediate and final files
        patient_name: Patient name for DICOM metadata
        orthanc_url: Orthanc server URL (default from env)
        ohif_url: OHIF viewer base URL
        run_tumor_seg: Whether to run tumor segmentation (requires PT for best results)
        pt_path: Path to PET NIfTI file (optional, for tumor segmentation)
        segmentation_timeout: Timeout for each segmentation step in seconds

    Returns:
        Dictionary with:
        - status: Processing status
        - study_uid: DICOM StudyInstanceUID
        - ohif_url: URL to open the study in OHIF
        - steps: Detailed status of each pipeline step
        - nerve_results: Nerve estimation results (if successful)
    """
    from segmentation_runner import (
        run_full_segmentation_pipeline,
        check_segmentation_images,
    )

    orthanc_url = orthanc_url or os.environ.get("ORTHANC_URL", "http://orthanc:8042")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "status": "processing",
        "steps": {},
    }

    # Step 0: Check Docker and segmentation images
    print("[Full Pipeline] Checking segmentation environment...")
    seg_check = check_segmentation_images()

    if not seg_check.get("docker_available"):
        results["status"] = "error"
        results["error"] = "Docker is not available. Cannot run segmentation."
        return results

    if not seg_check["images"]["normal_structure"]["available"]:
        print("[Full Pipeline] WARNING: Normal structure segmentation image not found")
        results["steps"]["environment_check"] = {
            "status": "warning",
            "message": "Normal structure segmentation image not available",
            "images": seg_check["images"],
        }
    else:
        results["steps"]["environment_check"] = {
            "status": "success",
            "images": seg_check["images"],
        }

    # Step 1: Run segmentation
    print("[Full Pipeline] Running segmentation...")
    seg_output_dir = output_dir / "segmentation"
    seg_output_dir.mkdir(exist_ok=True)

    try:
        seg_result = run_full_segmentation_pipeline(
            ct_nifti_path=ct_path,
            output_dir=str(seg_output_dir),
            pt_nifti_path=pt_path,
            run_normal_structure=True,
            run_tumor=run_tumor_seg,
            timeout=segmentation_timeout,
        )

        results["steps"]["segmentation"] = seg_result

        if seg_result["status"] == "error":
            results["status"] = "error"
            results["error"] = f"Segmentation failed: {seg_result.get('error', 'Unknown error')}"
            return results

        print(f"[Full Pipeline] Segmentation completed: {seg_result['status']}")

    except Exception as e:
        print(f"[Full Pipeline] Segmentation exception: {e}")
        results["steps"]["segmentation"] = {
            "status": "error",
            "error": str(e),
        }
        results["status"] = "error"
        results["error"] = f"Segmentation exception: {e}"
        return results

    # Step 2: Run the main pipeline (nerve estimation + DICOM conversion + upload)
    print("[Full Pipeline] Running nerve estimation and DICOM conversion...")

    try:
        pipeline_result = run_pipeline(
            ct_path=ct_path,
            segmentation_dir=str(seg_output_dir),
            output_dir=str(output_dir),
            patient_name=patient_name,
            orthanc_url=orthanc_url,
            ohif_url=ohif_url,
        )

        # Merge pipeline results
        for step_name, step_data in pipeline_result.get("steps", {}).items():
            results["steps"][step_name] = step_data

        results["study_uid"] = pipeline_result.get("study_uid")
        results["ohif_url"] = pipeline_result.get("ohif_url")
        results["nerve_results"] = pipeline_result.get("nerve_results")
        results["segments_created"] = pipeline_result.get("segments_created", 0)
        results["status"] = pipeline_result.get("status", "success")

        print(f"[Full Pipeline] Pipeline completed: {results['status']}")

    except Exception as e:
        print(f"[Full Pipeline] Pipeline exception: {e}")
        results["steps"]["pipeline"] = {
            "status": "error",
            "error": str(e),
        }
        results["status"] = "error"
        results["error"] = f"Pipeline exception: {e}"

    return results
