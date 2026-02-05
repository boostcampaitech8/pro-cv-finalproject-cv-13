"""Segmentation pipeline runner.

Docker 기반 분할 파이프라인을 실행합니다.
- Normal Structure: TotalSegmentator v2 + nnUNet
- Tumor: STU-Net + nnUNet

중요: Backend 컨테이너에서 Docker socket을 통해 다른 컨테이너를 실행할 때,
Named Volume을 사용해야 합니다. 컨테이너 내부 경로를 bind mount하면
호스트에서 해당 경로를 찾을 수 없습니다.
"""
import subprocess
import os
import shutil
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import uuid

import nibabel as nib
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NORMAL_SEG_IMAGE = os.environ.get("NORMAL_SEG_IMAGE", "next-ct/normal-seg")
TUMOR_SEG_IMAGE = os.environ.get("TUMOR_SEG_IMAGE", "next-ct/tumor-seg")
SEG_INPUT_DIR = os.environ.get("SEG_INPUT_DIR", "/data/seg_input")
SEG_OUTPUT_DIR = os.environ.get("SEG_OUTPUT_DIR", "/data/seg_output")
DOCKER_NETWORK = os.environ.get("DOCKER_NETWORK", "github_final_team_project_medical-network")

SEG_INPUT_VOLUME = os.environ.get("SEG_INPUT_VOLUME", "github_final_team_project_seg-input")
SEG_OUTPUT_VOLUME = os.environ.get("SEG_OUTPUT_VOLUME", "github_final_team_project_seg-output")
TS_CACHE_VOLUME = os.environ.get("TS_CACHE_VOLUME", "github_final_team_project_ts-cache")

NNUNET_LABELS = {
    1: "trachea",
    2: "esophagus",
    3: "thyroid_gland",
    4: "vertebrae_C1",
    5: "vertebrae_C2",
    6: "vertebrae_C3",
    7: "vertebrae_C4",
    8: "vertebrae_C5",
    9: "vertebrae_C6",
    10: "vertebrae_C7",
    11: "vertebrae_T1",
    12: "common_carotid_artery_left",
    13: "common_carotid_artery_right",
    14: "hyoid",
    15: "internal_jugular_vein_left",
    16: "internal_jugular_vein_right",
    17: "anterior_scalene_left",
    18: "anterior_scalene_right",
}


def split_multilabel_nifti(
    multilabel_path: str,
    output_dir: str,
    label_mapping: Dict[int, str] = None,
) -> Dict[str, str]:
    """Split a multi-label NIfTI file into individual binary masks.

    Args:
        multilabel_path: Path to multi-label NIfTI file
        output_dir: Output directory for individual masks
        label_mapping: {label_value: structure_name} mapping

    Returns:
        Dictionary of {structure_name: output_path}
    """
    if label_mapping is None:
        label_mapping = NNUNET_LABELS

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    nii = nib.load(multilabel_path)
    data = np.asarray(nii.dataobj)
    affine = nii.affine
    header = nii.header

    unique_labels = np.unique(data)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background

    logger.info(f"Found {len(unique_labels)} labels in {multilabel_path}")

    output_files = {}

    for label_val in unique_labels:
        label_val = int(label_val)

        if label_val not in label_mapping:
            logger.warning(f"Unknown label {label_val}, skipping")
            continue

        structure_name = label_mapping[label_val]

        binary_mask = (data == label_val).astype(np.uint8)

        out_nii = nib.Nifti1Image(binary_mask, affine, header)
        out_file = output_path / f"{structure_name}.nii.gz"
        nib.save(out_nii, str(out_file))

        output_files[structure_name] = str(out_file)
        logger.info(f"  Saved {structure_name} ({np.count_nonzero(binary_mask)} voxels)")

    return output_files


def run_docker_segmentation(
    image_name: str,
    session_id: str,
    command: List[str] = None,
    timeout: int = 1800,
    use_ts_cache: bool = False,
) -> Dict[str, Any]:
    """Run a segmentation Docker container using named volumes.

    Named volumes are shared between backend and segmentation containers.
    Backend writes to /data/seg_input/{session_id}/ which maps to the named volume.
    Segmentation container mounts the same named volume.

    Args:
        image_name: Docker image name
        session_id: Session ID for input/output subdirectory
        command: Command to run (should use /data/seg_input/{session_id} paths)
        timeout: Timeout in seconds
        use_ts_cache: Whether to mount TotalSegmentator cache volume

    Returns:
        Execution result
    """
    container_name = f"seg-{uuid.uuid4().hex[:8]}"

    docker_cmd = [
        "docker", "run",
        "--rm",
        "--name", container_name,
        "--gpus", "all",
        "--shm-size", "16g",
        "-v", f"{SEG_INPUT_VOLUME}:/data/seg_input:ro",
        "-v", f"{SEG_OUTPUT_VOLUME}:/data/seg_output",
        "--network", DOCKER_NETWORK,
    ]

    if use_ts_cache:
        docker_cmd.extend(["-v", f"{TS_CACHE_VOLUME}:/root/.totalsegmentator"])

    docker_cmd.append(image_name)

    if command:
        docker_cmd.extend(command)

    logger.info(f"Running: {' '.join(docker_cmd)}")

    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            logger.error(f"Docker run failed: {result.stderr}")
            return {
                "status": "error",
                "error": result.stderr,
                "stdout": result.stdout,
            }

        logger.info(f"Docker run completed: {result.stdout[-500:] if len(result.stdout) > 500 else result.stdout}")
        return {
            "status": "success",
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "stop", container_name], capture_output=True)
        return {
            "status": "error",
            "error": f"Timeout after {timeout} seconds",
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


def run_normal_structure_segmentation(
    ct_nifti_path: str,
    output_dir: str,
    timeout: int = 1800,
) -> Dict[str, Any]:
    """Run TotalSegmentator v2 + nnUNet segmentation pipeline.

    Args:
        ct_nifti_path: CT NIfTI file path
        output_dir: Output directory for masks

    Returns:
        Segmentation result with output paths
    """
    ct_path = Path(ct_nifti_path)
    output_path = Path(output_dir)

    if not ct_path.exists():
        raise FileNotFoundError(f"CT file not found: {ct_path}")

    session_id = uuid.uuid4().hex[:8]
    input_session = Path(SEG_INPUT_DIR) / session_id
    output_session = Path(SEG_OUTPUT_DIR) / session_id

    input_session.mkdir(parents=True, exist_ok=True)
    output_session.mkdir(parents=True, exist_ok=True)

    try:
        ct_stem = ct_path.stem.replace('.nii', '')
        ct_dest = input_session / f"{ct_stem}_0000.nii.gz"
        shutil.copy2(ct_path, ct_dest)
        logger.info(f"Copied CT to {ct_dest}")

        result = run_docker_segmentation(
            image_name=NORMAL_SEG_IMAGE,
            session_id=session_id,
            command=[
                "inference.sh",
                f"/data/seg_input/{session_id}",
                f"/data/seg_output/{session_id}",
                "--only-ct"
            ],
            timeout=timeout,
            use_ts_cache=True,
        )

        if result["status"] != "success":
            return result

        output_files = list(output_session.glob("*.nii.gz"))
        if not output_files:
            return {
                "status": "error",
                "error": f"No output files found in {output_session}",
            }

        multilabel_file = output_files[0]
        logger.info(f"Found output: {multilabel_file}")

        normal_structure_dir = output_path / "normal_structure"
        mask_files = split_multilabel_nifti(
            str(multilabel_file),
            str(normal_structure_dir),
            NNUNET_LABELS,
        )

        return {
            "status": "success",
            "output_dir": str(normal_structure_dir),
            "mask_count": len(mask_files),
            "masks": list(mask_files.keys()),
            "mask_files": mask_files,
        }

    finally:
        if input_session.exists():
            shutil.rmtree(input_session, ignore_errors=True)
        if output_session.exists():
            shutil.rmtree(output_session, ignore_errors=True)


def run_tumor_segmentation(
    ct_nifti_path: str,
    output_dir: str,
    pt_nifti_path: Optional[str] = None,
    timeout: int = 1800,
) -> Dict[str, Any]:
    """Run tumor segmentation pipeline (STU-Net + nnUNet).

    현재 모델은 CT+PT 2채널로 학습되어 PT가 필수입니다.
    나중에 CT-only 모델이 추가되면 PT 없이도 실행 가능하도록 확장 예정.

    Args:
        ct_nifti_path: CT NIfTI file path
        output_dir: Output directory
        pt_nifti_path: PET NIfTI file path (현재 필수)

    Returns:
        Segmentation result
    """
    ct_path = Path(ct_nifti_path)
    output_path = Path(output_dir)

    if not ct_path.exists():
        raise FileNotFoundError(f"CT file not found: {ct_path}")

    # PT 필수 체크 (현재 모델은 CT+PT 2채널로 학습됨)
    if pt_nifti_path is None:
        return {
            "status": "error",
            "error": "PT (PET) 파일이 필요합니다. 현재 tumor 모델은 CT+PT 2채널로 학습되었습니다.",
            "requires_pt": True,
        }

    pt_path = Path(pt_nifti_path)
    if not pt_path.exists():
        return {
            "status": "error",
            "error": f"PT file not found: {pt_path}",
            "requires_pt": True,
        }

    session_id = uuid.uuid4().hex[:8]
    input_session = Path(SEG_INPUT_DIR) / session_id
    output_session = Path(SEG_OUTPUT_DIR) / session_id

    input_session.mkdir(parents=True, exist_ok=True)
    output_session.mkdir(parents=True, exist_ok=True)

    try:
        case_name = ct_path.stem.replace('.nii', '').replace('_0000', '')
        ct_dest = input_session / f"{case_name}__CT.nii.gz"
        shutil.copy2(ct_path, ct_dest)

        pt_dest = input_session / f"{case_name}__PT.nii.gz"
        shutil.copy2(pt_path, pt_dest)
        pt_arg = ["--pt-path", f"/data/seg_input/{session_id}/{case_name}__PT.nii.gz"]

        command = [
            "inference.py",
            f"/data/seg_input/{session_id}/{case_name}__CT.nii.gz",
            f"/data/seg_output/{session_id}",
            "--model-folder", "plans",
            "--checkpoint", "checkpoints/checkpoint_best.pth",
        ] + pt_arg

        result = run_docker_segmentation(
            image_name=TUMOR_SEG_IMAGE,
            session_id=session_id,
            command=command,
            timeout=timeout,
        )

        if result["status"] != "success":
            return result

        output_files = list(output_session.glob("*.nii.gz"))
        if not output_files:
            return {
                "status": "error",
                "error": "No output files found",
            }

        tumor_dir = output_path / "tumor"
        tumor_dir.mkdir(parents=True, exist_ok=True)

        tumor_mask = output_files[0]
        tumor_dest = tumor_dir / "tumor_gtv.nii.gz"
        shutil.copy2(tumor_mask, tumor_dest)

        return {
            "status": "success",
            "output_dir": str(tumor_dir),
            "tumor_mask": str(tumor_dest),
        }

    finally:
        if input_session.exists():
            shutil.rmtree(input_session, ignore_errors=True)
        if output_session.exists():
            shutil.rmtree(output_session, ignore_errors=True)


def run_full_segmentation_pipeline(
    ct_nifti_path: str,
    output_dir: str,
    pt_nifti_path: Optional[str] = None,
    run_normal_structure: bool = True,
    run_tumor: bool = False,
    timeout: int = 1800,
) -> Dict[str, Any]:
    """Run full segmentation pipeline.

    Args:
        ct_nifti_path: CT NIfTI file path
        output_dir: Output directory
        pt_nifti_path: Optional PET NIfTI file path
        run_normal_structure: Whether to run normal structure segmentation
        run_tumor: Whether to run tumor segmentation
        timeout: Timeout per segmentation task

    Returns:
        Combined segmentation results
    """
    import time

    results = {
        "status": "success",
        "steps": {},
        "timing": {},
    }

    total_start = time.time()

    # 1. Normal structure segmentation
    if run_normal_structure:
        logger.info("  [분할] 정상 구조물 분할 시작...")
        ns_start = time.time()
        try:
            normal_result = run_normal_structure_segmentation(
                ct_nifti_path, output_dir, timeout
            )
            results["steps"]["normal_structure"] = normal_result
            if normal_result["status"] != "success":
                results["status"] = "partial_failure"
        except Exception as e:
            logger.exception("Normal structure segmentation failed")
            results["steps"]["normal_structure"] = {
                "status": "error",
                "error": str(e),
            }
            results["status"] = "partial_failure"
        ns_time = time.time() - ns_start
        results["timing"]["normal_structure"] = round(ns_time, 2)
        logger.info(f"  [분할] 정상 구조물 분할 완료: {ns_time:.2f}초")

    # 2. Tumor segmentation
    if run_tumor:
        logger.info("  [분할] 종양 분할 시작...")
        tumor_start = time.time()
        try:
            tumor_result = run_tumor_segmentation(
                ct_nifti_path, output_dir, pt_nifti_path, timeout
            )
            results["steps"]["tumor"] = tumor_result
            if tumor_result["status"] != "success":
                results["status"] = "partial_failure"
        except Exception as e:
            logger.exception("Tumor segmentation failed")
            results["steps"]["tumor"] = {
                "status": "error",
                "error": str(e),
            }
            results["status"] = "partial_failure"
        tumor_time = time.time() - tumor_start
        results["timing"]["tumor"] = round(tumor_time, 2)
        logger.info(f"  [분할] 종양 분할 완료: {tumor_time:.2f}초")

    total_time = time.time() - total_start
    results["timing"]["total"] = round(total_time, 2)
    logger.info(f"  [분할] 전체 분할 완료: {total_time:.2f}초")

    results["output_dir"] = output_dir
    return results


def check_docker_available() -> bool:
    """Check if Docker is available."""
    try:
        result = subprocess.run(
            ["docker", "version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def check_segmentation_images() -> Dict[str, Any]:
    """Check available segmentation Docker images."""
    images = {
        "normal_structure": {
            "image": NORMAL_SEG_IMAGE,
            "available": False,
        },
        "tumor": {
            "image": TUMOR_SEG_IMAGE,
            "available": False,
        },
    }

    if not check_docker_available():
        return {
            "docker_available": False,
            "images": images,
        }

    for key in images:
        image_name = images[key]["image"]
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", image_name],
                capture_output=True,
                timeout=10,
            )
            images[key]["available"] = result.returncode == 0
        except Exception:
            pass

    return {
        "docker_available": True,
        "images": images,
    }
