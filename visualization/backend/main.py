"""FastAPI backend for nerve estimation visualization with OHIF/Orthanc integration."""

import os
import sys
import json
import shutil
import uuid
import zipfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import numpy as np
import nibabel as nib
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from skimage import measure

sys.path.insert(0, "/app")
sys.path.insert(0, str(Path(__file__).parent.parent))

from nerve_estimation import run_nerve_estimation, export_from_json
from segmentation_runner import (
    run_full_segmentation_pipeline,
    check_segmentation_images,
)
from dicom_converter import (
    nifti_to_dicom_series,
    nifti_seg_to_dicom_seg,
    create_dicom_metadata,
    get_structure_color,
    STRUCTURE_COLORS,
)
from orthanc_client import (
    OrthancClient,
    upload_directory_to_orthanc,
)
from dicom_downloader import (
    DicomDownloader,
    download_study_from_orthanc,
    dicom_to_nifti,
)
from nerve_to_dicom import (
    nerve_json_to_nifti_masks,
    get_nerve_color,
    NERVE_COLORS,
)
from pipeline import run_pipeline, analyze_dicom_study, analyze_dicom_study_rtss
try:
    from colors import get_color_for_structure_rgba as get_color_for_structure
except ImportError:
    from .colors import get_color_for_structure_rgba as get_color_for_structure
from pydantic import BaseModel

app = FastAPI(
    title="Nerve Estimation Visualization API",
    description="API for nerve estimation with OHIF Viewer integration",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ORTHANC_URL = os.environ.get("ORTHANC_URL", "http://localhost:8042")
OHIF_URL = os.environ.get("OHIF_URL", "http://localhost:3000")
DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/uploads")) / "nerve_viz_data"
DATA_DIR.mkdir(exist_ok=True)

SEGMENTATION_DIR_METADATA_KEY = "SegmentationDir"


def save_segmentation_mapping(study_uid: str, seg_dir: str) -> bool:
    """Save study_uid -> segmentation_dir mapping to Orthanc metadata.

    Args:
        study_uid: DICOM StudyInstanceUID
        seg_dir: Segmentation directory path

    Returns:
        True if saved successfully
    """
    client = OrthancClient(url=ORTHANC_URL)

    orthanc_id = client.get_orthanc_id_by_study_uid(study_uid)
    if not orthanc_id:
        logger.warning(f"Could not find Orthanc ID for study: {study_uid[:50]}...")
        return False

    success = client.set_study_metadata(orthanc_id, SEGMENTATION_DIR_METADATA_KEY, seg_dir)
    if success:
        logger.info(f"Saved segmentation mapping to Orthanc: {study_uid[:50]}... -> {seg_dir}")
    else:
        logger.warning(f"Failed to save segmentation mapping for: {study_uid[:50]}...")
    return success


def get_segmentation_mapping(study_uid: str) -> Optional[str]:
    """Get segmentation directory from Orthanc metadata.

    Args:
        study_uid: DICOM StudyInstanceUID

    Returns:
        Segmentation directory path or None if not found
    """
    client = OrthancClient(url=ORTHANC_URL)

    orthanc_id = client.get_orthanc_id_by_study_uid(study_uid)
    if not orthanc_id:
        return None

    return client.get_study_metadata(orthanc_id, SEGMENTATION_DIR_METADATA_KEY)


def nifti_to_mesh(nifti_path: str, label: int = None) -> Dict[str, Any]:
    """Convert NIfTI mask to mesh (vertices and faces).

    Args:
        nifti_path: Path to NIfTI file
        label: Optional label value to extract (for multi-label masks)

    Returns:
        Dictionary with vertices, faces, and metadata
    """
    nii = nib.load(nifti_path)
    data = nii.get_fdata()
    affine = nii.affine

    if label is not None:
        mask = (data == label).astype(float)
    else:
        mask = (data > 0).astype(float)

    if np.sum(mask) == 0:
        return {"vertices": [], "faces": [], "empty": True}

    try:
        verts, faces, normals, values = measure.marching_cubes(mask, level=0.5)

        ones = np.ones((len(verts), 1))
        verts_h = np.hstack([verts, ones])
        verts_world = (affine @ verts_h.T).T[:, :3]

        return {
            "vertices": verts_world.tolist(),
            "faces": faces.tolist(),
            "normals": normals.tolist(),
            "empty": False
        }
    except Exception as e:
        return {"vertices": [], "faces": [], "empty": True, "error": str(e)}


def load_ct_volume(nifti_path: str) -> Dict[str, Any]:
    """Load CT volume data for slice viewing.

    Args:
        nifti_path: Path to CT NIfTI file

    Returns:
        Dictionary with volume data and metadata
    """
    nii = nib.load(nifti_path)
    data = nii.get_fdata()
    affine = nii.affine
    header = nii.header

    shape = data.shape
    spacing = header.get_zooms()[:3]

    data_min = float(np.min(data))
    data_max = float(np.max(data))

    return {
        "shape": list(shape),
        "spacing": list(spacing),
        "affine": affine.tolist(),
        "data_range": [data_min, data_max],
        "dtype": str(data.dtype)
    }


@app.get("/")
async def root():
    """Root endpoint - API status."""
    return {
        "message": "Nerve Estimation Visualization API",
        "version": "2.0.0",
        "status": "running",
        "orthanc_url": ORTHANC_URL,
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    import requests
    orthanc_status = "unknown"
    try:
        response = requests.get(f"{ORTHANC_URL}/system", timeout=5)
        if response.status_code == 200:
            orthanc_status = "healthy"
        else:
            orthanc_status = "unhealthy"
    except Exception:
        orthanc_status = "unreachable"

    return {
        "api": "healthy",
        "orthanc": orthanc_status,
    }


@app.post("/upload/ct")
async def upload_ct(file: UploadFile = File(...)):
    """Upload CT NIfTI file."""
    session_id = Path(file.filename).stem
    session_dir = DATA_DIR / session_id
    session_dir.mkdir(exist_ok=True)

    ct_path = session_dir / "ct.nii.gz"
    with open(ct_path, "wb") as f:
        content = await file.read()
        f.write(content)

    metadata = load_ct_volume(str(ct_path))
    metadata["session_id"] = session_id

    return JSONResponse(metadata)


@app.post("/upload/segmentation/{session_id}")
async def upload_segmentation(
    session_id: str,
    files: List[UploadFile] = File(...),
    folder: str = "normal_structure"
):
    """Upload segmentation masks."""
    session_dir = DATA_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    seg_dir = session_dir / folder
    seg_dir.mkdir(exist_ok=True)

    uploaded = []
    for file in files:
        file_path = seg_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        uploaded.append(file.filename)

    return {"uploaded": uploaded, "folder": folder}


@app.post("/run/nerve-estimation/{session_id}")
def run_estimation(session_id: str):
    """Run nerve estimation pipeline."""
    session_dir = DATA_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    output_dir = session_dir / "output"
    output_dir.mkdir(exist_ok=True)

    normal_structure_dir = session_dir / "normal_structure"
    tumor_dir = session_dir / "tumor"

    try:
        results = run_nerve_estimation(
            normal_structure_dir=str(normal_structure_dir) if normal_structure_dir.exists() else None,
            tumor_path=str(tumor_dir) if tumor_dir.exists() else None,
            output_dir=str(output_dir)
        )

        nerve_results_path = output_dir / "nerve_results.json"
        ref_nifti = None

        if normal_structure_dir.exists():
            for nii_file in normal_structure_dir.glob("*.nii.gz"):
                ref_nifti = str(nii_file)
                break

        if ref_nifti and nerve_results_path.exists():
            export_dir = output_dir / "masks"
            export_from_json(str(nerve_results_path), ref_nifti, str(export_dir))

            nerve_masks_dir = output_dir / "nerve_paths"
            try:
                nerve_mask_files = nerve_json_to_nifti_masks(
                    str(nerve_results_path),
                    ref_nifti,
                    str(nerve_masks_dir),
                )
                results["nerve_path_masks"] = nerve_mask_files
            except Exception as e:
                results["nerve_path_masks_error"] = str(e)

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/convert/to-dicom/{session_id}")
def convert_to_dicom(
    session_id: str,
    background_tasks: BackgroundTasks,
    patient_name: str = "Anonymous",
    upload_to_server: bool = True,
):
    """Convert NIfTI files to DICOM and optionally upload to Orthanc.

    Args:
        session_id: Session identifier
        patient_name: Patient name for DICOM
        upload_to_server: Whether to upload to Orthanc automatically
    """
    session_dir = DATA_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    dicom_dir = session_dir / "dicom"
    dicom_dir.mkdir(exist_ok=True)

    results = {
        "session_id": session_id,
        "ct_converted": False,
        "segmentations_converted": [],
        "orthanc_upload": None,
    }

    ct_path = session_dir / "ct.nii.gz"
    ct_dicom_dir = dicom_dir / "ct"

    study_uid = None
    series_uid = None

    if ct_path.exists():
        try:
            metadata = create_dicom_metadata(
                patient_name=patient_name,
                study_description="Nerve Estimation Study",
                series_description="CT Volume",
            )
            study_uid, series_uid, ct_files = nifti_to_dicom_series(
                str(ct_path),
                str(ct_dicom_dir),
                metadata=metadata,
            )
            results["ct_converted"] = True
            results["study_uid"] = study_uid
            results["ct_series_uid"] = series_uid
            results["ct_files_count"] = len(ct_files)
        except Exception as e:
            results["ct_error"] = str(e)

    seg_folders = ["normal_structure", "tumor", "output/masks"]
    segment_number = 1

    for folder in seg_folders:
        folder_path = session_dir / folder
        if not folder_path.exists():
            continue

        for nii_file in folder_path.glob("*.nii.gz"):
            structure_name = nii_file.stem.replace(".nii", "")
            color = get_structure_color(structure_name)

            seg_output = dicom_dir / "seg" / f"{structure_name}.dcm"

            try:
                if ct_dicom_dir.exists() and list(ct_dicom_dir.glob("*.dcm")):
                    nifti_seg_to_dicom_seg(
                        str(nii_file),
                        str(ct_dicom_dir),
                        str(seg_output),
                        segment_label=structure_name.replace("_", " ").title(),
                        segment_number=segment_number,
                        color=color,
                    )
                    results["segmentations_converted"].append({
                        "name": structure_name,
                        "file": str(seg_output),
                        "color": color,
                    })
                    segment_number += 1
            except Exception as e:
                results.setdefault("segmentation_errors", []).append({
                    "name": structure_name,
                    "error": str(e),
                })

    nerve_paths_dir = session_dir / "output" / "nerve_paths"
    if nerve_paths_dir.exists():
        for nii_file in nerve_paths_dir.glob("*.nii.gz"):
            nerve_name = nii_file.stem.replace(".nii", "")
            color = get_nerve_color(nerve_name)

            seg_output = dicom_dir / "seg" / f"nerve_{nerve_name}.dcm"

            try:
                if ct_dicom_dir.exists() and list(ct_dicom_dir.glob("*.dcm")):
                    nifti_seg_to_dicom_seg(
                        str(nii_file),
                        str(ct_dicom_dir),
                        str(seg_output),
                        segment_label=f"Nerve: {nerve_name.replace('_', ' ').title()}",
                        segment_number=segment_number,
                        color=color,
                    )
                    results["segmentations_converted"].append({
                        "name": f"nerve_{nerve_name}",
                        "file": str(seg_output),
                        "color": color,
                        "type": "nerve_path",
                    })
                    segment_number += 1
            except Exception as e:
                results.setdefault("nerve_path_errors", []).append({
                    "name": nerve_name,
                    "error": str(e),
                })

    if upload_to_server:
        try:
            upload_result = upload_directory_to_orthanc(str(dicom_dir), ORTHANC_URL)
            results["orthanc_upload"] = {
                "uploaded_count": len(upload_result["uploaded"]),
                "failed_count": len(upload_result["failed"]),
                "study_ids": upload_result["study_ids"],
                "series_ids": upload_result["series_ids"],
            }
        except Exception as e:
            results["orthanc_upload"] = {"error": str(e)}

    return JSONResponse(results)


@app.post("/upload/dicom-to-orthanc")
async def upload_dicom_to_orthanc(files: List[UploadFile] = File(...)):
    """Upload DICOM files directly to Orthanc."""
    import requests

    results = {"uploaded": [], "failed": []}

    for file in files:
        try:
            content = await file.read()
            response = requests.post(
                f"{ORTHANC_URL}/instances",
                data=content,
                headers={"Content-Type": "application/dicom"},
            )

            if response.status_code in (200, 201):
                result = response.json()
                results["uploaded"].append({
                    "filename": file.filename,
                    "orthanc_id": result.get("ID"),
                    "study_id": result.get("ParentStudy"),
                })
            else:
                results["failed"].append({
                    "filename": file.filename,
                    "status": response.status_code,
                    "error": response.text,
                })
        except Exception as e:
            results["failed"].append({
                "filename": file.filename,
                "error": str(e),
            })

    return JSONResponse(results)


@app.get("/orthanc/studies")
def get_orthanc_studies():
    """Get list of studies from Orthanc."""
    import requests

    try:
        response = requests.get(f"{ORTHANC_URL}/studies")
        if response.status_code == 200:
            study_ids = response.json()
            studies = []
            for study_id in study_ids:
                study_response = requests.get(f"{ORTHANC_URL}/studies/{study_id}")
                if study_response.status_code == 200:
                    studies.append(study_response.json())
            return {"studies": studies}
        else:
            raise HTTPException(status_code=response.status_code, detail="Failed to get studies")
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Orthanc connection error: {str(e)}")


@app.delete("/orthanc/studies/{study_id}")
def delete_orthanc_study(study_id: str):
    """Delete a study from Orthanc."""
    import requests

    try:
        response = requests.delete(f"{ORTHANC_URL}/studies/{study_id}")
        if response.status_code == 200:
            return {"deleted": study_id}
        else:
            raise HTTPException(status_code=response.status_code, detail="Failed to delete study")
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Orthanc connection error: {str(e)}")


@app.get("/mesh/{session_id}/{folder}/{filename}")
def get_mesh(session_id: str, folder: str, filename: str, label: Optional[int] = None):
    """Get mesh data for a segmentation mask."""
    session_dir = DATA_DIR / session_id
    nifti_path = session_dir / folder / filename

    if not nifti_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    mesh = nifti_to_mesh(str(nifti_path), label)
    return JSONResponse(mesh)


@app.get("/slice/{session_id}/{axis}/{index}")
def get_slice(session_id: str, axis: str, index: int):
    """Get a single slice from CT volume.

    Args:
        session_id: Session identifier
        axis: 'axial', 'coronal', or 'sagittal'
        index: Slice index
    """
    session_dir = DATA_DIR / session_id
    ct_path = session_dir / "ct.nii.gz"

    if not ct_path.exists():
        raise HTTPException(status_code=404, detail="CT not found")

    nii = nib.load(str(ct_path))
    data = nii.get_fdata()

    axis_map = {"axial": 2, "coronal": 1, "sagittal": 0}
    if axis not in axis_map:
        raise HTTPException(status_code=400, detail="Invalid axis")

    ax = axis_map[axis]
    max_idx = data.shape[ax] - 1
    index = max(0, min(index, max_idx))

    if ax == 0:
        slice_data = data[index, :, :]
    elif ax == 1:
        slice_data = data[:, index, :]
    else:
        slice_data = data[:, :, index]

    return {
        "slice": slice_data.tolist(),
        "shape": list(slice_data.shape),
        "index": index,
        "max_index": max_idx,
        "axis": axis
    }


@app.get("/volume/{session_id}")
def get_volume_info(session_id: str):
    """Get volume metadata and available structures."""
    session_dir = DATA_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    ct_path = session_dir / "ct.nii.gz"
    ct_info = None
    if ct_path.exists():
        ct_info = load_ct_volume(str(ct_path))

    structures = {}
    for folder in ["normal_structure", "tumor", "output/masks"]:
        folder_path = session_dir / folder
        if folder_path.exists():
            files = list(folder_path.glob("*.nii.gz"))
            structures[folder] = [f.name for f in files]

    nerve_results = None
    nerve_results_path = session_dir / "output" / "nerve_results.json"
    if nerve_results_path.exists():
        with open(nerve_results_path) as f:
            nerve_results = json.load(f)

    risk_results = None
    risk_path = session_dir / "output" / "risk_assessment.json"
    if risk_path.exists():
        with open(risk_path) as f:
            risk_results = json.load(f)

    return {
        "session_id": session_id,
        "ct": ct_info,
        "structures": structures,
        "nerve_results": nerve_results,
        "risk_results": risk_results
    }


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Delete a session and all its data."""
    session_dir = DATA_DIR / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir)
        return {"deleted": session_id}
    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/sessions")
def list_sessions():
    """List all available sessions."""
    sessions = []
    for d in DATA_DIR.iterdir():
        if d.is_dir():
            sessions.append(d.name)
    return {"sessions": sessions}


@app.get("/nifti/{session_id}/{filename:path}")
def get_nifti_file(session_id: str, filename: str):
    """Serve NIfTI file.

    Args:
        session_id: Session identifier
        filename: File path relative to session (e.g., 'ct.nii.gz', 'normal_structure/trachea.nii.gz')
    """
    session_dir = DATA_DIR / session_id
    file_path = session_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    return FileResponse(
        path=str(file_path),
        media_type="application/gzip",
        filename=file_path.name
    )


@app.get("/structure-colors")
async def get_structure_colors():
    """Get predefined colors for anatomical structures."""
    return STRUCTURE_COLORS


@app.get("/nerve-colors")
async def get_nerve_colors():
    """Get predefined colors for nerve structures."""
    return NERVE_COLORS


@app.post("/pipeline/full/{session_id}")
def run_full_pipeline(
    session_id: str,
    patient_name: str = "Anonymous",
    upload_to_server: bool = True,
):
    """Run full pipeline: nerve estimation + DICOM conversion + upload.

    This is the one-click workflow endpoint that:
    1. Runs nerve estimation
    2. Creates nerve path cylindrical masks
    3. Converts everything to DICOM
    4. Uploads to Orthanc

    Args:
        session_id: Session identifier (must have CT and segmentations uploaded)
        patient_name: Patient name for DICOM
        upload_to_server: Whether to upload to Orthanc
    """
    session_dir = DATA_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    results = {
        "session_id": session_id,
        "steps": {},
    }

    # Step 1: Run nerve estimation
    try:
        estimation_result = run_estimation(session_id)
        results["steps"]["nerve_estimation"] = {
            "status": "success",
            "result": estimation_result,
        }
    except Exception as e:
        results["steps"]["nerve_estimation"] = {
            "status": "error",
            "error": str(e),
        }

    # Step 2: Convert to DICOM and upload
    try:
        bg = BackgroundTasks()
        conversion_result = convert_to_dicom(
            session_id=session_id,
            background_tasks=bg,
            patient_name=patient_name,
            upload_to_server=upload_to_server,
        )
        results["steps"]["dicom_conversion"] = {
            "status": "success",
            "result": conversion_result.body.decode() if hasattr(conversion_result, 'body') else conversion_result,
        }
    except Exception as e:
        results["steps"]["dicom_conversion"] = {
            "status": "error",
            "error": str(e),
        }

    all_success = all(
        step.get("status") == "success"
        for step in results["steps"].values()
    )
    results["status"] = "success" if all_success else "partial_failure"

    return JSONResponse(results)


@app.post("/process")
def process_ct(
    ct_file: UploadFile = File(...),
    patient_name: str = "Anonymous",
):
    """Process CT file - production endpoint.

    This is the production endpoint for the full workflow:
    1. Save uploaded CT
    2. Run segmentation models (TotalSegmentator, TSv2, Tumor) - TODO: 팀원 연동
    3. Run nerve estimation
    4. Convert to DICOM (CT + segmentations + nerve paths)
    5. Upload to Orthanc
    6. Return OHIF viewer URL

    Args:
        ct_file: CT NIfTI file (.nii.gz)
        patient_name: Patient name for DICOM metadata

    Returns:
        JSON with:
        - study_uid: DICOM StudyInstanceUID
        - ohif_url: URL to view in OHIF
        - nerve_results: Nerve estimation results
        - status: Overall processing status

    Note:
        현재는 분할 모델이 연동되지 않아 에러를 반환합니다.
        테스트용으로는 /process-test 엔드포인트를 사용하세요.
    """
    # TODO: 분할 모델 연동 후 구현
    raise HTTPException(
        status_code=501,
        detail="분할 모델이 아직 연동되지 않았습니다. 테스트용으로 /process-test 엔드포인트를 사용하세요."
    )


class ProcessTestRequest(BaseModel):
    """Request body for /process-test endpoint."""
    ct_path: str
    segmentation_dir: str
    patient_name: str = "Anonymous"


@app.post("/process-test")
def process_test(request: ProcessTestRequest):
    """Process with local file paths - test endpoint.

    이미 분할이 완료된 로컬 데이터로 테스트할 때 사용합니다.
    분할 모델 이후의 파이프라인(nerve_estimation → DICOM 변환 → OHIF 시각화)을 테스트합니다.

    Args:
        ct_path: 로컬 CT NIfTI 파일 경로 (예: "/data/image.nii.gz")
        segmentation_dir: 로컬 분할 결과 폴더 경로 (예: "/data/segrap_0001")
            폴더 구조:
            - normal_structure/ (TSv2 + nnUNet 통합 출력)
            - tumor/ (종양 분할, optional)
        patient_name: DICOM 메타데이터용 환자 이름

    Returns:
        JSON with:
        - session_id: 세션 ID
        - study_uid: DICOM StudyInstanceUID
        - ohif_url: OHIF 뷰어 URL
        - nerve_results: 신경 추정 결과
        - status: 처리 상태
    """
    ct_path = Path(request.ct_path)
    segmentation_dir = Path(request.segmentation_dir)

    # Validate paths
    if not ct_path.exists():
        raise HTTPException(status_code=404, detail=f"CT 파일을 찾을 수 없습니다: {ct_path}")
    if not segmentation_dir.exists():
        raise HTTPException(status_code=404, detail=f"분할 결과 폴더를 찾을 수 없습니다: {segmentation_dir}")

    session_id = str(uuid.uuid4())[:8]
    session_dir = DATA_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    try:
        results = run_pipeline(
            ct_path=str(ct_path),
            segmentation_dir=str(segmentation_dir),
            output_dir=str(session_dir),
            patient_name=request.patient_name,
            orthanc_url=ORTHANC_URL,
            ohif_url=OHIF_URL,
        )

        results["session_id"] = session_id

        return JSONResponse(results)

    except Exception as e:
        if session_dir.exists():
            shutil.rmtree(session_dir)

        raise HTTPException(status_code=500, detail=str(e))


@app.get("/process/{session_id}/status")
def get_process_status(session_id: str):
    """Get the status of a processing session.

    Args:
        session_id: Session ID returned from /api/process

    Returns:
        Session status and results if available
    """
    session_dir = DATA_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    results_path = session_dir / "nerve_output" / "nerve_results.json"
    nerve_results = None
    if results_path.exists():
        with open(results_path) as f:
            nerve_results = json.load(f)

    dicom_dir = session_dir / "dicom"
    dicom_ready = dicom_dir.exists() and list(dicom_dir.rglob("*.dcm"))

    study_uid = None
    ct_dicom_dir = dicom_dir / "ct"
    if ct_dicom_dir.exists():
        import pydicom
        for dcm_file in ct_dicom_dir.glob("*.dcm"):
            try:
                ds = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
                study_uid = str(ds.StudyInstanceUID)
                break
            except Exception:
                pass

    return {
        "session_id": session_id,
        "status": "completed" if dicom_ready else "processing",
        "nerve_results": nerve_results,
        "study_uid": study_uid,
        "ohif_url": f"{OHIF_URL}/nerve-assessment?StudyInstanceUIDs={study_uid}" if study_uid else None,
    }


class AnalyzeStudyRequest(BaseModel):
    """Request body for /analyze-study endpoint."""
    study_instance_uid: str
    patient_name: str = "Anonymous"
    run_segmentation: bool = False  # TODO: 분할 모델 연동 시 활성화
    segmentation_dir: Optional[str] = None  # 기존 분할 결과 폴더 (테스트용)
    output_format: str = "rtss"  # "rtss" (default) or "seg" - RTSS avoids WebAssembly memory issues


@app.post("/analyze-study")
def analyze_study(request: AnalyzeStudyRequest):
    """Analyze a DICOM study from Orthanc.

    DICOM 워크플로우 메인 엔드포인트:
    1. Orthanc에서 CT DICOM 다운로드 (원본 보존)
    2. DICOM → NIfTI 변환 (분석용만)
    3. 분할 모델 실행 (또는 기존 분할 사용)
    4. 신경 분석 실행
    5. RTSS 또는 SEG 생성하여 원본 Study에 추가
    6. 분석 결과 + OHIF URL 반환

    Output Format:
    - RTSS (default): Server-side contour extraction, ~10-50MB file
      Avoids WebAssembly memory limit (~1.1GB for 18+ segments)
      Direct contour rendering in OHIF (no browser-side conversion)
    - SEG (legacy): DICOM Segmentation format, ~600MB file
      Requires browser-side LABELMAP → CONTOUR conversion (polyseg)
      May cause memory issues with many segments

    Args:
        study_instance_uid: 분석할 Study의 DICOM StudyInstanceUID
        patient_name: DICOM 메타데이터용 환자 이름
        run_segmentation: 분할 모델 실행 여부 (TODO: 미구현)
        segmentation_dir: 기존 분할 결과 폴더 경로 (테스트용)
        output_format: "rtss" (default) or "seg"

    Returns:
        JSON with:
        - session_id: 세션 ID
        - study_uid: 원본 DICOM StudyInstanceUID (변경 없음)
        - ohif_url: OHIF 뷰어 URL
        - nerve_results: 신경 추정 결과
        - output_format: 사용된 출력 형식 (rtss/seg)
        - structures_count/segments_created: 생성된 구조물 수
        - status: 처리 상태
    """
    session_id = str(uuid.uuid4())[:8]
    session_dir = DATA_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "session_id": session_id,
        "study_instance_uid": request.study_instance_uid,
        "steps": {},
    }

    try:
        # Step 1: Download DICOM from Orthanc (원본 DICOM 보존!)
        try:
            original_dicom_dir, study_info = download_study_from_orthanc(
                study_instance_uid=request.study_instance_uid,
                output_dir=str(session_dir),
                orthanc_url=ORTHANC_URL,
            )
            results["steps"]["download"] = {
                "status": "success",
                "dicom_dir": original_dicom_dir,
                "num_dicom_files": study_info.get("num_files", 0),
            }
        except Exception as e:
            results["steps"]["download"] = {
                "status": "error",
                "error": str(e),
            }
            raise HTTPException(status_code=500, detail=f"DICOM 다운로드 실패: {e}")

        # Step 2: Convert DICOM to NIfTI (분석용)
        try:
            ct_nifti_path = session_dir / "ct.nii.gz"
            dicom_to_nifti(original_dicom_dir, str(ct_nifti_path))
            results["steps"]["convert"] = {
                "status": "success",
                "nifti_path": str(ct_nifti_path),
            }
        except Exception as e:
            results["steps"]["convert"] = {
                "status": "error",
                "error": str(e),
            }
            raise HTTPException(status_code=500, detail=f"DICOM→NIfTI 변환 실패: {e}")

        # Step 3: Get segmentation (use existing or run model)
        segmentation_dir = None

        if request.segmentation_dir:
            # Use provided segmentation directory
            seg_dir_path = Path(request.segmentation_dir)
            if seg_dir_path.exists():
                segmentation_dir = str(seg_dir_path)
                results["steps"]["segmentation"] = {
                    "status": "success",
                    "source": "provided",
                    "path": segmentation_dir,
                }
            else:
                results["steps"]["segmentation"] = {
                    "status": "error",
                    "error": f"분할 결과 폴더를 찾을 수 없습니다: {request.segmentation_dir}",
                }
        elif request.run_segmentation:
            # TODO: Run segmentation models (TotalSegmentator, TSv2, Tumor)
            results["steps"]["segmentation"] = {
                "status": "error",
                "error": "분할 모델이 아직 연동되지 않았습니다.",
            }
            raise HTTPException(
                status_code=501,
                detail="분할 모델이 아직 연동되지 않았습니다. segmentation_dir 파라미터로 기존 분할 결과를 제공하세요."
            )
        else:
            # No segmentation provided or requested
            results["steps"]["segmentation"] = {
                "status": "skipped",
                "message": "분할 결과가 제공되지 않았습니다. segmentation_dir 파라미터를 사용하세요.",
            }

        # Step 4: Run DICOM-specific analysis (if segmentation is available)
        if segmentation_dir:
            try:
                # Choose output format: RTSS (default) or SEG
                # RTSS avoids WebAssembly memory issues (polyseg LABELMAP -> CONTOUR conversion)
                # RTSS: ~10-50MB file, vector contours, direct rendering
                # SEG: ~600MB file, voxel data, requires browser-side conversion
                output_format = getattr(request, 'output_format', 'rtss').lower()

                if output_format == "rtss":
                    # RTSS mode: Server-side contour extraction
                    # Avoids WebAssembly memory limit (~1.1GB for 18 segments)
                    pipeline_results = analyze_dicom_study_rtss(
                        original_dicom_dir=original_dicom_dir,
                        ct_nifti_path=str(ct_nifti_path),
                        segmentation_dir=segmentation_dir,
                        output_dir=str(session_dir),
                        study_instance_uid=request.study_instance_uid,
                        orthanc_url=ORTHANC_URL,
                        patient_name=request.patient_name,
                    )

                    results["steps"]["nerve_estimation"] = {
                        "status": "success",
                        "result": pipeline_results.get("nerve_results"),
                    }
                    results["steps"]["rtss_generation"] = {
                        "status": "success",
                        "structures_count": pipeline_results.get("structures_count"),
                        "rtss_path": pipeline_results.get("rtss_path"),
                        "upload_result": pipeline_results.get("upload_result"),
                    }
                    results["output_format"] = "rtss"
                    results["structures_count"] = pipeline_results.get("structures_count")

                    # Save labelmap mapping to Orthanc metadata (for polySeg rendering)
                    labelmap_result = pipeline_results.get("labelmap_result")
                    if labelmap_result and labelmap_result.get("labelmap_path"):
                        labelmap_dir = str(Path(labelmap_result["labelmap_path"]).parent)
                        save_labelmap_mapping(request.study_instance_uid, labelmap_dir)
                        results["labelmap_available"] = True
                    else:
                        results["labelmap_available"] = False
                else:
                    # SEG mode: Legacy DICOM Segmentation format
                    # May cause WebAssembly memory issues with many segments
                    pipeline_results = analyze_dicom_study(
                        original_dicom_dir=original_dicom_dir,
                        ct_nifti_path=str(ct_nifti_path),
                        segmentation_dir=segmentation_dir,
                        output_dir=str(session_dir),
                        study_instance_uid=request.study_instance_uid,
                        orthanc_url=ORTHANC_URL,
                    )

                    results["steps"]["nerve_estimation"] = {
                        "status": "success",
                        "result": pipeline_results.get("nerve_results"),
                    }
                    results["steps"]["seg_upload"] = {
                        "status": "success",
                        "segments_created": pipeline_results.get("segments_created"),
                        "upload_result": pipeline_results.get("upload_result"),
                    }
                    results["output_format"] = "seg"
                    results["segments_created"] = pipeline_results.get("segments_created")

                # Add final results (원본 UID 그대로!)
                results["study_uid"] = request.study_instance_uid
                results["ohif_url"] = f"{OHIF_URL}/nerve-assessment?StudyInstanceUIDs={request.study_instance_uid}"

                # Merge nerve + risk data for frontend display
                raw_nerve_results = pipeline_results.get("nerve_results")
                merged_nerves = _merge_nerve_and_risk_data(raw_nerve_results)

                results["nerve_results"] = {
                    "nerves": merged_nerves,
                    "analysis_summary": {
                        "total_nerves": len(merged_nerves),
                        "high_risk_count": sum(1 for n in merged_nerves if n.get("risk_level", "").upper() == "HIGH"),
                        "moderate_count": sum(1 for n in merged_nerves if n.get("risk_level", "").upper() == "MODERATE"),
                        "low_count": sum(1 for n in merged_nerves if n.get("risk_level", "").upper() == "LOW"),
                    },
                }
                results["risk_report"] = _generate_risk_report(raw_nerve_results)
                results["status"] = "success"

            except Exception as e:
                results["steps"]["processing"] = {
                    "status": "error",
                    "error": str(e),
                }
                results["status"] = "partial_failure"
        else:
            results["status"] = "incomplete"
            results["message"] = "분할 결과가 없어 신경 분석을 실행할 수 없습니다."

        return JSONResponse(results)

    except HTTPException:
        raise
    except Exception as e:
        # Cleanup on error
        if session_dir.exists():
            shutil.rmtree(session_dir)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-nifti")
async def upload_nifti(
    ct_file: UploadFile = File(...),
    patient_name: str = "Anonymous",
    study_description: str = "Uploaded NIfTI Study",
):
    """Upload NIfTI file and convert to DICOM in Orthanc.

    NIfTI 파일을 DICOM으로 변환하여 Orthanc에 업로드합니다.
    이후 /analyze-study 엔드포인트로 분석을 실행할 수 있습니다.

    Args:
        ct_file: CT NIfTI 파일 (.nii.gz)
        patient_name: DICOM 메타데이터용 환자 이름
        study_description: Study 설명

    Returns:
        JSON with:
        - session_id: 세션 ID
        - study_instance_uid: 생성된 DICOM StudyInstanceUID
        - ohif_url: OHIF 뷰어 URL
        - status: 처리 상태
    """
    # Generate session ID
    session_id = str(uuid.uuid4())[:8]
    session_dir = DATA_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    try:
        nifti_path = session_dir / "ct.nii.gz"
        with open(nifti_path, "wb") as f:
            content = await ct_file.read()
            f.write(content)

        try:
            nii = nib.load(str(nifti_path))
            shape = nii.shape
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"유효하지 않은 NIfTI 파일입니다: {e}")

        dicom_dir = session_dir / "dicom" / "ct"
        dicom_dir.mkdir(parents=True, exist_ok=True)

        metadata = create_dicom_metadata(
            patient_name=patient_name,
            study_description=study_description,
            series_description="CT Volume",
        )

        study_uid, series_uid, dicom_files = nifti_to_dicom_series(
            str(nifti_path),
            str(dicom_dir),
            metadata=metadata,
        )

        upload_result = upload_directory_to_orthanc(str(dicom_dir), ORTHANC_URL)

        return JSONResponse({
            "session_id": session_id,
            "study_instance_uid": study_uid,
            "series_instance_uid": series_uid,
            "num_slices": len(dicom_files),
            "volume_shape": list(shape),
            "orthanc_upload": {
                "uploaded_count": len(upload_result["uploaded"]),
                "failed_count": len(upload_result["failed"]),
            },
            "ohif_url": f"{OHIF_URL}/nerve-assessment?StudyInstanceUIDs={study_uid}",
            "analyze_url": f"/api/analyze-study",
            "analyze_body": {
                "study_instance_uid": study_uid,
                "patient_name": patient_name,
            },
            "status": "success",
        })

    except HTTPException:
        raise
    except Exception as e:
        # Cleanup on error
        if session_dir.exists():
            shutil.rmtree(session_dir)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/orthanc/study/{study_instance_uid}")
def get_study_info(study_instance_uid: str):
    """Get study information from Orthanc by StudyInstanceUID.

    Args:
        study_instance_uid: DICOM StudyInstanceUID

    Returns:
        Study details from Orthanc
    """
    downloader = DicomDownloader(orthanc_url=ORTHANC_URL)
    study = downloader.get_study_by_uid(study_instance_uid)

    if not study:
        raise HTTPException(status_code=404, detail=f"Study not found: {study_instance_uid}")

    series_list = downloader.get_series_for_study(study.get("ID"))

    return {
        "study": study,
        "series": series_list,
        "ct_series_id": downloader.find_ct_series(series_list),
    }


@app.post("/upload-and-process")
async def upload_and_process(
    ct_file: UploadFile = File(...),
    patient_name: str = "Anonymous",
    seg_files: List[UploadFile] = File(default=[]),
):
    """Upload NIfTI files and process them.

    Accepts CT NIfTI file and optional segmentation NIfTI files.
    Converts to DICOM and uploads to Orthanc.

    Args:
        ct_file: CT NIfTI file (.nii or .nii.gz)
        patient_name: Patient name for DICOM metadata
        seg_files: Optional list of segmentation NIfTI files

    Returns:
        Processing results with study_uid and ohif_url
    """
    session_id = str(uuid.uuid4())[:8]
    session_dir = DATA_DIR / f"upload_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    try:
        ct_path = session_dir / "ct.nii.gz"
        with open(ct_path, "wb") as f:
            content = await ct_file.read()
            f.write(content)

        seg_dir = session_dir / "segmentations"
        seg_dir.mkdir(exist_ok=True)

        for seg_file in seg_files:
            if seg_file.filename:
                seg_path = seg_dir / seg_file.filename
                with open(seg_path, "wb") as f:
                    content = await seg_file.read()
                    f.write(content)

        results = run_pipeline(
            ct_path=str(ct_path),
            segmentation_dir=str(seg_dir) if any(seg_dir.iterdir()) else None,
            output_dir=str(session_dir / "output"),
            patient_name=patient_name,
            orthanc_url=ORTHANC_URL,
        )

        return JSONResponse({
            "status": results.get("status", "success"),
            "study_uid": results.get("study_uid"),
            "ohif_url": results.get("ohif_url"),
            "message": "Upload and processing complete",
            "segments_created": results.get("segments_created", 0),
        })

    except Exception as e:
        # Cleanup on error
        if session_dir.exists():
            shutil.rmtree(session_dir)
        raise HTTPException(status_code=500, detail=str(e))



class UploadCtAndSegmentRequest(BaseModel):
    """Request body for CT upload with segmentation."""
    patient_name: str = "Anonymous"


def detect_file_type(filename: str) -> str:
    """Detect file type from filename.

    Returns:
        'nifti', 'dicom_zip', or 'dicom' based on file extension
    """
    filename_lower = filename.lower()
    if filename_lower.endswith(('.nii', '.nii.gz')):
        return 'nifti'
    elif filename_lower.endswith('.zip'):
        return 'dicom_zip'
    elif filename_lower.endswith('.dcm'):
        return 'dicom'
    else:
        # Assume DICOM for unknown extensions (DICOM files often have no extension)
        return 'dicom'


async def convert_uploaded_dicom_to_nifti(
    uploaded_file: UploadFile,
    session_dir: Path,
    output_name: str = "ct",
) -> Path:
    """Convert uploaded DICOM file(s) to NIfTI.

    Handles:
    - ZIP file containing DICOM series
    - Single DICOM file (returns error - need multiple files for volume)

    Args:
        uploaded_file: Uploaded file (ZIP or DICOM)
        session_dir: Session directory
        output_name: Output file name (without extension)

    Returns:
        Path to the converted NIfTI file
    """
    file_type = detect_file_type(uploaded_file.filename or "")

    if file_type == 'dicom_zip':
        # Extract ZIP file
        dicom_extract_dir = session_dir / f"{output_name}_dicom"
        dicom_extract_dir.mkdir(parents=True, exist_ok=True)

        # Save and extract ZIP
        zip_path = session_dir / f"{output_name}_temp.zip"
        with open(zip_path, "wb") as f:
            content = await uploaded_file.read()
            f.write(content)

        # Extract ZIP
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dicom_extract_dir)

        # Remove temp ZIP
        zip_path.unlink()

        # Find DICOM files (may be in subdirectory)
        dicom_files = list(dicom_extract_dir.rglob("*.dcm"))
        if not dicom_files:
            # Try files without extension (common in DICOM)
            all_files = list(dicom_extract_dir.rglob("*"))
            dicom_files = [f for f in all_files if f.is_file() and not f.suffix]

        if not dicom_files:
            raise ValueError("ZIP 파일에서 DICOM 파일을 찾을 수 없습니다")

        # Get directory containing DICOM files
        dicom_dir = dicom_files[0].parent

        # Convert to NIfTI
        nifti_path = session_dir / f"{output_name}.nii.gz"
        dicom_to_nifti(str(dicom_dir), str(nifti_path))

        return nifti_path

    else:
        # Single DICOM file - not supported for CT volume
        raise ValueError(
            "단일 DICOM 파일은 지원되지 않습니다. "
            "DICOM 시리즈가 포함된 ZIP 파일을 업로드하거나, NIfTI 파일(.nii.gz)을 사용하세요."
        )


@app.post("/upload-ct-and-segment")
async def upload_ct_and_segment(
    ct_file: UploadFile = File(...),
    pt_file: Optional[UploadFile] = File(default=None),
    patient_name: str = Form(default="Anonymous"),
    run_tumor: bool = Form(default=False),
):
    """CT (+PT) 업로드 → 분할 모델 실행 → Orthanc 업로드.

    Upload Tab에서 사용하는 엔드포인트:
    1. CT 파일 저장 (+ PT 파일, optional) - NIfTI 또는 DICOM ZIP 지원
    2. DICOM인 경우 NIfTI로 변환
    3. TSv2 + nnUNet 분할 실행 (normal structure)
    4. PT가 있고 run_tumor=True면 tumor 분할 실행
    5. CT + 분할 마스크 → DICOM 변환
    6. Orthanc 업로드
    7. Study UID 반환

    Args:
        ct_file: CT 파일 (.nii.gz NIfTI 또는 .zip DICOM 시리즈)
        pt_file: PT (PET) 파일 (.nii.gz NIfTI 또는 .zip DICOM 시리즈) - tumor 분할 시 필수
        patient_name: DICOM 메타데이터용 환자 이름
        run_tumor: tumor 분할 실행 여부 (PT 파일 필요)

    Returns:
        JSON with:
        - status: 처리 상태
        - session_id: 세션 ID
        - study_uid: DICOM StudyInstanceUID
        - ohif_url: OHIF 뷰어 URL
        - segmentation_result: 분할 결과 정보
        - input_format: 입력 파일 형식 (nifti/dicom_zip)
    """
    import time as time_module

    session_id = str(uuid.uuid4())[:8]
    session_dir = DATA_DIR / f"seg_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "session_id": session_id,
        "status": "processing",
        "steps": {},
        "timing": {},
    }

    total_start = time_module.time()
    logger.info(f"[{session_id}] ========== 파이프라인 시작 ==========")

    try:
        # Step 1: CT 파일 처리 (NIfTI 또는 DICOM)
        step1_start = time_module.time()
        ct_file_type = detect_file_type(ct_file.filename or "")
        results["input_format"] = {"ct": ct_file_type}

        if ct_file_type == 'nifti':
            # NIfTI 파일 - 직접 저장
            ct_path = session_dir / "ct.nii.gz"
            with open(ct_path, "wb") as f:
                content = await ct_file.read()
                f.write(content)
            results["steps"]["upload_ct"] = {"status": "success", "path": str(ct_path), "format": "nifti"}
        else:
            # DICOM 파일 - 변환 필요
            try:
                ct_path = await convert_uploaded_dicom_to_nifti(ct_file, session_dir, "ct")
                results["steps"]["upload_ct"] = {
                    "status": "success",
                    "path": str(ct_path),
                    "format": ct_file_type,
                    "converted": True,
                }
            except Exception as e:
                results["steps"]["upload_ct"] = {"status": "error", "error": str(e)}
                raise HTTPException(status_code=400, detail=f"CT DICOM 변환 실패: {e}")

        # Step 1-2: PT 파일 처리 (있으면) - NIfTI 또는 DICOM
        pt_path = None
        if pt_file is not None:
            pt_file_type = detect_file_type(pt_file.filename or "")
            results["input_format"]["pt"] = pt_file_type

            if pt_file_type == 'nifti':
                # NIfTI 파일 - 직접 저장
                pt_path = session_dir / "pt.nii.gz"
                with open(pt_path, "wb") as f:
                    content = await pt_file.read()
                    f.write(content)
                results["steps"]["upload_pt"] = {"status": "success", "path": str(pt_path), "format": "nifti"}
            else:
                # DICOM 파일 - 변환 필요
                try:
                    pt_path = await convert_uploaded_dicom_to_nifti(pt_file, session_dir, "pt")
                    results["steps"]["upload_pt"] = {
                        "status": "success",
                        "path": str(pt_path),
                        "format": pt_file_type,
                        "converted": True,
                    }
                except Exception as e:
                    results["steps"]["upload_pt"] = {"status": "error", "error": str(e)}
                    raise HTTPException(status_code=400, detail=f"PT DICOM 변환 실패: {e}")
        else:
            results["steps"]["upload_pt"] = {"status": "skipped", "message": "PT 파일 없음"}

        step1_time = time_module.time() - step1_start
        results["timing"]["upload"] = round(step1_time, 2)
        logger.info(f"[{session_id}] Step 1 (업로드): {step1_time:.2f}초")

        # tumor 분할 요청했는데 PT 없으면 경고
        if run_tumor and pt_path is None:
            results["steps"]["tumor_warning"] = {
                "status": "warning",
                "message": "Tumor 분할을 요청했지만 PT 파일이 없습니다. Tumor 분할은 CT+PT가 필요합니다.",
            }
            run_tumor = False  # PT 없으면 tumor 분할 비활성화

        # Step 2: 분할 모델 실행
        step2_start = time_module.time()
        logger.info(f"[{session_id}] Step 2 (분할) 시작...")
        try:
            seg_result = run_full_segmentation_pipeline(
                ct_nifti_path=str(ct_path),
                output_dir=str(session_dir),
                pt_nifti_path=str(pt_path) if pt_path else None,
                run_normal_structure=True,
                run_tumor=run_tumor,
            )
            results["steps"]["segmentation"] = seg_result
            segmentation_dir = seg_result.get("output_dir", str(session_dir))
            step2_time = time_module.time() - step2_start
            results["timing"]["segmentation"] = round(step2_time, 2)
            logger.info(f"[{session_id}] Step 2 (분할): {step2_time:.2f}초")
        except Exception as e:
            results["steps"]["segmentation"] = {
                "status": "error",
                "error": str(e),
            }
            raise HTTPException(
                status_code=500,
                detail=f"분할 모델 실행 실패: {e}"
            )

        # Step 3: Pipeline 실행 (DICOM 변환 + Orthanc 업로드)
        step3_start = time_module.time()
        logger.info(f"[{session_id}] Step 3 (DICOM 변환 + 업로드) 시작...")
        try:
            pipeline_results = run_pipeline(
                ct_path=str(ct_path),
                segmentation_dir=segmentation_dir,
                output_dir=str(session_dir / "output"),
                patient_name=patient_name,
                orthanc_url=ORTHANC_URL,
                ohif_url=OHIF_URL,
            )
            results["steps"]["pipeline"] = pipeline_results
            results["study_uid"] = pipeline_results.get("study_uid")
            results["ohif_url"] = pipeline_results.get("ohif_url")
            results["status"] = "success"
            step3_time = time_module.time() - step3_start
            results["timing"]["pipeline"] = round(step3_time, 2)
            logger.info(f"[{session_id}] Step 3 (DICOM 변환 + 업로드): {step3_time:.2f}초")

            # Save study_uid -> segmentation_dir mapping to Orthanc metadata
            study_uid = pipeline_results.get("study_uid")
            if study_uid and segmentation_dir:
                save_segmentation_mapping(study_uid, segmentation_dir)

            # Save labelmap mapping to Orthanc metadata (for polySeg rendering)
            labelmap_result = pipeline_results.get("labelmap_result")
            if labelmap_result and labelmap_result.get("labelmap_path"):
                labelmap_dir = str(Path(labelmap_result["labelmap_path"]).parent)
                save_labelmap_mapping(study_uid, labelmap_dir)
                logger.info(f"[{session_id}] Saved labelmap mapping: {labelmap_dir}")
        except Exception as e:
            results["steps"]["pipeline"] = {
                "status": "error",
                "error": str(e),
            }
            results["status"] = "partial_failure"

        total_time = time_module.time() - total_start
        results["timing"]["total"] = round(total_time, 2)
        logger.info(f"[{session_id}] ========== 파이프라인 완료: 총 {total_time:.2f}초 ==========")
        logger.info(f"[{session_id}] 요약: 업로드 {results['timing'].get('upload', 0):.1f}s | 분할 {results['timing'].get('segmentation', 0):.1f}s | 변환+업로드 {results['timing'].get('pipeline', 0):.1f}s")

        return JSONResponse(results)

    except HTTPException:
        raise
    except Exception as e:
        if session_dir.exists():
            shutil.rmtree(session_dir)
        raise HTTPException(status_code=500, detail=str(e))


class ImportExistingDataRequest(BaseModel):
    """Request body for importing existing data."""
    data_dir: str  # 단일 경로: ct.nii.gz, normal_structure/, tumor/ 포함
    patient_name: str = "Anonymous"


@app.post("/import-existing-data")
def import_existing_data(request: ImportExistingDataRequest):
    """기존 데이터 가져오기 (분할 없이).

    Server Path Tab에서 사용하는 엔드포인트:
    1. 데이터 폴더에서 CT + 마스크 자동 탐지
    2. DICOM 변환
    3. Orthanc 업로드
    4. Study UID 반환

    Args:
        data_dir: 데이터 폴더 경로
            폴더 구조:
            - ct.nii.gz (CT 파일)
            - normal_structure/ (정상 구조물 마스크)
            - tumor/ (종양 마스크, optional)
        patient_name: DICOM 메타데이터용 환자 이름

    Returns:
        JSON with:
        - status: 처리 상태
        - session_id: 세션 ID
        - study_uid: DICOM StudyInstanceUID
        - ohif_url: OHIF 뷰어 URL
    """
    data_dir = Path(request.data_dir)

    # 경로 유효성 검증
    if not data_dir.exists():
        raise HTTPException(status_code=404, detail=f"데이터 폴더를 찾을 수 없습니다: {data_dir}")

    # CT 파일 자동 탐지
    ct_path = data_dir / "ct.nii.gz"
    if not ct_path.exists():
        # ct.nii.gz가 없으면 image.nii.gz 시도
        ct_path = data_dir / "image.nii.gz"
    if not ct_path.exists():
        raise HTTPException(status_code=404, detail=f"CT 파일을 찾을 수 없습니다: {data_dir}/ct.nii.gz 또는 image.nii.gz")

    # 마스크 폴더는 data_dir 자체 (normal_structure/, tumor/ 포함)
    mask_dir = data_dir

    session_id = str(uuid.uuid4())[:8]
    session_dir = DATA_DIR / f"import_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Pipeline 실행 (DICOM 변환 + Orthanc 업로드)
        pipeline_results = run_pipeline(
            ct_path=str(ct_path),
            segmentation_dir=str(mask_dir),
            output_dir=str(session_dir / "output"),
            patient_name=request.patient_name,
            orthanc_url=ORTHANC_URL,
            ohif_url=OHIF_URL,
        )

        return JSONResponse({
            "status": pipeline_results.get("status", "success"),
            "session_id": session_id,
            "study_uid": pipeline_results.get("study_uid"),
            "ohif_url": pipeline_results.get("ohif_url"),
            "segments_created": pipeline_results.get("segments_created", 0),
        })

    except Exception as e:
        if session_dir.exists():
            shutil.rmtree(session_dir)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run-nerve-analysis/{study_uid}")
def run_nerve_analysis(study_uid: str, segmentation_dir: Optional[str] = None):
    """신경 추정 파이프라인 실행.

    AnalysisPanel에서 사용하는 엔드포인트:
    1. Orthanc에서 Study 다운로드
    2. DICOM → NIfTI 변환
    3. nerve_estimation 실행
    4. 신경 마스크 생성
    5. RTSS에 신경 contour 추가
    6. Orthanc에 업로드
    7. Risk Report 반환

    Args:
        study_uid: 분석할 Study의 DICOM StudyInstanceUID
        segmentation_dir: 기존 분할 결과 폴더 경로 (optional, 없으면 Study에서 가져옴)

    Returns:
        JSON with:
        - status: 처리 상태
        - nerve_results: 신경 추정 결과
        - risk_report: Risk Report 요약
        - ohif_url: OHIF 뷰어 URL
    """
    session_id = str(uuid.uuid4())[:8]
    session_dir = DATA_DIR / f"nerve_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "session_id": session_id,
        "study_uid": study_uid,
        "status": "processing",
        "steps": {},
    }

    try:
        # Step 1: Orthanc에서 DICOM 다운로드
        try:
            original_dicom_dir, study_info = download_study_from_orthanc(
                study_instance_uid=study_uid,
                output_dir=str(session_dir),
                orthanc_url=ORTHANC_URL,
            )
            results["steps"]["download"] = {
                "status": "success",
                "dicom_dir": original_dicom_dir,
            }
        except Exception as e:
            results["steps"]["download"] = {"status": "error", "error": str(e)}
            raise HTTPException(status_code=500, detail=f"DICOM 다운로드 실패: {e}")

        # Step 2: DICOM → NIfTI 변환
        try:
            ct_nifti_path = session_dir / "ct.nii.gz"
            dicom_to_nifti(original_dicom_dir, str(ct_nifti_path))
            results["steps"]["convert"] = {
                "status": "success",
                "nifti_path": str(ct_nifti_path),
            }
        except Exception as e:
            results["steps"]["convert"] = {"status": "error", "error": str(e)}
            raise HTTPException(status_code=500, detail=f"DICOM→NIfTI 변환 실패: {e}")

        # Step 3: 분할 결과 확인
        seg_dir = None
        seg_source = None

        if segmentation_dir:
            # User provided explicit path
            seg_dir = Path(segmentation_dir)
            seg_source = "user_provided"
        else:
            # Try to get from Orthanc metadata
            mapped_path = get_segmentation_mapping(study_uid)
            if mapped_path:
                seg_dir = Path(mapped_path)
                seg_source = "orthanc_metadata"
                logger.info(f"[{session_id}] Found segmentation dir from Orthanc metadata: {mapped_path}")

        if seg_dir and not seg_dir.exists():
            logger.warning(f"[{session_id}] Segmentation dir not found: {seg_dir}")
            seg_dir = None

        results["steps"]["segmentation"] = {
            "status": "success" if seg_dir else "skipped",
            "path": str(seg_dir) if seg_dir else None,
            "source": seg_source,
        }

        if not seg_dir:
            results["status"] = "incomplete"
            results["message"] = "분할 결과가 없어 신경 분석을 실행할 수 없습니다. Upload 패널에서 CT/PT를 업로드하고 분할을 먼저 실행하세요."
            return JSONResponse(results)

        # Step 4: RTSS 기반 분석 실행
        try:
            pipeline_results = analyze_dicom_study_rtss(
                original_dicom_dir=original_dicom_dir,
                ct_nifti_path=str(ct_nifti_path),
                segmentation_dir=str(seg_dir),
                output_dir=str(session_dir),
                study_instance_uid=study_uid,
                orthanc_url=ORTHANC_URL,
                patient_name="Patient",
            )

            results["steps"]["nerve_analysis"] = {
                "status": "success",
                "structures_count": pipeline_results.get("structures_count"),
            }

            # Raw nerve results (original structure)
            raw_nerve_results = pipeline_results.get("nerve_results")

            # Merge nerve + risk data for frontend display
            merged_nerves = _merge_nerve_and_risk_data(raw_nerve_results)

            # nerve_results with merged data for frontend
            results["nerve_results"] = {
                "nerves": merged_nerves,  # Frontend가 기대하는 형식
                "analysis_summary": {
                    "total_nerves": len(merged_nerves),
                    "high_risk_count": sum(1 for n in merged_nerves if n.get("risk_level", "").upper() == "HIGH"),
                    "moderate_count": sum(1 for n in merged_nerves if n.get("risk_level", "").upper() == "MODERATE"),
                    "low_count": sum(1 for n in merged_nerves if n.get("risk_level", "").upper() == "LOW"),
                },
            }

            results["risk_report"] = _generate_risk_report(raw_nerve_results)
            results["ohif_url"] = f"{OHIF_URL}/nerve-assessment?StudyInstanceUIDs={study_uid}"
            results["status"] = "success"

            # Save labelmap mapping to Orthanc metadata (for polySeg rendering)
            labelmap_result = pipeline_results.get("labelmap_result")
            if labelmap_result and labelmap_result.get("labelmap_path"):
                labelmap_dir = str(Path(labelmap_result["labelmap_path"]).parent)
                save_labelmap_mapping(study_uid, labelmap_dir)
                results["labelmap_available"] = True
            else:
                results["labelmap_available"] = False

        except Exception as e:
            results["steps"]["nerve_analysis"] = {"status": "error", "error": str(e)}
            results["status"] = "error"
            raise HTTPException(status_code=500, detail=f"신경 분석 실패: {e}")

        return JSONResponse(results)

    except HTTPException:
        raise
    except Exception as e:
        if session_dir.exists():
            shutil.rmtree(session_dir)
        raise HTTPException(status_code=500, detail=str(e))


def _merge_nerve_and_risk_data(nerve_results: Optional[Dict]) -> List[Dict[str, Any]]:
    """Merge nerve estimation and risk assessment data for frontend.

    nerve_results.json 구조:
    - nerves: [{nerve, side, type, uncertainty_mm, pathway_mm, ...}]
    - risks: [{nerve, side, risk_level, min_distance_mm, ...}]

    Frontend가 기대하는 구조:
    - [{name, side, risk_level, min_distance_to_tumor, path_length, ...}]

    Args:
        nerve_results: 신경 추정 결과 (nerves + risks 포함)

    Returns:
        Frontend 형식으로 병합된 신경 데이터 리스트
    """
    if not nerve_results:
        return []

    nerves = nerve_results.get("nerves", [])
    risks = nerve_results.get("risks", [])

    # Build risk lookup: (nerve, side) -> risk_data
    risk_lookup = {}
    for risk in risks:
        key = (risk.get("nerve"), risk.get("side"))
        risk_lookup[key] = risk

    merged = []
    for nerve in nerves:
        nerve_name = nerve.get("nerve", "")
        side = nerve.get("side", "")
        key = (nerve_name, side)

        # Get risk data if available
        risk_data = risk_lookup.get(key, {})

        # Calculate path length from pathway_mm if available
        path_length = None
        pathway_mm = nerve.get("pathway_mm", [])
        if pathway_mm and len(pathway_mm) > 1:
            total_length = 0.0
            for i in range(1, len(pathway_mm)):
                p1, p2 = pathway_mm[i - 1], pathway_mm[i]
                dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
                total_length += dist
            path_length = total_length

        merged.append({
            "name": f"{nerve_name}_{side}",
            "nerve": nerve_name,
            "side": side,
            "type": nerve.get("type"),
            "method": nerve.get("method"),
            "reference": nerve.get("reference"),
            "uncertainty_mm": nerve.get("uncertainty_mm"),
            # Risk data
            "risk_level": risk_data.get("risk_level", "unknown"),
            "min_distance_to_tumor": risk_data.get("min_distance_mm"),
            "effective_distance_mm": risk_data.get("effective_distance_mm"),
            "overlap": risk_data.get("overlap", False),
            "overlap_ratio": risk_data.get("overlap_ratio"),
            # Calculated
            "path_length": path_length,
            # Keep original data for debugging
            "has_risk_assessment": bool(risk_data),
        })

    return merged


def _generate_risk_report(nerve_results: Optional[Dict]) -> Dict[str, Any]:
    """Generate risk report summary from nerve results.

    Args:
        nerve_results: 신경 추정 결과 (nerves + risks 배열 포함)

    Returns:
        Risk report 요약
    """
    if not nerve_results:
        return {"status": "no_data"}

    # risks 배열에서 risk 정보를 가져옴 (nerves가 아님!)
    risks = nerve_results.get("risks", [])
    nerves = nerve_results.get("nerves", [])

    if not nerves:
        return {"status": "no_nerves"}

    if not risks:
        return {
            "status": "no_tumor",
            "message": "No tumor mask available for risk assessment",
            "total_nerves": len(nerves),
        }

    # Risk level 분류 (risks 배열 사용)
    high_risks = []
    moderate_risks = []
    low_risks = []

    for risk in risks:
        risk_level = risk.get("risk_level", "unknown").upper()
        nerve_name = f"{risk.get('nerve', '')}_{risk.get('side', '')}"

        if risk_level == "HIGH":
            high_risks.append(nerve_name)
        elif risk_level == "MODERATE":
            moderate_risks.append(nerve_name)
        else:
            low_risks.append(nerve_name)

    return {
        "status": "analyzed",
        "total_nerves": len(nerves),
        "total_risks_assessed": len(risks),
        "high_risk_count": len(high_risks),
        "moderate_risk_count": len(moderate_risks),
        "low_risk_count": len(low_risks),
        "high_risk_nerves": high_risks,
        "overall_risk": (
            "HIGH" if high_risks else
            "MODERATE" if moderate_risks else
            "LOW"
        ),
    }


@app.get("/segmentation-models")
def get_segmentation_models():
    """Get available segmentation models status.

    Returns:
        Available segmentation models and their status
    """
    return check_segmentation_images()


@app.delete("/delete-series/{series_instance_uid}")
def delete_series(series_instance_uid: str):
    """Delete a series from Orthanc by SeriesInstanceUID.

    OHIF 좌측 패널에서 RTSTRUCT/SEG 시리즈를 삭제할 때 사용합니다.

    Args:
        series_instance_uid: DICOM SeriesInstanceUID

    Returns:
        JSON with:
        - status: 처리 상태
        - deleted: 삭제된 시리즈 ID
        - series_instance_uid: 삭제된 SeriesInstanceUID
    """
    import requests

    try:
        # Step 1: Find series in Orthanc by SeriesInstanceUID
        find_response = requests.post(
            f"{ORTHANC_URL}/tools/find",
            json={
                "Level": "Series",
                "Query": {"SeriesInstanceUID": series_instance_uid},
            },
        )

        if find_response.status_code != 200:
            raise HTTPException(
                status_code=find_response.status_code,
                detail=f"Orthanc 검색 실패: {find_response.text}",
            )

        series_ids = find_response.json()

        if not series_ids:
            raise HTTPException(
                status_code=404,
                detail=f"시리즈를 찾을 수 없습니다: {series_instance_uid}",
            )

        # Step 2: Delete the series
        orthanc_series_id = series_ids[0]
        delete_response = requests.delete(f"{ORTHANC_URL}/series/{orthanc_series_id}")

        if delete_response.status_code == 200:
            return JSONResponse({
                "status": "success",
                "deleted": orthanc_series_id,
                "series_instance_uid": series_instance_uid,
            })
        else:
            raise HTTPException(
                status_code=delete_response.status_code,
                detail=f"시리즈 삭제 실패: {delete_response.text}",
            )

    except requests.RequestException as e:
        raise HTTPException(
            status_code=503,
            detail=f"Orthanc 연결 오류: {str(e)}",
        )


LABELMAP_DIR_METADATA_KEY = "LabelmapDir"


def save_labelmap_mapping(study_uid: str, labelmap_dir: str) -> bool:
    """Save study_uid -> labelmap_dir mapping to Orthanc metadata.

    Args:
        study_uid: DICOM StudyInstanceUID
        labelmap_dir: Labelmap directory path (contains labelmap.nii.gz and label_config.json)

    Returns:
        True if saved successfully
    """
    client = OrthancClient(url=ORTHANC_URL)

    orthanc_id = client.get_orthanc_id_by_study_uid(study_uid)
    if not orthanc_id:
        logger.warning(f"Could not find Orthanc ID for study: {study_uid[:50]}...")
        return False

    success = client.set_study_metadata(orthanc_id, LABELMAP_DIR_METADATA_KEY, labelmap_dir)
    if success:
        logger.info(f"Saved labelmap mapping to Orthanc: {study_uid[:50]}... -> {labelmap_dir}")
    else:
        logger.warning(f"Failed to save labelmap mapping for: {study_uid[:50]}...")
    return success


def get_labelmap_mapping(study_uid: str) -> Optional[str]:
    """Get labelmap directory from Orthanc metadata.

    Args:
        study_uid: DICOM StudyInstanceUID

    Returns:
        Labelmap directory path or None if not found
    """
    client = OrthancClient(url=ORTHANC_URL)

    orthanc_id = client.get_orthanc_id_by_study_uid(study_uid)
    if not orthanc_id:
        return None

    return client.get_study_metadata(orthanc_id, LABELMAP_DIR_METADATA_KEY)


@app.get("/nifti-labelmap/{study_instance_uid}")
def get_nifti_labelmap(study_instance_uid: str):
    """Serve NIfTI multi-label labelmap for a study.

    This endpoint serves the multi-label NIfTI file for Cornerstone3D polySeg rendering.
    Used to work around RTSTRUCT limitations in sagittal/coronal views.

    Args:
        study_instance_uid: DICOM StudyInstanceUID

    Returns:
        NIfTI file (.nii.gz) as binary response
    """
    # Get labelmap directory from Orthanc metadata
    labelmap_dir = get_labelmap_mapping(study_instance_uid)

    if not labelmap_dir:
        raise HTTPException(
            status_code=404,
            detail=f"Labelmap not found for study: {study_instance_uid}. Run nerve analysis first."
        )

    labelmap_path = Path(labelmap_dir) / "labelmap.nii.gz"

    if not labelmap_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Labelmap file not found: {labelmap_path}"
        )

    return FileResponse(
        path=str(labelmap_path),
        media_type="application/gzip",
        filename=f"labelmap_{study_instance_uid[:8]}.nii.gz",
        headers={
            "Content-Disposition": f"attachment; filename=labelmap_{study_instance_uid[:8]}.nii.gz"
        }
    )


@app.get("/label-config/{study_instance_uid}")
def get_label_config(study_instance_uid: str):
    """Serve label configuration JSON for a study.

    This endpoint serves the label-to-color mapping for the multi-label NIfTI.
    Used by frontend to configure segment display in Cornerstone3D.

    Args:
        study_instance_uid: DICOM StudyInstanceUID

    Returns:
        JSON with:
        - labels: {"structure_name": label_id}
        - colors: {label_id: [R, G, B, A]}
        - segments: [{segmentIndex, label, color}]
    """
    # Get labelmap directory from Orthanc metadata
    labelmap_dir = get_labelmap_mapping(study_instance_uid)

    if not labelmap_dir:
        raise HTTPException(
            status_code=404,
            detail=f"Label config not found for study: {study_instance_uid}. Run nerve analysis first."
        )

    config_path = Path(labelmap_dir) / "label_config.json"

    if not config_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Label config file not found: {config_path}"
        )

    with open(config_path, "r") as f:
        config = json.load(f)

    # Dynamically apply latest colors from get_color_for_structure()
    # so color changes in nifti_generator.py take effect without re-importing data
    if "segments" in config:
        for seg in config["segments"]:
            label = seg.get("label", "")
            color = list(get_color_for_structure(label))
            seg["color"] = color
        # Also update the colors dict
        if "labels" in config:
            config["colors"] = {}
            for name, label_id in config["labels"].items():
                color = list(get_color_for_structure(name))
                config["colors"][str(label_id)] = color

    return JSONResponse(config)


def _compute_mesh(mask: np.ndarray, affine: np.ndarray, decimate: float):
    """Marching cubes → affine → RAS-to-LPS → decimate → smooth. Returns (vertices, faces)."""
    verts, faces, normals, values = measure.marching_cubes(mask, level=0.5)

    ones = np.ones((len(verts), 1))
    verts_h = np.hstack([verts, ones])
    verts_world = (affine @ verts_h.T).T[:, :3]

    verts_lps = verts_world.copy()
    verts_lps[:, 0] = -verts_world[:, 0]
    verts_lps[:, 1] = -verts_world[:, 1]

    try:
        import trimesh

        if decimate < 1.0 and len(faces) > 1000:
            mesh = trimesh.Trimesh(vertices=verts_lps, faces=faces)
            target_faces = max(int(len(faces) * decimate), 500)
            mesh = mesh.simplify_quadric_decimation(target_faces)
            verts_lps = mesh.vertices
            faces = mesh.faces

        mesh = trimesh.Trimesh(vertices=verts_lps, faces=faces)
        trimesh.smoothing.filter_humphrey(mesh, iterations=10)
        verts_lps = mesh.vertices
        faces = mesh.faces
    except ImportError:
        pass

    return verts_lps.astype(np.float32), faces.astype(np.int32)


def _load_labelmap_and_config(study_instance_uid: str):
    """Load labelmap NIfTI + config JSON for a study. Returns (data, affine, config) or raises HTTPException."""
    labelmap_dir = get_labelmap_mapping(study_instance_uid)
    if not labelmap_dir:
        raise HTTPException(status_code=404, detail=f"Labelmap not found for study: {study_instance_uid}. Run nerve analysis first.")

    labelmap_path = Path(labelmap_dir) / "labelmap.nii.gz"
    config_path = Path(labelmap_dir) / "label_config.json"

    if not labelmap_path.exists():
        raise HTTPException(status_code=404, detail=f"Labelmap file not found: {labelmap_path}")
    if not config_path.exists():
        raise HTTPException(status_code=404, detail=f"Label config not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    nii = nib.load(str(labelmap_path))
    return nii.get_fdata(), nii.affine, config


@app.get("/surface-mesh/{study_instance_uid}")
def get_surface_mesh(study_instance_uid: str, decimate: float = 0.3):
    """Generate and return 3D surface meshes for all segments."""
    data, affine, config = _load_labelmap_and_config(study_instance_uid)

    structures = {}

    for segment in config.get("segments", []):
        label_id = segment["segmentIndex"]
        name = segment["label"]
        color = segment.get("color", [128, 128, 128, 150])

        mask = (data == label_id).astype(np.float32)

        if np.sum(mask) < 100:
            print(f"[SurfaceMesh] Skipping {name}: too few voxels")
            continue

        try:
            verts_lps, faces = _compute_mesh(mask, affine, decimate)

            structures[name] = {
                "vertices": verts_lps.tolist(),
                "faces": faces.tolist(),
                "color": color[:3] if len(color) > 3 else color,
                "segmentIndex": label_id,
                "vertexCount": len(verts_lps),
                "faceCount": len(faces),
            }

            print(f"[SurfaceMesh] {name}: {len(verts_lps)} vertices, {len(faces)} faces")

        except Exception as e:
            print(f"[SurfaceMesh] Failed to generate mesh for {name}: {e}")
            continue

    return JSONResponse({
        "studyUID": study_instance_uid,
        "structureCount": len(structures),
        "structures": structures,
    })


@app.get("/surface-mesh/{study_instance_uid}/{structure_name}")
def get_single_surface_mesh(study_instance_uid: str, structure_name: str, decimate: float = 0.3):
    """Get mesh for a single structure."""
    data, affine, config = _load_labelmap_and_config(study_instance_uid)

    segment = None
    for s in config.get("segments", []):
        if s["label"] == structure_name:
            segment = s
            break

    if segment is None:
        raise HTTPException(status_code=404, detail=f"Structure '{structure_name}' not found in study")

    label_id = segment["segmentIndex"]
    color = segment.get("color", [128, 128, 128, 150])
    mask = (data == label_id).astype(np.float32)

    if np.sum(mask) < 100:
        raise HTTPException(status_code=404, detail=f"Structure '{structure_name}' has too few voxels")

    verts_lps, faces = _compute_mesh(mask, affine, decimate)

    return JSONResponse({
        "studyUID": study_instance_uid,
        "structure": {
            "vertices": verts_lps.tolist(),
            "faces": faces.tolist(),
            "color": color[:3] if len(color) > 3 else color,
            "segmentIndex": label_id,
            "vertexCount": len(verts_lps),
            "faceCount": len(faces),
        },
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
