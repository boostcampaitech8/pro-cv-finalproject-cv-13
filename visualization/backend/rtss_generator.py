"""
DICOM RT Structure Set (RTSS) Generator

NIfTI 마스크에서 2D contour를 추출하여 DICOM RTSS 파일 생성.
WebAssembly 메모리 한계 회피를 위해 서버에서 contour 계산.

기능:
- 새 RTSS 생성
- 기존 RTSS에 구조물 추가 (신경 추가용)
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from skimage import measure
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
import datetime
import requests

try:
    from colors import STRUCTURE_COLORS
except ImportError:
    from .colors import STRUCTURE_COLORS


# DICOM RTSS SOP Class UID
RT_STRUCTURE_SET_STORAGE = "1.2.840.10008.5.1.4.1.1.481.3"


def extract_contours_from_mask(
    mask_data: np.ndarray,
    affine: np.ndarray,
    min_points: int = 3,
) -> Dict[int, List[np.ndarray]]:
    """
    Extract 2D contours from each axial slice of a 3D mask.

    Note: Only axial contours are generated because Cornerstone's RTSTRUCT
    renderer uses polySeg to convert contours to surfaces for MPR rendering.
    polySeg expects axial-only contours (standard RTSTRUCT format).

    Args:
        mask_data: 3D binary mask (X, Y, Z)
        affine: NIfTI affine matrix for coordinate transformation
        min_points: Minimum contour points (filter out too small contours)

    Returns:
        {slice_index: [contour1_world_coords, contour2_world_coords, ...]}
    """
    contours_by_slice = {}

    for z in range(mask_data.shape[2]):
        slice_2d = mask_data[:, :, z].T  # Transpose for correct orientation

        if not np.any(slice_2d > 0):
            continue

        # skimage.measure.find_contours returns (row, col) coordinates
        slice_contours = measure.find_contours(slice_2d, 0.5)

        valid_contours = []
        for contour in slice_contours:
            if len(contour) < min_points:
                continue

            # Convert (row, col) to world coordinates (mm)
            world_coords = []
            for point in contour:
                row, col = point
                # Voxel coordinates (col, row, z) -> homogeneous
                voxel = np.array([col, row, z, 1.0])
                world = affine @ voxel
                world_coords.append(world[:3])

            valid_contours.append(np.array(world_coords))

        if valid_contours:
            contours_by_slice[z] = valid_contours

    return contours_by_slice


def create_rtss_dataset(
    reference_ct_dir: str,
    structures: Dict[str, Dict],
    study_uid: str = None,
    patient_name: str = "Anonymous",
) -> Dataset:
    """
    Create a DICOM RT Structure Set.

    Args:
        reference_ct_dir: CT DICOM series directory
        structures: {
            "structure_name": {
                "contours": {slice_idx: [contour_coords, ...]},
                "color": (R, G, B),
            }
        }
        study_uid: StudyInstanceUID (None to get from reference)
        patient_name: Patient name

    Returns:
        pydicom Dataset
    """
    ct_files = sorted(Path(reference_ct_dir).glob("*.dcm"))
    if not ct_files:
        raise ValueError(f"No DICOM files in {reference_ct_dir}")

    ref_ds = pydicom.dcmread(str(ct_files[0]))

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = RT_STRUCTURE_SET_STORAGE
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\x00" * 128)

    # Patient module
    ds.PatientName = patient_name
    ds.PatientID = getattr(ref_ds, 'PatientID', generate_uid()[-16:])
    ds.PatientBirthDate = getattr(ref_ds, 'PatientBirthDate', '')
    ds.PatientSex = getattr(ref_ds, 'PatientSex', '')

    # Study module
    ds.StudyInstanceUID = study_uid or ref_ds.StudyInstanceUID
    ds.StudyDate = getattr(ref_ds, 'StudyDate', datetime.date.today().strftime('%Y%m%d'))
    ds.StudyTime = getattr(ref_ds, 'StudyTime', datetime.datetime.now().strftime('%H%M%S'))
    ds.StudyDescription = getattr(ref_ds, 'StudyDescription', 'Nerve Assessment')
    ds.AccessionNumber = getattr(ref_ds, 'AccessionNumber', '')
    ds.ReferringPhysicianName = ''
    ds.StudyID = getattr(ref_ds, 'StudyID', '1')

    # Series module
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber = 999
    ds.SeriesDescription = "RT Structure Set - Nerve Assessment"
    ds.Modality = "RTSTRUCT"

    # Instance module
    ds.SOPClassUID = RT_STRUCTURE_SET_STORAGE
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.InstanceCreationDate = datetime.date.today().strftime('%Y%m%d')
    ds.InstanceCreationTime = datetime.datetime.now().strftime('%H%M%S')

    # Structure Set module
    ds.StructureSetLabel = "Nerve Assessment Structures"
    ds.StructureSetName = "NerveStructures"
    ds.StructureSetDate = ds.InstanceCreationDate
    ds.StructureSetTime = ds.InstanceCreationTime

    # Referenced Frame of Reference Sequence
    ref_for_seq = Sequence()
    ref_for_item = Dataset()
    ref_for_item.FrameOfReferenceUID = getattr(ref_ds, 'FrameOfReferenceUID', generate_uid())

    # RT Referenced Study Sequence
    rt_ref_study_seq = Sequence()
    rt_ref_study_item = Dataset()
    rt_ref_study_item.ReferencedSOPClassUID = "1.2.840.10008.3.1.2.3.1"  # Detached Study
    rt_ref_study_item.ReferencedSOPInstanceUID = ds.StudyInstanceUID

    # RT Referenced Series Sequence
    rt_ref_series_seq = Sequence()
    rt_ref_series_item = Dataset()
    rt_ref_series_item.SeriesInstanceUID = ref_ds.SeriesInstanceUID

    # Contour Image Sequence (reference all CT slices)
    contour_image_seq = Sequence()
    for ct_file in ct_files:
        ct_ds = pydicom.dcmread(str(ct_file), stop_before_pixels=True)
        contour_image_item = Dataset()
        contour_image_item.ReferencedSOPClassUID = ct_ds.SOPClassUID
        contour_image_item.ReferencedSOPInstanceUID = ct_ds.SOPInstanceUID
        contour_image_seq.append(contour_image_item)

    rt_ref_series_item.ContourImageSequence = contour_image_seq
    rt_ref_series_seq.append(rt_ref_series_item)
    rt_ref_study_item.RTReferencedSeriesSequence = rt_ref_series_seq
    rt_ref_study_seq.append(rt_ref_study_item)
    ref_for_item.RTReferencedStudySequence = rt_ref_study_seq
    ref_for_seq.append(ref_for_item)
    ds.ReferencedFrameOfReferenceSequence = ref_for_seq

    # Structure Set ROI Sequence
    structure_set_roi_seq = Sequence()
    roi_contour_seq = Sequence()
    rt_roi_observations_seq = Sequence()

    # Build CT slice lookup (ImagePositionPatient Z -> SOPInstanceUID)
    ct_slice_lookup = {}
    ct_sop_class_lookup = {}  # Z position -> SOPClassUID
    for ct_file in ct_files:
        ct_ds = pydicom.dcmread(str(ct_file), stop_before_pixels=True)
        if hasattr(ct_ds, 'ImagePositionPatient'):
            z_pos = float(ct_ds.ImagePositionPatient[2])
            ct_slice_lookup[z_pos] = ct_ds.SOPInstanceUID
            ct_sop_class_lookup[z_pos] = ct_ds.SOPClassUID

    if not ct_slice_lookup:
        raise ValueError("No CT slices with ImagePositionPatient found")

    for roi_number, (name, data) in enumerate(structures.items(), start=1):
        contours = data.get("contours", {})
        color = data.get("color", (255, 0, 0))

        # Structure Set ROI
        roi_item = Dataset()
        roi_item.ROINumber = roi_number
        roi_item.ReferencedFrameOfReferenceUID = ref_for_item.FrameOfReferenceUID
        roi_item.ROIName = name
        roi_item.ROIGenerationAlgorithm = "AUTOMATIC"
        structure_set_roi_seq.append(roi_item)

        # ROI Contour
        roi_contour_item = Dataset()
        roi_contour_item.ROIDisplayColor = list(color)
        roi_contour_item.ReferencedROINumber = roi_number

        contour_seq = Sequence()
        for slice_idx, slice_contours in contours.items():
            for contour_points in slice_contours:
                if len(contour_points) == 0:
                    continue

                contour_item = Dataset()
                contour_item.ContourGeometricType = "CLOSED_PLANAR"
                contour_item.NumberOfContourPoints = len(contour_points)

                # Flatten to DICOM format [x1,y1,z1,x2,y2,z2,...]
                contour_data = []
                for point in contour_points:
                    contour_data.extend([float(point[0]), float(point[1]), float(point[2])])
                contour_item.ContourData = contour_data

                # Reference CT slice (find closest Z)
                z_val = float(contour_points[0][2])
                closest_z = min(ct_slice_lookup.keys(), key=lambda x: abs(x - z_val))

                contour_image_ref_seq = Sequence()
                ref_item = Dataset()
                # Use SOPClassUID from the matched CT slice
                ref_item.ReferencedSOPClassUID = ct_sop_class_lookup.get(closest_z, ref_ds.SOPClassUID)
                ref_item.ReferencedSOPInstanceUID = ct_slice_lookup[closest_z]
                contour_image_ref_seq.append(ref_item)
                contour_item.ContourImageSequence = contour_image_ref_seq

                contour_seq.append(contour_item)

        roi_contour_item.ContourSequence = contour_seq
        roi_contour_seq.append(roi_contour_item)

        # RT ROI Observations
        obs_item = Dataset()
        obs_item.ObservationNumber = roi_number
        obs_item.ReferencedROINumber = roi_number
        obs_item.RTROIInterpretedType = "ORGAN"
        obs_item.ROIInterpreter = ""
        rt_roi_observations_seq.append(obs_item)

    ds.StructureSetROISequence = structure_set_roi_seq
    ds.ROIContourSequence = roi_contour_seq
    ds.RTROIObservationsSequence = rt_roi_observations_seq

    return ds


def nifti_masks_to_rtss(
    mask_files: Dict[str, str],
    reference_ct_dir: str,
    output_path: str,
    colors: Dict[str, Tuple[int, int, int]] = None,
    study_uid: str = None,
    patient_name: str = "Anonymous",
) -> str:
    """
    Convert multiple NIfTI masks to a single DICOM RTSS file.

    Args:
        mask_files: {"structure_name": "/path/to/mask.nii.gz"}
        reference_ct_dir: CT DICOM series directory
        output_path: Output RTSS file path
        colors: {"structure_name": (R, G, B)} color map
        study_uid: StudyInstanceUID
        patient_name: Patient name

    Returns:
        Generated RTSS file path
    """
    default_colors = STRUCTURE_COLORS
    structures = {}

    for name, mask_path in mask_files.items():
        try:
            nii = nib.load(mask_path)
            mask_data = nii.get_fdata()
            affine = nii.affine

            # Binarize mask (in case it's not already binary)
            mask_data = (mask_data > 0).astype(np.float32)

            contours = extract_contours_from_mask(mask_data, affine)

            if contours:
                # Determine color: provided colors > nerve color function > default colors > red fallback
                color = (255, 0, 0)  # Default red
                name_lower = name.lower()

                if colors and name in colors:
                    color = colors[name]
                elif colors and name_lower in colors:
                    color = colors[name_lower]
                elif "nerve" in name_lower:
                    # Use get_nerve_color for proper nerve color matching
                    from nerve_to_dicom import get_nerve_color
                    color = get_nerve_color(name)
                elif name in default_colors:
                    color = default_colors[name]
                elif name_lower in default_colors:
                    color = default_colors[name_lower]

                structures[name] = {
                    "contours": contours,
                    "color": color,
                }
                print(f"[RTSS] Extracted {sum(len(c) for c in contours.values())} contours from {name}")
            else:
                print(f"[RTSS] No valid contours from {name}")

        except Exception as e:
            print(f"[RTSS] Error processing {name}: {e}")
            continue

    if not structures:
        raise ValueError("No valid contours extracted from any masks")

    ds = create_rtss_dataset(
        reference_ct_dir=reference_ct_dir,
        structures=structures,
        study_uid=study_uid,
        patient_name=patient_name,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(output_path, write_like_original=False)

    print(f"[RTSS] Created RTSS file with {len(structures)} structures: {output_path}")
    return output_path


def find_existing_rtstruct(study_uid: str, orthanc_url: str) -> Optional[str]:
    """Find existing RTSTRUCT series in Orthanc for a study.

    Args:
        study_uid: StudyInstanceUID
        orthanc_url: Orthanc server URL

    Returns:
        Orthanc series ID of the RTSTRUCT, or None if not found
    """
    try:
        # Search for RTSTRUCT in the study
        response = requests.post(
            f"{orthanc_url}/tools/find",
            json={
                "Level": "Series",
                "Query": {
                    "StudyInstanceUID": study_uid,
                    "Modality": "RTSTRUCT",
                },
            },
            timeout=30,
        )

        if response.status_code == 200:
            series_ids = response.json()
            if series_ids:
                print(f"[RTSS] Found existing RTSTRUCT: {series_ids[0]}")
                return series_ids[0]

        return None
    except Exception as e:
        print(f"[RTSS] Error searching for existing RTSTRUCT: {e}")
        return None


def download_rtstruct(series_id: str, orthanc_url: str, output_path: str) -> Optional[str]:
    """Download RTSTRUCT from Orthanc.

    Args:
        series_id: Orthanc series ID
        orthanc_url: Orthanc server URL
        output_path: Output file path

    Returns:
        Path to downloaded file, or None if failed
    """
    try:
        # Get instances in the series
        response = requests.get(f"{orthanc_url}/series/{series_id}/instances", timeout=30)
        if response.status_code != 200:
            return None

        instances = response.json()
        if not instances:
            return None

        # Download the first (usually only) instance
        instance_id = instances[0]["ID"]
        response = requests.get(f"{orthanc_url}/instances/{instance_id}/file", timeout=60)

        if response.status_code == 200:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"[RTSS] Downloaded existing RTSTRUCT to: {output_path}")
            return output_path

        return None
    except Exception as e:
        print(f"[RTSS] Error downloading RTSTRUCT: {e}")
        return None


def delete_rtstruct_from_orthanc(series_id: str, orthanc_url: str) -> bool:
    """Delete RTSTRUCT series from Orthanc.

    Args:
        series_id: Orthanc series ID
        orthanc_url: Orthanc server URL

    Returns:
        True if deleted successfully
    """
    try:
        response = requests.delete(f"{orthanc_url}/series/{series_id}", timeout=30)
        if response.status_code == 200:
            print(f"[RTSS] Deleted old RTSTRUCT from Orthanc: {series_id}")
            return True
        return False
    except Exception as e:
        print(f"[RTSS] Error deleting RTSTRUCT: {e}")
        return False


def append_structures_to_rtss(
    existing_rtss_path: str,
    new_structures: Dict[str, Dict],
    reference_ct_dir: str,
    output_path: str,
) -> str:
    """Append new structures to an existing RTSS file.

    Args:
        existing_rtss_path: Path to existing RTSS file
        new_structures: {
            "structure_name": {
                "contours": {slice_idx: [contour_coords, ...]},
                "color": (R, G, B),
            }
        }
        reference_ct_dir: CT DICOM series directory (for slice lookup)
        output_path: Output path for updated RTSS

    Returns:
        Path to updated RTSS file
    """
    ds = pydicom.dcmread(existing_rtss_path)

    existing_roi_numbers = []
    existing_roi_names = set()

    if hasattr(ds, 'StructureSetROISequence'):
        for roi in ds.StructureSetROISequence:
            existing_roi_numbers.append(roi.ROINumber)
            existing_roi_names.add(roi.ROIName.lower())

    next_roi_number = max(existing_roi_numbers) + 1 if existing_roi_numbers else 1

    ct_files = sorted(Path(reference_ct_dir).glob("*.dcm"))
    ct_slice_lookup = {}
    ct_sop_class_lookup = {}

    for ct_file in ct_files:
        ct_ds = pydicom.dcmread(str(ct_file), stop_before_pixels=True)
        if hasattr(ct_ds, 'ImagePositionPatient'):
            z_pos = float(ct_ds.ImagePositionPatient[2])
            ct_slice_lookup[z_pos] = ct_ds.SOPInstanceUID
            ct_sop_class_lookup[z_pos] = ct_ds.SOPClassUID

    frame_of_ref_uid = None
    if hasattr(ds, 'ReferencedFrameOfReferenceSequence') and ds.ReferencedFrameOfReferenceSequence:
        frame_of_ref_uid = ds.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID

    ref_ct = pydicom.dcmread(str(ct_files[0]), stop_before_pixels=True) if ct_files else None

    added_count = 0
    for name, data in new_structures.items():
        if name.lower() in existing_roi_names:
            print(f"[RTSS] Structure '{name}' already exists, skipping")
            continue

        contours = data.get("contours", {})
        color = data.get("color", (255, 0, 0))

        if not contours:
            continue

        roi_number = next_roi_number
        next_roi_number += 1

        roi_item = Dataset()
        roi_item.ROINumber = roi_number
        roi_item.ReferencedFrameOfReferenceUID = frame_of_ref_uid or generate_uid()
        roi_item.ROIName = name
        roi_item.ROIGenerationAlgorithm = "AUTOMATIC"
        ds.StructureSetROISequence.append(roi_item)

        roi_contour_item = Dataset()
        roi_contour_item.ROIDisplayColor = list(color)
        roi_contour_item.ReferencedROINumber = roi_number

        contour_seq = Sequence()
        for slice_idx, slice_contours in contours.items():
            for contour_points in slice_contours:
                if len(contour_points) == 0:
                    continue

                contour_item = Dataset()
                contour_item.ContourGeometricType = "CLOSED_PLANAR"
                contour_item.NumberOfContourPoints = len(contour_points)

                contour_data = []
                for point in contour_points:
                    contour_data.extend([float(point[0]), float(point[1]), float(point[2])])
                contour_item.ContourData = contour_data

                z_val = float(contour_points[0][2])
                if ct_slice_lookup:
                    closest_z = min(ct_slice_lookup.keys(), key=lambda x: abs(x - z_val))

                    contour_image_ref_seq = Sequence()
                    ref_item = Dataset()
                    ref_item.ReferencedSOPClassUID = ct_sop_class_lookup.get(closest_z, "1.2.840.10008.5.1.4.1.1.2")
                    ref_item.ReferencedSOPInstanceUID = ct_slice_lookup[closest_z]
                    contour_image_ref_seq.append(ref_item)
                    contour_item.ContourImageSequence = contour_image_ref_seq

                contour_seq.append(contour_item)

        roi_contour_item.ContourSequence = contour_seq
        ds.ROIContourSequence.append(roi_contour_item)

        obs_item = Dataset()
        obs_item.ObservationNumber = roi_number
        obs_item.ReferencedROINumber = roi_number
        obs_item.RTROIInterpretedType = "ORGAN"
        obs_item.ROIInterpreter = ""
        ds.RTROIObservationsSequence.append(obs_item)

        added_count += 1
        print(f"[RTSS] Added structure '{name}' (ROI #{roi_number})")

    ds.StructureSetDate = datetime.date.today().strftime('%Y%m%d')
    ds.StructureSetTime = datetime.datetime.now().strftime('%H%M%S')
    ds.StructureSetDescription = "RT Structure Set - Updated with Nerve Assessment"

    # Generate new SOPInstanceUID (it's a modified file)
    ds.SOPInstanceUID = generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(output_path, write_like_original=False)

    print(f"[RTSS] Updated RTSS with {added_count} new structures: {output_path}")
    return output_path


def add_nerves_to_existing_rtss(
    nerve_mask_files: Dict[str, str],
    reference_ct_dir: str,
    output_path: str,
    study_uid: str,
    orthanc_url: str,
    colors: Dict[str, Tuple[int, int, int]] = None,
) -> Optional[str]:
    """Add nerve structures to existing RTSTRUCT in Orthanc.

    This function:
    1. Finds existing RTSTRUCT in Orthanc
    2. Downloads it
    3. Extracts contours from nerve masks
    4. Appends nerve structures to the RTSTRUCT
    5. Saves updated file

    Args:
        nerve_mask_files: {"nerve_name": "/path/to/mask.nii.gz"}
        reference_ct_dir: CT DICOM series directory
        output_path: Output RTSS file path
        study_uid: StudyInstanceUID
        orthanc_url: Orthanc server URL
        colors: Optional color map

    Returns:
        Path to updated RTSS file, or None if no existing RTSTRUCT found
    """
    # Step 1: Find existing RTSTRUCT
    existing_series_id = find_existing_rtstruct(study_uid, orthanc_url)

    if not existing_series_id:
        print("[RTSS] No existing RTSTRUCT found, cannot append")
        return None

    # Step 2: Download existing RTSTRUCT
    temp_rtss_path = str(Path(output_path).parent / "existing_rtss.dcm")
    downloaded = download_rtstruct(existing_series_id, orthanc_url, temp_rtss_path)

    if not downloaded:
        print("[RTSS] Failed to download existing RTSTRUCT")
        return None

    # Step 3: Extract contours from nerve masks
    default_colors = STRUCTURE_COLORS
    new_structures = {}

    for name, mask_path in nerve_mask_files.items():
        try:
            nii = nib.load(mask_path)
            mask_data = nii.get_fdata()
            affine = nii.affine

            mask_data = (mask_data > 0).astype(np.float32)
            contours = extract_contours_from_mask(mask_data, affine)

            if contours:
                # Use get_nerve_color for proper color matching
                from nerve_to_dicom import get_nerve_color
                color = get_nerve_color(name)

                new_structures[name] = {
                    "contours": contours,
                    "color": color,
                }
                print(f"[RTSS] Extracted {sum(len(c) for c in contours.values())} contours from {name} (color: {color})")
        except Exception as e:
            print(f"[RTSS] Error processing nerve mask {name}: {e}")
            continue

    if not new_structures:
        print("[RTSS] No valid nerve contours extracted")
        return None

    # Step 4: Append to existing RTSS
    updated_path = append_structures_to_rtss(
        existing_rtss_path=temp_rtss_path,
        new_structures=new_structures,
        reference_ct_dir=reference_ct_dir,
        output_path=output_path,
    )

    # Step 5: Delete old RTSTRUCT from Orthanc (will be replaced by upload)
    delete_rtstruct_from_orthanc(existing_series_id, orthanc_url)

    try:
        Path(temp_rtss_path).unlink()
    except OSError:
        pass

    return updated_path
