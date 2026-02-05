"""NIfTI to DICOM conversion utilities for OHIF Viewer integration."""

import os
import uuid
import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

try:
    from colors import STRUCTURE_COLORS, get_structure_color
except ImportError:
    from .colors import STRUCTURE_COLORS, get_structure_color

import numpy as np
import nibabel as nib
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import (
    generate_uid,
    ExplicitVRLittleEndian,
    CTImageStorage,
    SegmentationStorage,
)
from pydicom.sequence import Sequence


IMPLEMENTATION_CLASS_UID = "1.2.826.0.1.3680043.8.498.1"
IMPLEMENTATION_VERSION_NAME = "NerveEstimation1.0"


def pack_bits(frame: np.ndarray) -> bytes:
    """Pack binary frame into bit-packed bytes (DICOM BINARY SEG format).

    DICOM SEG BINARY type requires 1 bit per pixel (8 pixels = 1 byte).

    Args:
        frame: 2D numpy array with 0/1 values, shape (rows, cols)

    Returns:
        Bit-packed bytes where 8 pixels = 1 byte
    """
    flat = (frame.flatten() > 0).astype(np.uint8)

    pad_len = (8 - len(flat) % 8) % 8
    if pad_len > 0:
        flat = np.concatenate([flat, np.zeros(pad_len, dtype=np.uint8)])

    # Pack 8 pixels into each byte (LSB first per DICOM standard)
    packed = np.packbits(flat, bitorder='little')
    return packed.tobytes()


def create_dicom_metadata(
    patient_name: str = "Anonymous",
    patient_id: str = None,
    study_description: str = "CT Study",
    series_description: str = "CT Series",
    modality: str = "CT",
) -> Dict[str, Any]:
    """Create common DICOM metadata.

    Args:
        patient_name: Patient name
        patient_id: Patient ID (generated if not provided)
        study_description: Study description
        series_description: Series description
        modality: Imaging modality (CT, MR, etc.)

    Returns:
        Dictionary with DICOM metadata
    """
    now = datetime.datetime.now()

    return {
        "PatientName": patient_name,
        "PatientID": patient_id or str(uuid.uuid4())[:8].upper(),
        "PatientBirthDate": "",
        "PatientSex": "",
        "StudyInstanceUID": generate_uid(),
        "SeriesInstanceUID": generate_uid(),
        "FrameOfReferenceUID": generate_uid(),
        "StudyDate": now.strftime("%Y%m%d"),
        "StudyTime": now.strftime("%H%M%S"),
        "SeriesDate": now.strftime("%Y%m%d"),
        "SeriesTime": now.strftime("%H%M%S"),
        "AccessionNumber": "",
        "StudyDescription": study_description,
        "SeriesDescription": series_description,
        "Modality": modality,
        "Manufacturer": "NerveEstimation",
        "InstitutionName": "Research",
        "SeriesNumber": 1,
    }


def nifti_to_dicom_series(
    nifti_path: str,
    output_dir: str,
    metadata: Dict[str, Any] = None,
    window_center: float = 40,
    window_width: float = 350,
) -> Tuple[str, str, List[str]]:
    """Convert NIfTI volume to DICOM series.

    Args:
        nifti_path: Path to NIfTI file
        output_dir: Output directory for DICOM files
        metadata: Optional DICOM metadata dictionary
        window_center: Default window center (HU)
        window_width: Default window width (HU)

    Returns:
        Tuple of (StudyInstanceUID, SeriesInstanceUID, list of DICOM file paths)
    """
    nii = nib.load(nifti_path)
    data = nii.get_fdata()
    affine = nii.affine
    header = nii.header

    spacing = header.get_zooms()[:3]
    slice_thickness = float(spacing[2])
    pixel_spacing = [float(spacing[0]), float(spacing[1])]

    rotation = affine[:3, :3]
    rotation_normalized = rotation / np.abs(spacing)

    # RAS→LPS: negate X and Y for DICOM
    ras_to_lps = np.diag([-1, -1, 1])
    rotation_lps = ras_to_lps @ rotation_normalized

    row_cosines = rotation_lps[:, 0].tolist()
    col_cosines = rotation_lps[:, 1].tolist()
    image_orientation = row_cosines + col_cosines

    origin = (ras_to_lps @ affine[:3, 3]).tolist()
    slice_direction_lps = ras_to_lps @ affine[:3, 2]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if metadata is None:
        metadata = create_dicom_metadata(
            study_description="CT from NIfTI",
            series_description=Path(nifti_path).stem,
        )

    study_uid = metadata["StudyInstanceUID"]
    series_uid = metadata["SeriesInstanceUID"]
    frame_of_ref_uid = metadata["FrameOfReferenceUID"]

    # Convert data to int16 (assuming HU values)
    data = data.astype(np.float64)

    # Handle NaN values
    data = np.nan_to_num(data, nan=0)

    # Rescale to int16 range if needed
    rescale_intercept = 0
    rescale_slope = 1

    if data.min() < -32768 or data.max() > 32767:
        data_min = data.min()
        data_max = data.max()
        rescale_intercept = data_min
        rescale_slope = (data_max - data_min) / 65535
        data = ((data - data_min) / rescale_slope).astype(np.int16)
    else:
        data = data.astype(np.int16)

    dicom_files = []
    num_slices = data.shape[2]

    for i in range(num_slices):
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = CTImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.ImplementationClassUID = IMPLEMENTATION_CLASS_UID
        file_meta.ImplementationVersionName = IMPLEMENTATION_VERSION_NAME
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(
            filename_or_obj="",
            dataset={},
            file_meta=file_meta,
            preamble=b"\0" * 128,
        )

        # Patient module
        ds.PatientName = metadata["PatientName"]
        ds.PatientID = metadata["PatientID"]
        ds.PatientBirthDate = metadata.get("PatientBirthDate", "")
        ds.PatientSex = metadata.get("PatientSex", "")

        # Study module
        ds.StudyInstanceUID = study_uid
        ds.StudyDate = metadata["StudyDate"]
        ds.StudyTime = metadata["StudyTime"]
        ds.AccessionNumber = metadata.get("AccessionNumber", "")
        ds.StudyDescription = metadata["StudyDescription"]
        ds.StudyID = "1"

        # Series module
        ds.SeriesInstanceUID = series_uid
        ds.SeriesDate = metadata["SeriesDate"]
        ds.SeriesTime = metadata["SeriesTime"]
        ds.SeriesDescription = metadata["SeriesDescription"]
        ds.SeriesNumber = metadata.get("SeriesNumber", 1)
        ds.Modality = metadata["Modality"]
        ds.Manufacturer = metadata.get("Manufacturer", "Unknown")
        ds.InstitutionName = metadata.get("InstitutionName", "")

        # Frame of Reference module
        ds.FrameOfReferenceUID = frame_of_ref_uid
        ds.PositionReferenceIndicator = ""

        # Image module
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.InstanceNumber = i + 1
        ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]

        # Slice position (in LPS)
        slice_position = [
            origin[0] + i * slice_direction_lps[0],
            origin[1] + i * slice_direction_lps[1],
            origin[2] + i * slice_direction_lps[2],
        ]
        ds.ImagePositionPatient = slice_position
        ds.ImageOrientationPatient = image_orientation
        ds.SliceLocation = float(slice_position[2])
        ds.SliceThickness = slice_thickness

        # Pixel data
        ds.Rows = data.shape[1]
        ds.Columns = data.shape[0]
        ds.PixelSpacing = pixel_spacing
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1  # Signed

        # Rescale
        ds.RescaleIntercept = rescale_intercept
        ds.RescaleSlope = rescale_slope
        ds.RescaleType = "HU"

        # Window/Level
        ds.WindowCenter = window_center
        ds.WindowWidth = window_width

        slice_data = data[:, :, i].T  # Transpose for DICOM orientation
        ds.PixelData = slice_data.tobytes()

        filename = f"CT_{i:04d}.dcm"
        filepath = os.path.join(output_dir, filename)
        ds.save_as(filepath, write_like_original=False)
        dicom_files.append(filepath)

    return study_uid, series_uid, dicom_files


def create_segmentation_dataset(
    mask_data: np.ndarray,
    reference_ds: Dataset,
    segment_label: str,
    segment_number: int,
    color: Tuple[int, int, int] = (255, 0, 0),
    algorithm_type: str = "AUTOMATIC",
    all_source_images: List[Dataset] = None,
    study_uid: str = None,
) -> Dataset:
    """Create a DICOM Segmentation dataset.

    Args:
        mask_data: 3D binary mask array
        reference_ds: Reference DICOM dataset for geometry
        segment_label: Label for the segment
        segment_number: Segment number (1-indexed)
        color: RGB color tuple (0-255)
        algorithm_type: Algorithm type (AUTOMATIC, MANUAL, etc.)

    Returns:
        DICOM Segmentation dataset
    """
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SegmentationStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = IMPLEMENTATION_CLASS_UID
    file_meta.ImplementationVersionName = IMPLEMENTATION_VERSION_NAME
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(
        filename_or_obj="",
        dataset={},
        file_meta=file_meta,
        preamble=b"\0" * 128,
    )

    ds.PatientName = reference_ds.PatientName
    ds.PatientID = reference_ds.PatientID
    ds.PatientBirthDate = getattr(reference_ds, "PatientBirthDate", "")
    ds.PatientSex = getattr(reference_ds, "PatientSex", "")
    if study_uid:
        ds.StudyInstanceUID = study_uid
    else:
        ds.StudyInstanceUID = reference_ds.StudyInstanceUID
    ds.StudyDate = reference_ds.StudyDate
    ds.StudyTime = reference_ds.StudyTime
    ds.AccessionNumber = getattr(reference_ds, "AccessionNumber", "")
    ds.StudyDescription = getattr(reference_ds, "StudyDescription", "")
    ds.StudyID = getattr(reference_ds, "StudyID", "1")

    # Frame of Reference
    ds.FrameOfReferenceUID = reference_ds.FrameOfReferenceUID

    # Referenced Series Sequence (required for OHIF to link SEG to CT)
    ref_series_seq = Sequence()
    ref_series_item = Dataset()
    ref_series_item.SeriesInstanceUID = reference_ds.SeriesInstanceUID

    # Referenced Instance Sequence
    ref_instance_seq = Sequence()
    if all_source_images:
        for src_img in all_source_images:
            ref_instance = Dataset()
            ref_instance.ReferencedSOPClassUID = src_img.SOPClassUID
            ref_instance.ReferencedSOPInstanceUID = src_img.SOPInstanceUID
            ref_instance_seq.append(ref_instance)
    else:
        # Fallback: just reference the single reference image
        ref_instance = Dataset()
        ref_instance.ReferencedSOPClassUID = reference_ds.SOPClassUID
        ref_instance.ReferencedSOPInstanceUID = reference_ds.SOPInstanceUID
        ref_instance_seq.append(ref_instance)

    ref_series_item.ReferencedInstanceSequence = ref_instance_seq
    ref_series_seq.append(ref_series_item)
    ds.ReferencedSeriesSequence = ref_series_seq

    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber = int(getattr(reference_ds, "SeriesNumber", 1)) + 100
    ds.SeriesDescription = f"Segmentation - {segment_label}"
    ds.Modality = "SEG"
    ds.Manufacturer = "NerveEstimation"

    ds.SOPClassUID = SegmentationStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.ImageType = ["DERIVED", "PRIMARY"]
    ds.InstanceNumber = 1

    now = datetime.datetime.now()
    ds.ContentDate = now.strftime("%Y%m%d")
    ds.ContentTime = now.strftime("%H%M%S")
    ds.ContentLabel = segment_label.upper().replace(" ", "_")
    ds.ContentDescription = segment_label
    ds.ContentCreatorName = "NerveEstimation"

    # Segmentation specific (BINARY type - no fractional settings)
    ds.SegmentationType = "BINARY"

    segment_sequence = Sequence()
    segment = Dataset()
    segment.SegmentNumber = segment_number
    segment.SegmentLabel = segment_label
    segment.SegmentAlgorithmType = algorithm_type
    segment.SegmentAlgorithmName = "NerveEstimation"
    segment.RecommendedDisplayCIELabValue = rgb_to_cielab(color)

    # Segmented Property Category/Type (required)
    segment.SegmentedPropertyCategoryCodeSequence = Sequence([Dataset()])
    segment.SegmentedPropertyCategoryCodeSequence[0].CodeValue = "T-D0050"
    segment.SegmentedPropertyCategoryCodeSequence[0].CodingSchemeDesignator = "SRT"
    segment.SegmentedPropertyCategoryCodeSequence[0].CodeMeaning = "Tissue"

    segment.SegmentedPropertyTypeCodeSequence = Sequence([Dataset()])
    segment.SegmentedPropertyTypeCodeSequence[0].CodeValue = "T-D0050"
    segment.SegmentedPropertyTypeCodeSequence[0].CodingSchemeDesignator = "SRT"
    segment.SegmentedPropertyTypeCodeSequence[0].CodeMeaning = segment_label

    segment_sequence.append(segment)
    ds.SegmentSequence = segment_sequence

    # Image geometry from reference
    ds.Rows = reference_ds.Rows
    ds.Columns = reference_ds.Columns
    ds.PixelSpacing = reference_ds.PixelSpacing
    ds.SliceThickness = getattr(reference_ds, "SliceThickness", 1.0)

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 1  # BINARY: 1 bit per pixel
    ds.BitsStored = 1
    ds.HighBit = 0
    ds.PixelRepresentation = 0  # Unsigned
    ds.LossyImageCompression = "00"

    # Dimension module (required for OHIF)
    ds.DimensionOrganizationType = "3D"

    # Dimension Organization Sequence
    dim_org_seq = Sequence()
    dim_org_item = Dataset()
    dim_org_item.DimensionOrganizationUID = generate_uid()
    dim_org_seq.append(dim_org_item)
    ds.DimensionOrganizationSequence = dim_org_seq

    # Dimension Index Sequence
    dim_idx_seq = Sequence()

    # First dimension: Referenced Segment Number
    dim_idx_1 = Dataset()
    dim_idx_1.DimensionOrganizationUID = dim_org_item.DimensionOrganizationUID
    dim_idx_1.DimensionIndexPointer = 0x0062000B  # ReferencedSegmentNumber
    dim_idx_1.FunctionalGroupPointer = 0x0062000A  # SegmentIdentificationSequence
    dim_idx_seq.append(dim_idx_1)

    # Second dimension: Image Position Patient (slice position)
    dim_idx_2 = Dataset()
    dim_idx_2.DimensionOrganizationUID = dim_org_item.DimensionOrganizationUID
    dim_idx_2.DimensionIndexPointer = 0x00200032  # ImagePositionPatient
    dim_idx_2.FunctionalGroupPointer = 0x00209113  # PlanePositionSequence
    dim_idx_seq.append(dim_idx_2)

    ds.DimensionIndexSequence = dim_idx_seq

    # Shared Functional Groups Sequence (required for OHIF)
    shared_fg = Dataset()

    # Pixel Measures Sequence - ensure proper format
    pixel_measures = Dataset()
    pixel_spacing = getattr(reference_ds, "PixelSpacing", [1.0, 1.0])
    if hasattr(pixel_spacing, '__iter__'):
        pixel_spacing = [float(x) for x in pixel_spacing]
    else:
        pixel_spacing = [float(pixel_spacing), float(pixel_spacing)]
    pixel_measures.PixelSpacing = pixel_spacing

    slice_thickness = float(getattr(reference_ds, "SliceThickness", 1.0))
    pixel_measures.SliceThickness = slice_thickness
    pixel_measures.SpacingBetweenSlices = slice_thickness
    shared_fg.PixelMeasuresSequence = Sequence([pixel_measures])

    # Plane Orientation Sequence
    plane_orientation = Dataset()
    image_orientation = getattr(reference_ds, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])
    if hasattr(image_orientation, '__iter__'):
        image_orientation = [float(x) for x in image_orientation]
    else:
        image_orientation = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    plane_orientation.ImageOrientationPatient = image_orientation
    shared_fg.PlaneOrientationSequence = Sequence([plane_orientation])

    ds.SharedFunctionalGroupsSequence = Sequence([shared_fg])

    # Per-frame functional groups
    num_slices = mask_data.shape[2]
    per_frame_seq = Sequence()

    sorted_source_images = None
    if all_source_images:
        sorted_source_images = sorted(
            all_source_images,
            key=lambda x: float(x.ImagePositionPatient[2]) if hasattr(x, 'ImagePositionPatient') else 0
        )

    for i in range(num_slices):
        frame_content = Dataset()

        # Frame content sequence
        frame_content_seq = Dataset()
        frame_content_seq.DimensionIndexValues = [1, i + 1]
        frame_content.FrameContentSequence = Sequence([frame_content_seq])

        # Plane position sequence - use actual CT slice positions if available
        plane_pos_seq = Dataset()
        if sorted_source_images and i < len(sorted_source_images):
            # Use actual position from source CT slice
            slice_pos = [float(x) for x in sorted_source_images[i].ImagePositionPatient]
        else:
            # Fallback: calculate from reference
            slice_pos = [
                float(reference_ds.ImagePositionPatient[0]),
                float(reference_ds.ImagePositionPatient[1]),
                float(reference_ds.ImagePositionPatient[2]) + i * float(ds.SliceThickness),
            ]
        plane_pos_seq.ImagePositionPatient = slice_pos
        frame_content.PlanePositionSequence = Sequence([plane_pos_seq])

        # Plane orientation sequence
        plane_orient_seq = Dataset()
        plane_orient_seq.ImageOrientationPatient = reference_ds.ImageOrientationPatient
        frame_content.PlaneOrientationSequence = Sequence([plane_orient_seq])

        # Segment identification
        seg_id_seq = Dataset()
        seg_id_seq.ReferencedSegmentNumber = segment_number
        frame_content.SegmentIdentificationSequence = Sequence([seg_id_seq])

        # Derivation image sequence - reference the specific source CT slice
        if sorted_source_images and i < len(sorted_source_images):
            src_img = sorted_source_images[i]
            derivation_image_seq = Dataset()

            # Source image sequence
            source_image_seq = Dataset()
            source_image_seq.ReferencedSOPClassUID = src_img.SOPClassUID
            source_image_seq.ReferencedSOPInstanceUID = src_img.SOPInstanceUID
            derivation_image_seq.SourceImageSequence = Sequence([source_image_seq])

            frame_content.DerivationImageSequence = Sequence([derivation_image_seq])

        per_frame_seq.append(frame_content)

    ds.PerFrameFunctionalGroupsSequence = per_frame_seq
    ds.NumberOfFrames = num_slices

    # Pack pixel data (all frames) - BINARY uses bit-packing (8 pixels = 1 byte)
    packed_frames = []
    for i in range(num_slices):
        frame = (mask_data[:, :, i].T > 0).astype(np.uint8)
        packed_frames.append(pack_bits(frame))

    pixel_data = b''.join(packed_frames)
    ds.PixelData = pixel_data

    return ds


def rgb_to_cielab(rgb: Tuple[int, int, int]) -> List[int]:
    """Convert RGB to CIELab for DICOM display.

    Args:
        rgb: RGB tuple (0-255)

    Returns:
        CIELab values as list [L, a, b] scaled for DICOM (0-65535)
    """
    r, g, b = [x / 255.0 for x in rgb]

    # Approximate L*a*b*
    L = 0.2126 * r + 0.7152 * g + 0.0722 * b
    a = (r - g) * 0.5 + 0.5
    b_val = (r + g - 2 * b) * 0.25 + 0.5

    return [int(L * 65535), int(a * 65535), int(b_val * 65535)]


def nifti_seg_to_dicom_seg(
    seg_nifti_path: str,
    reference_dicom_dir: str,
    output_path: str,
    segment_label: str,
    segment_number: int = 1,
    color: Tuple[int, int, int] = (255, 0, 0),
    study_uid: str = None,
) -> str:
    """Convert NIfTI segmentation mask to DICOM SEG.

    Uses manual DICOM SEG creation for better OHIF compatibility.

    Args:
        seg_nifti_path: Path to segmentation NIfTI file
        reference_dicom_dir: Directory with reference CT DICOM series
        output_path: Output path for DICOM SEG file
        segment_label: Label for the segment
        segment_number: Segment number
        color: RGB color for display

    Returns:
        Path to created DICOM SEG file
    """
    seg_nii = nib.load(seg_nifti_path)
    seg_data = seg_nii.get_fdata()

    ref_files = sorted(Path(reference_dicom_dir).glob("*.dcm"))
    if not ref_files:
        raise ValueError(f"No DICOM files found in {reference_dicom_dir}")

    source_images = [pydicom.dcmread(str(f)) for f in ref_files]
    if np.sum(seg_data > 0) == 0:
        # Create empty file marker
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).touch()
        return output_path

    # Use manual DICOM SEG creation for OHIF compatibility
    ref_ds = source_images[0]
    seg_ds = create_segmentation_dataset(
        mask_data=seg_data,
        reference_ds=ref_ds,
        segment_label=segment_label,
        segment_number=segment_number,
        color=color,
        all_source_images=source_images,
        study_uid=study_uid,
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    seg_ds.save_as(output_path, write_like_original=False)

    return output_path


def create_multi_segment_dicom_seg(
    segment_masks: List[Dict[str, Any]],
    reference_dicom_dir: str,
    output_path: str,
    study_uid: str = None,
) -> str:
    """Create a single DICOM SEG file with multiple segments.

    This is required for OHIF to display all segments in the right panel
    with on/off toggle capability.

    Args:
        segment_masks: List of dicts with keys:
            - 'nifti_path': Path to NIfTI mask file
            - 'label': Segment label (e.g., 'Trachea')
            - 'color': RGB tuple (e.g., (0, 255, 255))
        reference_dicom_dir: Directory with reference CT DICOM series
        output_path: Output path for combined DICOM SEG file
        study_uid: Optional StudyInstanceUID to use (for DICOM workflow)

    Returns:
        Path to created DICOM SEG file
    """
    ref_files = sorted(Path(reference_dicom_dir).glob("*.dcm"))
    if not ref_files:
        raise ValueError(f"No DICOM files found in {reference_dicom_dir}")

    source_images = [pydicom.dcmread(str(f)) for f in ref_files]
    sorted_source_images = sorted(
        source_images,
        key=lambda x: float(x.ImagePositionPatient[2]) if hasattr(x, 'ImagePositionPatient') else 0
    )

    reference_ds = sorted_source_images[0]
    num_slices = len(sorted_source_images)

    valid_segments = []
    for seg_info in segment_masks:
        seg_nii = nib.load(seg_info['nifti_path'])
        seg_data = seg_nii.get_fdata()
        if np.sum(seg_data > 0) > 0:  # Skip empty masks
            valid_segments.append({
                'data': seg_data,
                'label': seg_info['label'],
                'color': seg_info['color'],
            })

    if not valid_segments:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).touch()
        return output_path

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SegmentationStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = IMPLEMENTATION_CLASS_UID
    file_meta.ImplementationVersionName = IMPLEMENTATION_VERSION_NAME
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(
        filename_or_obj="",
        dataset={},
        file_meta=file_meta,
        preamble=b"\0" * 128,
    )
    ds.PatientName = reference_ds.PatientName
    ds.PatientID = reference_ds.PatientID
    ds.PatientBirthDate = getattr(reference_ds, "PatientBirthDate", "")
    ds.PatientSex = getattr(reference_ds, "PatientSex", "")
    ds.StudyInstanceUID = study_uid if study_uid else reference_ds.StudyInstanceUID
    ds.StudyDate = reference_ds.StudyDate
    ds.StudyTime = reference_ds.StudyTime
    ds.AccessionNumber = getattr(reference_ds, "AccessionNumber", "")
    ds.StudyDescription = getattr(reference_ds, "StudyDescription", "")
    ds.StudyID = getattr(reference_ds, "StudyID", "1")

    # Frame of Reference
    ds.FrameOfReferenceUID = reference_ds.FrameOfReferenceUID

    # Referenced Series Sequence
    ref_series_seq = Sequence()
    ref_series_item = Dataset()
    ref_series_item.SeriesInstanceUID = reference_ds.SeriesInstanceUID

    ref_instance_seq = Sequence()
    for src_img in sorted_source_images:
        ref_instance = Dataset()
        ref_instance.ReferencedSOPClassUID = src_img.SOPClassUID
        ref_instance.ReferencedSOPInstanceUID = src_img.SOPInstanceUID
        ref_instance_seq.append(ref_instance)

    ref_series_item.ReferencedInstanceSequence = ref_instance_seq
    ref_series_seq.append(ref_series_item)
    ds.ReferencedSeriesSequence = ref_series_seq

    # Series info (single series for all segments)
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber = int(getattr(reference_ds, "SeriesNumber", 1)) + 100
    ds.SeriesDescription = "Segmentation - All Structures"
    ds.Modality = "SEG"
    ds.Manufacturer = "NerveEstimation"

    ds.SOPClassUID = SegmentationStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.ImageType = ["DERIVED", "PRIMARY"]
    ds.InstanceNumber = 1

    # Content info
    now = datetime.datetime.now()
    ds.ContentDate = now.strftime("%Y%m%d")
    ds.ContentTime = now.strftime("%H%M%S")
    ds.ContentLabel = "SEGMENTATION"
    ds.ContentDescription = "Multi-structure Segmentation"
    ds.ContentCreatorName = "NerveEstimation"

    # Segmentation type
    ds.SegmentationType = "BINARY"

    segment_sequence = Sequence()
    for seg_idx, seg_info in enumerate(valid_segments):
        segment = Dataset()
        segment.SegmentNumber = seg_idx + 1
        segment.SegmentLabel = seg_info['label']
        segment.SegmentAlgorithmType = "AUTOMATIC"
        segment.SegmentAlgorithmName = "NerveEstimation"
        segment.RecommendedDisplayCIELabValue = rgb_to_cielab(seg_info['color'])

        segment.SegmentedPropertyCategoryCodeSequence = Sequence([Dataset()])
        segment.SegmentedPropertyCategoryCodeSequence[0].CodeValue = "T-D0050"
        segment.SegmentedPropertyCategoryCodeSequence[0].CodingSchemeDesignator = "SRT"
        segment.SegmentedPropertyCategoryCodeSequence[0].CodeMeaning = "Tissue"

        segment.SegmentedPropertyTypeCodeSequence = Sequence([Dataset()])
        segment.SegmentedPropertyTypeCodeSequence[0].CodeValue = "T-D0050"
        segment.SegmentedPropertyTypeCodeSequence[0].CodingSchemeDesignator = "SRT"
        segment.SegmentedPropertyTypeCodeSequence[0].CodeMeaning = seg_info['label']

        segment_sequence.append(segment)

    ds.SegmentSequence = segment_sequence

    # Image geometry
    ds.Rows = reference_ds.Rows
    ds.Columns = reference_ds.Columns
    ds.PixelSpacing = reference_ds.PixelSpacing
    ds.SliceThickness = getattr(reference_ds, "SliceThickness", 1.0)
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 1  # BINARY: 1 bit per pixel
    ds.BitsStored = 1
    ds.HighBit = 0
    ds.PixelRepresentation = 0
    ds.LossyImageCompression = "00"

    # Dimension module
    ds.DimensionOrganizationType = "3D"

    dim_org_uid = generate_uid()
    dim_org_seq = Sequence()
    dim_org_item = Dataset()
    dim_org_item.DimensionOrganizationUID = dim_org_uid
    dim_org_seq.append(dim_org_item)
    ds.DimensionOrganizationSequence = dim_org_seq

    # Dimension Index Sequence
    dim_idx_seq = Sequence()

    dim_idx_1 = Dataset()
    dim_idx_1.DimensionOrganizationUID = dim_org_uid
    dim_idx_1.DimensionIndexPointer = 0x0062000B  # ReferencedSegmentNumber
    dim_idx_1.FunctionalGroupPointer = 0x0062000A  # SegmentIdentificationSequence
    dim_idx_seq.append(dim_idx_1)

    dim_idx_2 = Dataset()
    dim_idx_2.DimensionOrganizationUID = dim_org_uid
    dim_idx_2.DimensionIndexPointer = 0x00200032  # ImagePositionPatient
    dim_idx_2.FunctionalGroupPointer = 0x00209113  # PlanePositionSequence
    dim_idx_seq.append(dim_idx_2)

    ds.DimensionIndexSequence = dim_idx_seq

    # Shared Functional Groups
    shared_fg = Dataset()

    pixel_measures = Dataset()
    pixel_spacing = getattr(reference_ds, "PixelSpacing", [1.0, 1.0])
    pixel_measures.PixelSpacing = [float(x) for x in pixel_spacing]
    slice_thickness = float(getattr(reference_ds, "SliceThickness", 1.0))
    pixel_measures.SliceThickness = slice_thickness
    pixel_measures.SpacingBetweenSlices = slice_thickness
    shared_fg.PixelMeasuresSequence = Sequence([pixel_measures])

    plane_orientation = Dataset()
    image_orientation = getattr(reference_ds, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])
    plane_orientation.ImageOrientationPatient = [float(x) for x in image_orientation]
    shared_fg.PlaneOrientationSequence = Sequence([plane_orientation])

    ds.SharedFunctionalGroupsSequence = Sequence([shared_fg])

    # Per-frame functional groups (DENSE: include ALL frames for OHIF compatibility)
    # OHIF/Cornerstone requires all frames present even if empty
    per_frame_seq = Sequence()
    all_frames = []

    for seg_idx, seg_info in enumerate(valid_segments):
        seg_data = seg_info['data']
        segment_number = seg_idx + 1

        for slice_idx in range(num_slices):
            # Extract frame data (DENSE: include even if empty)
            frame = (seg_data[:, :, slice_idx].T > 0).astype(np.uint8)

            frame_content = Dataset()

            # Frame content
            frame_content_seq = Dataset()
            frame_content_seq.DimensionIndexValues = [segment_number, slice_idx + 1]
            frame_content.FrameContentSequence = Sequence([frame_content_seq])

            # Plane position
            src_img = sorted_source_images[slice_idx]
            plane_pos_seq = Dataset()
            plane_pos_seq.ImagePositionPatient = [float(x) for x in src_img.ImagePositionPatient]
            frame_content.PlanePositionSequence = Sequence([plane_pos_seq])

            # Segment identification
            seg_id_seq = Dataset()
            seg_id_seq.ReferencedSegmentNumber = segment_number
            frame_content.SegmentIdentificationSequence = Sequence([seg_id_seq])

            # Derivation image
            derivation_image_seq = Dataset()
            source_image_seq = Dataset()
            source_image_seq.ReferencedSOPClassUID = src_img.SOPClassUID
            source_image_seq.ReferencedSOPInstanceUID = src_img.SOPInstanceUID
            derivation_image_seq.SourceImageSequence = Sequence([source_image_seq])
            frame_content.DerivationImageSequence = Sequence([derivation_image_seq])

            per_frame_seq.append(frame_content)
            all_frames.append(frame)

    ds.PerFrameFunctionalGroupsSequence = per_frame_seq
    ds.NumberOfFrames = len(all_frames)

    # Pack pixel data - BINARY uses bit-packing (8 pixels = 1 byte)
    # DENSE encoding: all frames included for OHIF compatibility
    packed_frames = [pack_bits(frame) for frame in all_frames]
    pixel_data = b''.join(packed_frames)
    ds.PixelData = pixel_data

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(output_path, write_like_original=False)

    return output_path
