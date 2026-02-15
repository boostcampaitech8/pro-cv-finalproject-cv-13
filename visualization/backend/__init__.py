"""Backend package for nerve estimation visualization with OHIF/Orthanc integration."""

from .dicom_converter import (
    nifti_to_dicom_series,
    nifti_seg_to_dicom_seg,
    create_dicom_metadata,
    get_structure_color,
    STRUCTURE_COLORS,
)

from .orthanc_client import (
    OrthancClient,
    upload_to_orthanc,
    upload_directory_to_orthanc,
)

from .nerve_to_dicom import (
    nerve_json_to_nifti_masks,
    get_nerve_color,
    NERVE_COLORS,
)

from .pipeline import run_pipeline

__all__ = [
    # DICOM conversion
    "nifti_to_dicom_series",
    "nifti_seg_to_dicom_seg",
    "create_dicom_metadata",
    "get_structure_color",
    "STRUCTURE_COLORS",
    # Orthanc client
    "OrthancClient",
    "upload_to_orthanc",
    "upload_directory_to_orthanc",
    # Nerve conversion
    "nerve_json_to_nifti_masks",
    "get_nerve_color",
    "NERVE_COLORS",
    # Pipeline
    "run_pipeline",
]
