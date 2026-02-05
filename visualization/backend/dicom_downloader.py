"""DICOM download and conversion utilities for Orthanc integration."""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import requests
import nibabel as nib


class DicomDownloader:
    """Client for downloading DICOM files from Orthanc and converting to NIfTI."""

    def __init__(
        self,
        orthanc_url: str = None,
        username: str = None,
        password: str = None,
        timeout: int = 60,
    ):
        """Initialize DICOM downloader.

        Args:
            orthanc_url: Orthanc server URL (default: from ORTHANC_URL env)
            username: Optional username for authentication
            password: Optional password for authentication
            timeout: Request timeout in seconds
        """
        self.url = orthanc_url or os.environ.get("ORTHANC_URL", "http://orthanc:8042")
        self.url = self.url.rstrip("/")
        self.timeout = timeout

        self.auth = None
        if username and password:
            self.auth = (username, password)

    def get_study_by_uid(self, study_instance_uid: str) -> Optional[Dict[str, Any]]:
        """Find Orthanc study ID by DICOM StudyInstanceUID.

        Args:
            study_instance_uid: DICOM StudyInstanceUID

        Returns:
            Study details dictionary or None if not found
        """
        try:
            response = requests.post(
                f"{self.url}/tools/find",
                json={
                    "Level": "Study",
                    "Query": {"StudyInstanceUID": study_instance_uid},
                },
                auth=self.auth,
                timeout=self.timeout,
            )

            if response.status_code == 200:
                results = response.json()
                if results:
                    study_response = requests.get(
                        f"{self.url}/studies/{results[0]}",
                        auth=self.auth,
                        timeout=self.timeout,
                    )
                    if study_response.status_code == 200:
                        return study_response.json()
            return None
        except requests.RequestException:
            return None

    def get_series_for_study(self, orthanc_study_id: str) -> List[Dict[str, Any]]:
        """Get all series for a study.

        Args:
            orthanc_study_id: Orthanc study ID

        Returns:
            List of series details
        """
        try:
            response = requests.get(
                f"{self.url}/studies/{orthanc_study_id}",
                auth=self.auth,
                timeout=self.timeout,
            )

            if response.status_code == 200:
                study = response.json()
                series_list = []
                for series_id in study.get("Series", []):
                    series_response = requests.get(
                        f"{self.url}/series/{series_id}",
                        auth=self.auth,
                        timeout=self.timeout,
                    )
                    if series_response.status_code == 200:
                        series_list.append(series_response.json())
                return series_list
            return []
        except requests.RequestException:
            return []

    def find_ct_series(self, series_list: List[Dict[str, Any]]) -> Optional[str]:
        """Find CT series from a list of series.

        Args:
            series_list: List of series details from get_series_for_study

        Returns:
            Orthanc series ID for the CT series, or None if not found
        """
        for series in series_list:
            main_tags = series.get("MainDicomTags", {})
            modality = main_tags.get("Modality", "")
            if modality == "CT":
                return series.get("ID")

        for series in series_list:
            main_tags = series.get("MainDicomTags", {})
            modality = main_tags.get("Modality", "")
            if modality not in ["SEG", "RTSS", "RTSTRUCT", "SR"]:
                return series.get("ID")

        return None

    def download_series(
        self,
        orthanc_series_id: str,
        output_dir: str,
    ) -> List[str]:
        """Download all DICOM instances for a series.

        Args:
            orthanc_series_id: Orthanc series ID
            output_dir: Directory to save DICOM files

        Returns:
            List of downloaded DICOM file paths
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        downloaded_files = []

        try:
            response = requests.get(
                f"{self.url}/series/{orthanc_series_id}",
                auth=self.auth,
                timeout=self.timeout,
            )

            if response.status_code != 200:
                raise ValueError(f"Failed to get series: {response.status_code}")

            series = response.json()
            instances = series.get("Instances", [])

            for i, instance_id in enumerate(instances):
                file_response = requests.get(
                    f"{self.url}/instances/{instance_id}/file",
                    auth=self.auth,
                    timeout=self.timeout,
                )

                if file_response.status_code == 200:
                    filepath = Path(output_dir) / f"{i:04d}.dcm"
                    with open(filepath, "wb") as f:
                        f.write(file_response.content)
                    downloaded_files.append(str(filepath))

            return downloaded_files

        except requests.RequestException as e:
            raise ValueError(f"Failed to download series: {e}")

    def download_study_archive(
        self,
        orthanc_study_id: str,
        output_path: str,
    ) -> str:
        """Download entire study as a ZIP archive.

        Args:
            orthanc_study_id: Orthanc study ID
            output_path: Path for the output ZIP file

        Returns:
            Path to the downloaded ZIP file
        """
        try:
            response = requests.get(
                f"{self.url}/studies/{orthanc_study_id}/archive",
                auth=self.auth,
                timeout=self.timeout * 5,  # Longer timeout for archive
                stream=True,
            )

            if response.status_code != 200:
                raise ValueError(f"Failed to get study archive: {response.status_code}")

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return output_path

        except requests.RequestException as e:
            raise ValueError(f"Failed to download study archive: {e}")


def dicom_to_nifti_simpleitk(
    dicom_dir: str,
    output_path: str,
) -> str:
    """Convert DICOM series to NIfTI using SimpleITK.

    Args:
        dicom_dir: Directory containing DICOM files
        output_path: Output path for NIfTI file

    Returns:
        Path to the created NIfTI file
    """
    try:
        import SimpleITK as sitk
    except ImportError:
        raise ImportError("SimpleITK is required for DICOM to NIfTI conversion. Install with: pip install SimpleITK")

    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(dicom_dir)

    if not dicom_files:
        raise ValueError(f"No DICOM files found in {dicom_dir}")

    reader.SetFileNames(dicom_files)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    image = reader.Execute()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, output_path)

    return output_path


def dicom_to_nifti_dcm2niix(
    dicom_dir: str,
    output_dir: str,
    output_name: str = "image",
) -> str:
    """Convert DICOM series to NIfTI using dcm2niix.

    Args:
        dicom_dir: Directory containing DICOM files
        output_dir: Output directory for NIfTI file
        output_name: Base name for output file

    Returns:
        Path to the created NIfTI file
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        "dcm2niix",
        "-z", "y",          # Compress output
        "-f", output_name,  # Output filename
        "-o", output_dir,   # Output directory
        "-b", "n",          # No BIDS sidecar
        dicom_dir,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise ValueError(f"dcm2niix failed: {e.stderr}")
    except FileNotFoundError:
        raise ValueError("dcm2niix not found. Install with: apt-get install dcm2niix")

    nifti_path = Path(output_dir) / f"{output_name}.nii.gz"
    if not nifti_path.exists():
        # Try without .gz
        nifti_path = Path(output_dir) / f"{output_name}.nii"
        if not nifti_path.exists():
            # Look for any NIfTI file
            nifti_files = list(Path(output_dir).glob("*.nii*"))
            if nifti_files:
                nifti_path = nifti_files[0]
            else:
                raise ValueError(f"No NIfTI file found after conversion in {output_dir}")

    return str(nifti_path)


def dicom_to_nifti(
    dicom_dir: str,
    output_path: str,
    method: str = "auto",
) -> str:
    """Convert DICOM series to NIfTI.

    Tries dcm2niix first (better quality), falls back to SimpleITK.

    Args:
        dicom_dir: Directory containing DICOM files
        output_path: Output path for NIfTI file
        method: Conversion method - "auto", "dcm2niix", or "simpleitk"

    Returns:
        Path to the created NIfTI file
    """
    output_dir = str(Path(output_path).parent)
    output_name = Path(output_path).stem.replace(".nii", "")

    if method == "simpleitk":
        return dicom_to_nifti_simpleitk(dicom_dir, output_path)

    if method == "dcm2niix":
        return dicom_to_nifti_dcm2niix(dicom_dir, output_dir, output_name)

    # Auto: try dcm2niix first, fall back to SimpleITK
    try:
        result_path = dicom_to_nifti_dcm2niix(dicom_dir, output_dir, output_name)
        # Rename if needed
        if result_path != output_path:
            import shutil
            shutil.move(result_path, output_path)
        return output_path
    except (ValueError, FileNotFoundError):
        pass

    # Fall back to SimpleITK
    return dicom_to_nifti_simpleitk(dicom_dir, output_path)


def download_study_from_orthanc(
    study_instance_uid: str,
    output_dir: str,
    orthanc_url: str = None,
) -> Tuple[str, Dict[str, Any]]:
    """Download DICOM study from Orthanc by StudyInstanceUID.

    Args:
        study_instance_uid: DICOM StudyInstanceUID
        output_dir: Directory to save DICOM files
        orthanc_url: Orthanc server URL

    Returns:
        Tuple of (path to CT DICOM directory, study info dict)
    """
    downloader = DicomDownloader(orthanc_url=orthanc_url)

    # Find study
    study = downloader.get_study_by_uid(study_instance_uid)
    if not study:
        raise ValueError(f"Study not found: {study_instance_uid}")

    orthanc_study_id = study.get("ID")

    # Get series
    series_list = downloader.get_series_for_study(orthanc_study_id)
    if not series_list:
        raise ValueError(f"No series found in study: {study_instance_uid}")

    # Find CT series
    ct_series_id = downloader.find_ct_series(series_list)
    if not ct_series_id:
        raise ValueError(f"No CT series found in study: {study_instance_uid}")

    # Download CT series
    ct_dicom_dir = Path(output_dir) / "dicom" / "ct"
    downloaded_files = downloader.download_series(ct_series_id, str(ct_dicom_dir))

    if not downloaded_files:
        raise ValueError(f"Failed to download DICOM files for series: {ct_series_id}")

    study_info = {
        "study_instance_uid": study_instance_uid,
        "orthanc_study_id": orthanc_study_id,
        "ct_series_id": ct_series_id,
        "ct_dicom_dir": str(ct_dicom_dir),
        "num_files": len(downloaded_files),
        "series_list": series_list,
    }

    return str(ct_dicom_dir), study_info


def download_and_convert_to_nifti(
    study_instance_uid: str,
    output_dir: str,
    orthanc_url: str = None,
) -> Tuple[str, Dict[str, Any]]:
    """Download DICOM study from Orthanc and convert to NIfTI.

    Args:
        study_instance_uid: DICOM StudyInstanceUID
        output_dir: Output directory
        orthanc_url: Orthanc server URL

    Returns:
        Tuple of (path to NIfTI file, study info dict)
    """
    # Download DICOM
    ct_dicom_dir, study_info = download_study_from_orthanc(
        study_instance_uid=study_instance_uid,
        output_dir=output_dir,
        orthanc_url=orthanc_url,
    )

    # Convert to NIfTI
    nifti_path = Path(output_dir) / "ct.nii.gz"
    dicom_to_nifti(ct_dicom_dir, str(nifti_path))

    study_info["nifti_path"] = str(nifti_path)

    return str(nifti_path), study_info
