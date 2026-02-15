"""Orthanc REST API client for DICOM server operations."""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional

import requests


class OrthancClient:
    """Client for Orthanc DICOM server REST API.

    Provides methods for uploading DICOM files, querying studies,
    and managing data on the Orthanc server.
    """

    def __init__(
        self,
        url: str = None,
        username: str = None,
        password: str = None,
        timeout: int = 30,
    ):
        """Initialize Orthanc client.

        Args:
            url: Orthanc server URL (default: from ORTHANC_URL env or http://orthanc:8042)
            username: Optional username for authentication
            password: Optional password for authentication
            timeout: Request timeout in seconds
        """
        self.url = url or os.environ.get("ORTHANC_URL", "http://orthanc:8042")
        self.url = self.url.rstrip("/")
        self.timeout = timeout

        self.auth = None
        if username and password:
            self.auth = (username, password)

    def check_health(self) -> bool:
        """Check if Orthanc server is healthy and reachable.

        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = requests.get(
                f"{self.url}/system",
                auth=self.auth,
                timeout=self.timeout,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def get_system_info(self) -> Optional[Dict[str, Any]]:
        """Get Orthanc system information.

        Returns:
            System info dictionary or None if unavailable
        """
        try:
            response = requests.get(
                f"{self.url}/system",
                auth=self.auth,
                timeout=self.timeout,
            )
            if response.status_code == 200:
                return response.json()
            return None
        except requests.RequestException:
            return None

    def upload_file(self, dicom_path: str) -> Dict[str, Any]:
        """Upload a single DICOM file to Orthanc.

        Args:
            dicom_path: Path to DICOM file

        Returns:
            Dictionary with upload result containing:
            - success: bool
            - orthanc_id: Orthanc instance ID (if successful)
            - study_id: Parent study ID (if successful)
            - series_id: Parent series ID (if successful)
            - error: Error message (if failed)
        """
        try:
            with open(dicom_path, "rb") as f:
                data = f.read()

            response = requests.post(
                f"{self.url}/instances",
                data=data,
                headers={"Content-Type": "application/dicom"},
                auth=self.auth,
                timeout=self.timeout,
            )

            if response.status_code in (200, 201):
                result = response.json()
                return {
                    "success": True,
                    "orthanc_id": result.get("ID"),
                    "study_id": result.get("ParentStudy"),
                    "series_id": result.get("ParentSeries"),
                    "status": result.get("Status"),
                }
            else:
                return {
                    "success": False,
                    "status_code": response.status_code,
                    "error": response.text,
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def upload_bytes(self, data: bytes) -> Dict[str, Any]:
        """Upload DICOM data as bytes to Orthanc.

        Args:
            data: DICOM file content as bytes

        Returns:
            Dictionary with upload result
        """
        try:
            response = requests.post(
                f"{self.url}/instances",
                data=data,
                headers={"Content-Type": "application/dicom"},
                auth=self.auth,
                timeout=self.timeout,
            )

            if response.status_code in (200, 201):
                result = response.json()
                return {
                    "success": True,
                    "orthanc_id": result.get("ID"),
                    "study_id": result.get("ParentStudy"),
                    "series_id": result.get("ParentSeries"),
                }
            else:
                return {
                    "success": False,
                    "status_code": response.status_code,
                    "error": response.text,
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def upload_directory(self, dir_path: str) -> Dict[str, Any]:
        """Upload all DICOM files in a directory to Orthanc.

        Args:
            dir_path: Directory containing DICOM files

        Returns:
            Dictionary with:
            - uploaded: List of successfully uploaded files
            - failed: List of failed uploads with errors
            - study_ids: Set of unique study IDs
            - series_ids: Set of unique series IDs
        """
        results = {
            "uploaded": [],
            "failed": [],
            "study_ids": set(),
            "series_ids": set(),
        }

        dicom_files = list(Path(dir_path).rglob("*.dcm"))

        for filepath in dicom_files:
            result = self.upload_file(str(filepath))

            if result.get("success"):
                results["uploaded"].append(str(filepath))
                if result.get("study_id"):
                    results["study_ids"].add(result["study_id"])
                if result.get("series_id"):
                    results["series_ids"].add(result["series_id"])
            else:
                results["failed"].append({
                    "file": str(filepath),
                    "error": result.get("error", "Unknown error"),
                })

        results["study_ids"] = list(results["study_ids"])
        results["series_ids"] = list(results["series_ids"])

        return results

    def get_studies(self) -> List[str]:
        """Get list of all study IDs in Orthanc.

        Returns:
            List of study IDs
        """
        try:
            response = requests.get(
                f"{self.url}/studies",
                auth=self.auth,
                timeout=self.timeout,
            )
            if response.status_code == 200:
                return response.json()
            return []
        except requests.RequestException:
            return []

    def get_study(self, study_id: str) -> Optional[Dict[str, Any]]:
        """Get study details by Orthanc study ID.

        Args:
            study_id: Orthanc study ID

        Returns:
            Study details dictionary or None if not found
        """
        try:
            response = requests.get(
                f"{self.url}/studies/{study_id}",
                auth=self.auth,
                timeout=self.timeout,
            )
            if response.status_code == 200:
                return response.json()
            return None
        except requests.RequestException:
            return None

    def get_study_by_uid(self, study_instance_uid: str) -> Optional[Dict[str, Any]]:
        """Get study details by DICOM StudyInstanceUID.

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
                    return self.get_study(results[0])
            return None
        except requests.RequestException:
            return None

    def get_studies_with_details(self) -> List[Dict[str, Any]]:
        """Get all studies with their details.

        Returns:
            List of study detail dictionaries
        """
        study_ids = self.get_studies()
        studies = []

        for study_id in study_ids:
            study = self.get_study(study_id)
            if study:
                studies.append(study)

        return studies

    def delete_study(self, study_id: str) -> bool:
        """Delete a study from Orthanc.

        Args:
            study_id: Orthanc study ID

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            response = requests.delete(
                f"{self.url}/studies/{study_id}",
                auth=self.auth,
                timeout=self.timeout,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def delete_series(self, series_id: str) -> bool:
        """Delete a series from Orthanc.

        Args:
            series_id: Orthanc series ID

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            response = requests.delete(
                f"{self.url}/series/{series_id}",
                auth=self.auth,
                timeout=self.timeout,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def get_dicomweb_url(self, study_instance_uid: str) -> str:
        """Generate DICOMweb WADO-RS URL for a study.

        Args:
            study_instance_uid: DICOM StudyInstanceUID

        Returns:
            DICOMweb URL for the study
        """
        return f"{self.url}/dicom-web/studies/{study_instance_uid}"

    def get_ohif_url(
        self,
        study_instance_uid: str,
        ohif_base_url: str = "http://localhost:3000",
    ) -> str:
        """Generate OHIF Viewer URL for a study.

        Args:
            study_instance_uid: DICOM StudyInstanceUID
            ohif_base_url: OHIF Viewer base URL

        Returns:
            OHIF Viewer URL for the study
        """
        return f"{ohif_base_url}/nerve-assessment?StudyInstanceUIDs={study_instance_uid}"

    def set_study_metadata(
        self,
        study_id: str,
        key: str,
        value: str,
    ) -> bool:
        """Set metadata on a study.

        Args:
            study_id: Orthanc study ID
            key: Metadata key name
            value: Metadata value

        Returns:
            True if successful, False otherwise
        """
        try:
            response = requests.put(
                f"{self.url}/studies/{study_id}/metadata/{key}",
                data=value,
                auth=self.auth,
                timeout=self.timeout,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def get_study_metadata(
        self,
        study_id: str,
        key: str,
    ) -> Optional[str]:
        """Get metadata from a study.

        Args:
            study_id: Orthanc study ID
            key: Metadata key name

        Returns:
            Metadata value or None if not found
        """
        try:
            response = requests.get(
                f"{self.url}/studies/{study_id}/metadata/{key}",
                auth=self.auth,
                timeout=self.timeout,
            )
            if response.status_code == 200:
                return response.text
            return None
        except requests.RequestException:
            return None

    def get_orthanc_id_by_study_uid(self, study_instance_uid: str) -> Optional[str]:
        """Get Orthanc internal ID from StudyInstanceUID.

        Args:
            study_instance_uid: DICOM StudyInstanceUID

        Returns:
            Orthanc study ID or None if not found
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
                    return results[0]
            return None
        except requests.RequestException:
            return None


def upload_to_orthanc(
    file_paths: List[str],
    orthanc_url: str = "http://localhost:8042",
) -> Dict[str, Any]:
    """Upload DICOM files to Orthanc server.

    Args:
        file_paths: List of DICOM file paths to upload
        orthanc_url: Orthanc server URL

    Returns:
        Dictionary with upload results
    """
    client = OrthancClient(url=orthanc_url)
    results = {
        "uploaded": [],
        "failed": [],
        "study_ids": set(),
        "series_ids": set(),
    }

    for filepath in file_paths:
        result = client.upload_file(filepath)

        if result.get("success"):
            results["uploaded"].append(filepath)
            if result.get("study_id"):
                results["study_ids"].add(result["study_id"])
            if result.get("series_id"):
                results["series_ids"].add(result["series_id"])
        else:
            results["failed"].append({
                "file": filepath,
                "error": result.get("error", "Unknown error"),
            })

    results["study_ids"] = list(results["study_ids"])
    results["series_ids"] = list(results["series_ids"])

    return results


def upload_directory_to_orthanc(
    directory: str,
    orthanc_url: str = "http://localhost:8042",
) -> Dict[str, Any]:
    """Upload all DICOM files in a directory to Orthanc.

    Args:
        directory: Directory containing DICOM files
        orthanc_url: Orthanc server URL

    Returns:
        Upload results
    """
    client = OrthancClient(url=orthanc_url)
    return client.upload_directory(directory)
