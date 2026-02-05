"""Convert nerve estimation results (JSON) to DICOM SEG for OHIF visualization.

This module creates cylindrical masks from nerve path coordinates (pathway type)
and spherical masks from danger zones (danger_zone type), then converts them
to DICOM Segmentation format for display in OHIF Viewer.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import nibabel as nib
from scipy.interpolate import splprep, splev

try:
    from colors import NERVE_COLORS, get_nerve_color
except ImportError:
    from .colors import NERVE_COLORS, get_nerve_color


def create_spline_path(
    points: List[List[float]],
    num_samples: int = 500,
    smoothing: float = 0.0,
) -> np.ndarray:
    """Create smooth spline path from control points.

    Args:
        points: List of [x, y, z] coordinates
        num_samples: Number of points to sample on spline
        smoothing: Spline smoothing factor (0 = interpolate exactly)

    Returns:
        Array of shape (num_samples, 3) with interpolated coordinates
    """
    points = np.array(points)
    if len(points) < 4:
        # Not enough points for spline, just interpolate linearly
        t = np.linspace(0, 1, num_samples)
        path = np.zeros((num_samples, 3))
        for i in range(3):
            path[:, i] = np.interp(t, np.linspace(0, 1, len(points)), points[:, i])
        return path

    try:
        tck, u = splprep([points[:, 0], points[:, 1], points[:, 2]], s=smoothing)
        u_new = np.linspace(0, 1, num_samples)
        x, y, z = splev(u_new, tck)
        return np.column_stack([x, y, z])
    except Exception:
        # Fallback to linear interpolation
        t = np.linspace(0, 1, num_samples)
        path = np.zeros((num_samples, 3))
        for i in range(3):
            path[:, i] = np.interp(t, np.linspace(0, 1, len(points)), points[:, i])
        return path


def create_cylindrical_mask(
    shape: Tuple[int, int, int],
    affine: np.ndarray,
    path_points: np.ndarray,
    radius_mm: float,
) -> np.ndarray:
    """Create cylindrical mask around a path.

    Args:
        shape: Volume shape (x, y, z)
        affine: NIfTI affine transformation matrix
        path_points: World coordinates of path centerline (N, 3)
        radius_mm: Cylinder radius in mm

    Returns:
        Binary mask array of given shape
    """
    mask = np.zeros(shape, dtype=np.uint8)

    inv_affine = np.linalg.inv(affine)
    spacing = np.abs(np.diag(affine)[:3])
    ones = np.ones((len(path_points), 1))
    path_h = np.hstack([path_points, ones])
    path_voxels = (inv_affine @ path_h.T).T[:, :3]

    # Create distance field from path
    # For efficiency, iterate over a bounding box around the path
    padding = int(np.ceil(radius_mm / np.min(spacing))) + 2

    x_min = max(0, int(np.floor(path_voxels[:, 0].min())) - padding)
    x_max = min(shape[0], int(np.ceil(path_voxels[:, 0].max())) + padding)
    y_min = max(0, int(np.floor(path_voxels[:, 1].min())) - padding)
    y_max = min(shape[1], int(np.ceil(path_voxels[:, 1].max())) + padding)
    z_min = max(0, int(np.floor(path_voxels[:, 2].min())) - padding)
    z_max = min(shape[2], int(np.ceil(path_voxels[:, 2].max())) + padding)

    x = np.arange(x_min, x_max)
    y = np.arange(y_min, y_max)
    z = np.arange(z_min, z_max)

    if len(x) == 0 or len(y) == 0 or len(z) == 0:
        return mask

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    voxel_coords = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
    ones_v = np.ones((len(voxel_coords), 1))
    voxel_h = np.hstack([voxel_coords, ones_v])
    world_coords = (affine @ voxel_h.T).T[:, :3]

    # Calculate minimum distance to path for each voxel
    # Use vectorized distance calculation for efficiency
    min_distances = np.full(len(world_coords), np.inf)

    # Process path in chunks to manage memory
    chunk_size = 50
    for i in range(0, len(path_points), chunk_size):
        chunk = path_points[i:i+chunk_size]
        # Distance from each voxel to each point in chunk
        # Shape: (num_voxels, chunk_size)
        diff = world_coords[:, np.newaxis, :] - chunk[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))
        chunk_min = np.min(distances, axis=1)
        min_distances = np.minimum(min_distances, chunk_min)

    inside = min_distances <= radius_mm
    voxel_indices = voxel_coords[inside].astype(int)

    if len(voxel_indices) > 0:
        mask[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1

    return mask


def create_spherical_mask(
    shape: Tuple[int, int, int],
    affine: np.ndarray,
    center: List[float],
    radius_mm: float,
) -> np.ndarray:
    """Create spherical mask at a given center point.

    Args:
        shape: Volume shape (x, y, z)
        affine: NIfTI affine transformation matrix
        center: World coordinates of sphere center [x, y, z]
        radius_mm: Sphere radius in mm

    Returns:
        Binary mask array of given shape
    """
    mask = np.zeros(shape, dtype=np.uint8)

    inv_affine = np.linalg.inv(affine)
    spacing = np.abs(np.diag(affine)[:3])
    center = np.array(center)
    center_h = np.append(center, 1)
    center_voxel = (inv_affine @ center_h)[:3]

    padding = int(np.ceil(radius_mm / np.min(spacing))) + 2

    x_min = max(0, int(np.floor(center_voxel[0])) - padding)
    x_max = min(shape[0], int(np.ceil(center_voxel[0])) + padding)
    y_min = max(0, int(np.floor(center_voxel[1])) - padding)
    y_max = min(shape[1], int(np.ceil(center_voxel[1])) + padding)
    z_min = max(0, int(np.floor(center_voxel[2])) - padding)
    z_max = min(shape[2], int(np.ceil(center_voxel[2])) + padding)

    x = np.arange(x_min, x_max)
    y = np.arange(y_min, y_max)
    z = np.arange(z_min, z_max)

    if len(x) == 0 or len(y) == 0 or len(z) == 0:
        return mask

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    voxel_coords = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
    ones_v = np.ones((len(voxel_coords), 1))
    voxel_h = np.hstack([voxel_coords, ones_v])
    world_coords = (affine @ voxel_h.T).T[:, :3]

    distances = np.sqrt(np.sum((world_coords - center) ** 2, axis=1))
    inside = distances <= radius_mm
    voxel_indices = voxel_coords[inside].astype(int)

    if len(voxel_indices) > 0:
        mask[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1

    return mask


def nerve_json_to_nifti_masks(
    nerve_results_path: str,
    reference_nifti_path: str,
    output_dir: str,
) -> Dict[str, str]:
    """Convert nerve estimation JSON results to NIfTI masks.

    Handles both pathway type (cylindrical masks) and danger_zone type
    (spherical masks).

    Args:
        nerve_results_path: Path to nerve_results.json
        reference_nifti_path: Reference NIfTI for geometry
        output_dir: Output directory for masks

    Returns:
        Dictionary mapping nerve names to output file paths
    """
    with open(nerve_results_path) as f:
        results = json.load(f)

    ref_nii = nib.load(reference_nifti_path)
    shape = ref_nii.shape
    affine = ref_nii.affine

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = {}

    # Process pathways (nerve paths)
    nerves = results.get("nerves", [])
    if not nerves and "estimated_paths" in results:
        nerves = results["estimated_paths"]

    for nerve_data in nerves:
        # Support both formats: "name" or "nerve"+"side"
        nerve_name = nerve_data.get("name")
        if not nerve_name:
            nerve = nerve_data.get("nerve", "unknown")
            side = nerve_data.get("side", "")
            nerve_name = f"{nerve}_{side}" if side else nerve

        nerve_type = nerve_data.get("type", "pathway")

        if nerve_type == "danger_zone":
            # Handle danger zone (spherical mask)
            # Support both: "center" (world), "center_voxels" (voxel), "position" (world)
            is_voxel_coords = False
            center = nerve_data.get("center") or nerve_data.get("position")
            if center is None:
                center = nerve_data.get("center_voxels")
                is_voxel_coords = True  # center_voxels are in voxel coordinates

            radius = nerve_data.get("radius_mm", nerve_data.get("radius", 5.0))

            if center is None:
                print(f"[nerve_to_dicom] Skipping {nerve_name}: no center found")
                continue

            center = np.array(center)

            # Convert voxel coordinates to world coordinates if needed
            if is_voxel_coords:
                center_h = np.append(center, 1)
                center = (affine @ center_h)[:3]
                print(f"[nerve_to_dicom] Processing danger zone {nerve_name}: center converted from voxel to world")
            else:
                print(f"[nerve_to_dicom] Processing danger zone {nerve_name}: center in world coords")

            # Create spherical mask
            zone_mask = create_spherical_mask(shape, affine, center, radius)

            zone_filename = f"{nerve_name}.nii.gz"
            zone_path = output_dir / zone_filename
            zone_nii = nib.Nifti1Image(zone_mask, affine)
            nib.save(zone_nii, str(zone_path))
            output_files[nerve_name] = str(zone_path)
            print(f"[nerve_to_dicom] Saved danger zone: {zone_path}")

        else:
            # Handle pathway (cylindrical mask) - default behavior
            # Support both: "path" (world), "points" (world), "pathway_voxels" (voxel)
            is_voxel_coords = False
            path_points = nerve_data.get("path") or nerve_data.get("points")
            if not path_points:
                path_points = nerve_data.get("pathway_voxels", [])
                is_voxel_coords = True  # pathway_voxels are in voxel coordinates

            if not path_points:
                print(f"[nerve_to_dicom] Skipping {nerve_name}: no path points found")
                continue

            path_points = np.array(path_points)

            # Convert voxel coordinates to world coordinates if needed
            if is_voxel_coords:
                ones = np.ones((len(path_points), 1))
                path_h = np.hstack([path_points, ones])
                path_points = (affine @ path_h.T).T[:, :3]
                print(f"[nerve_to_dicom] Processing {nerve_name}: {len(path_points)} points (converted from voxel to world)")
            else:
                print(f"[nerve_to_dicom] Processing {nerve_name}: {len(path_points)} points (world coords)")

            spline_path = create_spline_path(path_points, num_samples=500)

            # Create uncertainty zone mask (full cylinder covering the nerve's possible location)
            uncertainty_mm = nerve_data.get(
                "uncertainty_mm",
                nerve_data.get("uncertainty", 5.0)
            )
            if isinstance(uncertainty_mm, (list, tuple)):
                uncertainty_mm = max(uncertainty_mm)

            uncertainty_radius = uncertainty_mm
            uncertainty_mask = create_cylindrical_mask(
                shape, affine, spline_path, uncertainty_radius
            )

            uncertainty_filename = f"{nerve_name}_uncertainty.nii.gz"
            uncertainty_path = output_dir / uncertainty_filename
            uncertainty_nii = nib.Nifti1Image(uncertainty_mask, affine)
            nib.save(uncertainty_nii, str(uncertainty_path))
            output_files[f"{nerve_name}_uncertainty"] = str(uncertainty_path)
            print(f"[nerve_to_dicom] Saved uncertainty mask: {uncertainty_path} (radius={uncertainty_radius}mm)")

    # Process explicit danger zones
    danger_zones = results.get("danger_zones", [])
    for zone_data in danger_zones:
        zone_name = zone_data.get("name", "danger_zone")
        center = zone_data.get("center", zone_data.get("position"))
        radius = zone_data.get("radius_mm", zone_data.get("radius", 5.0))

        if center is None:
            continue

        zone_mask = create_spherical_mask(shape, affine, center, radius)

        zone_filename = f"{zone_name}.nii.gz"
        zone_path = output_dir / zone_filename
        zone_nii = nib.Nifti1Image(zone_mask, affine)
        nib.save(zone_nii, str(zone_path))
        output_files[zone_name] = str(zone_path)

    return output_files


