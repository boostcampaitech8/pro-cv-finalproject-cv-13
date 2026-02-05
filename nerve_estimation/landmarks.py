"""해부학적 랜드마크 추출."""

import numpy as np
from typing import Optional, Tuple
from .utils import get_anatomical_direction, get_lateral_direction


def get_coords_at_z(mask: np.ndarray, z_level: int) -> Optional[np.ndarray]:
    """특정 Z 레벨에서 마스크의 복셀 좌표 반환."""
    if mask is None or z_level < 0 or z_level >= mask.shape[2]:
        return None

    slice_2d = mask[:, :, z_level]
    coords_2d = np.argwhere(slice_2d > 0)

    if len(coords_2d) == 0:
        return None

    return np.column_stack([
        coords_2d[:, 0],
        coords_2d[:, 1],
        np.full(len(coords_2d), z_level)
    ])


def get_mask_z_range(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    """마스크의 Z축 범위 반환."""
    if mask is None:
        return None

    z_indices = np.where(np.any(mask > 0, axis=(0, 1)))[0]
    if len(z_indices) == 0:
        return None

    return (int(z_indices[0]), int(z_indices[-1]))



def get_center_at_z(mask: np.ndarray, z_level: int) -> Optional[np.ndarray]:
    """특정 Z 레벨에서 마스크의 중심점 반환."""
    coords = get_coords_at_z(mask, z_level)
    if coords is None or len(coords) == 0:
        return None
    return np.mean(coords, axis=0)


def get_superior_pole(
    mask: np.ndarray,
    affine: np.ndarray,
    side: Optional[str] = None,
) -> Optional[np.ndarray]:
    """구조물의 상극(최상단) 좌표 반환."""
    if mask is None:
        return None

    axis, sign = get_anatomical_direction(affine, 'superior')
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None

    if side is not None:
        lateral_axis, lateral_sign = get_lateral_direction(affine, side)
        centroid = np.mean(coords, axis=0)
        if lateral_sign > 0:
            side_coords = coords[coords[:, lateral_axis] >= centroid[lateral_axis]]
        else:
            side_coords = coords[coords[:, lateral_axis] <= centroid[lateral_axis]]
        if len(side_coords) > 0:
            coords = side_coords

    if sign > 0:
        superior_idx = np.argmax(coords[:, axis])
    else:
        superior_idx = np.argmin(coords[:, axis])

    superior_z = coords[superior_idx, axis]
    tolerance = 2

    if sign > 0:
        top_coords = coords[coords[:, axis] >= superior_z - tolerance]
    else:
        top_coords = coords[coords[:, axis] <= superior_z + tolerance]

    return np.mean(top_coords, axis=0)



def get_lateral_border_at_z(
    mask: np.ndarray,
    z_level: int,
    affine: np.ndarray,
    side: str,
) -> Optional[np.ndarray]:
    """특정 Z 레벨에서 측면 경계점 반환."""
    coords = get_coords_at_z(mask, z_level)
    if coords is None or len(coords) == 0:
        return None

    axis, sign = get_lateral_direction(affine, side)
    idx = np.argmax(coords[:, axis]) if sign > 0 else np.argmin(coords[:, axis])
    return coords[idx].astype(float)


def get_anterior_surface_at_z(
    mask: np.ndarray,
    z_level: int,
    affine: np.ndarray,
    percentile: float = 10.0,
) -> Optional[np.ndarray]:
    """특정 Z 레벨에서 전방 표면 중심 반환."""
    coords = get_coords_at_z(mask, z_level)
    if coords is None or len(coords) == 0:
        return None

    axis, sign = get_anatomical_direction(affine, 'anterior')

    if sign > 0:
        threshold = np.percentile(coords[:, axis], 100 - percentile)
        anterior_coords = coords[coords[:, axis] >= threshold]
    else:
        threshold = np.percentile(coords[:, axis], percentile)
        anterior_coords = coords[coords[:, axis] <= threshold]

    return np.mean(anterior_coords, axis=0) if len(anterior_coords) > 0 else None
