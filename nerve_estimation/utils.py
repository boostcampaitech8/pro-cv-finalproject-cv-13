"""유틸리티 함수."""

import numpy as np
from typing import Dict, Tuple
from scipy.interpolate import splprep, splev


def get_anatomical_directions(affine: np.ndarray) -> Dict[str, Tuple[int, int]]:
    """NIfTI affine 행렬에서 해부학적 방향 추출."""
    rotation = affine[:3, :3]

    dominants = [int(np.argmax(np.abs(rotation[:, axis]))) for axis in range(3)]
    if len(set(dominants)) != 3:
        raise ValueError("Degenerate affine: multiple axes map to same physical direction")

    directions = {}

    for axis in range(3):
        vec = rotation[:, axis]
        abs_vec = np.abs(vec)
        dominant = np.argmax(abs_vec)
        sign = 1 if vec[dominant] > 0 else -1

        if dominant == 0:  # R/L axis
            if sign > 0:
                directions['right'] = (axis, 1)
                directions['left'] = (axis, -1)
            else:
                directions['left'] = (axis, 1)
                directions['right'] = (axis, -1)
        elif dominant == 1:  # A/P axis
            if sign > 0:
                directions['anterior'] = (axis, 1)
                directions['posterior'] = (axis, -1)
            else:
                directions['posterior'] = (axis, 1)
                directions['anterior'] = (axis, -1)
        else:  # S/I axis
            if sign > 0:
                directions['superior'] = (axis, 1)
                directions['inferior'] = (axis, -1)
            else:
                directions['inferior'] = (axis, 1)
                directions['superior'] = (axis, -1)

    return directions


def get_lateral_direction(affine: np.ndarray, side: str) -> Tuple[int, int]:
    """주어진 side에 대한 측면 방향(axis, sign) 반환."""
    directions = get_anatomical_directions(affine)
    if side.lower() not in ['left', 'right']:
        raise ValueError(f"Side must be 'left' or 'right', got: {side}")
    return directions[side.lower()]


def get_anatomical_direction(affine: np.ndarray, direction: str) -> Tuple[int, int]:
    """특정 해부학적 방향에 대한 axis와 sign 반환."""
    directions = get_anatomical_directions(affine)
    direction = direction.lower()
    if direction not in directions:
        raise ValueError(f"Unknown direction: {direction}")
    return directions[direction]


def voxel_to_mm(voxel_coords: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """복셀 좌표를 mm 좌표로 변환."""
    voxel_coords = np.atleast_2d(voxel_coords)
    ones = np.ones((voxel_coords.shape[0], 1))
    voxel_homo = np.hstack([voxel_coords, ones])
    mm_homo = voxel_homo @ affine.T
    mm_coords = mm_homo[:, :3]
    return mm_coords[0] if mm_coords.shape[0] == 1 else mm_coords



def get_spacing(affine: np.ndarray) -> np.ndarray:
    """affine 행렬에서 복셀 간격 추출."""
    rotation = affine[:3, :3]
    return np.sqrt(np.sum(rotation ** 2, axis=0))


def smooth_centerline(points: np.ndarray, smoothing_factor: float = 0.0) -> np.ndarray:
    """B-spline으로 중심선 스무딩."""
    if len(points) < 4:
        return points

    try:
        tck, u = splprep([points[:, 0], points[:, 1], points[:, 2]],
                         s=smoothing_factor, k=min(3, len(points) - 1))
        u_new = np.linspace(0, 1, len(points))
        return np.array(splev(u_new, tck)).T
    except Exception:
        return points


