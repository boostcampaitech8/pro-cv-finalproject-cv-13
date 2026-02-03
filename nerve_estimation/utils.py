"""유틸리티 함수."""

import numpy as np
from typing import Dict, Tuple
from scipy.interpolate import splprep, splev


def get_anatomical_directions(affine: np.ndarray) -> Dict[str, Tuple[int, int]]:
    """NIfTI affine 행렬에서 해부학적 방향 추출."""
    rotation = affine[:3, :3]
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


def mm_to_voxel(mm_coords: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """mm 좌표를 복셀 좌표로 변환."""
    mm_coords = np.atleast_2d(mm_coords)
    affine_inv = np.linalg.inv(affine)
    ones = np.ones((mm_coords.shape[0], 1))
    mm_homo = np.hstack([mm_coords, ones])
    voxel_homo = mm_homo @ affine_inv.T
    voxel_coords = voxel_homo[:, :3]
    return voxel_coords[0] if voxel_coords.shape[0] == 1 else voxel_coords


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


def resample_centerline(points: np.ndarray, num_points: int) -> np.ndarray:
    """중심선을 특정 포인트 수로 리샘플링."""
    if len(points) < 2 or len(points) == num_points:
        return points

    try:
        tck, u = splprep([points[:, 0], points[:, 1], points[:, 2]],
                         s=0, k=min(3, len(points) - 1))
        u_new = np.linspace(0, 1, num_points)
        return np.array(splev(u_new, tck)).T
    except Exception:
        indices = np.linspace(0, len(points) - 1, num_points)
        resampled = np.zeros((num_points, 3))
        for i, idx in enumerate(indices):
            lower = int(np.floor(idx))
            upper = min(lower + 1, len(points) - 1)
            frac = idx - lower
            resampled[i] = points[lower] * (1 - frac) + points[upper] * frac
        return resampled


def calculate_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """두 점 사이의 유클리드 거리 계산."""
    return float(np.linalg.norm(point1 - point2))


def points_to_mm_distance(voxel_points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """경로를 따른 누적 거리(mm) 계산."""
    if len(voxel_points) < 2:
        return np.array([0.0])

    mm_points = voxel_to_mm(voxel_points, affine)
    segments = np.diff(mm_points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    return np.concatenate([[0], np.cumsum(lengths)])
