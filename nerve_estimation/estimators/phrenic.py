"""횡격신경(Phrenic) 추정기."""

import numpy as np
from typing import Optional

from .base import BaseNerveEstimator, EstimationResult
from ..mask_loader import MaskLoader
from ..landmarks import get_mask_z_range, get_anterior_surface_at_z
from ..utils import smooth_centerline
from ..config import NERVE_CONFIG


class PhrenicEstimator(BaseNerveEstimator):
    """전사각근 전방 표면 방법으로 횡격신경 위치 추정."""

    nerve_name = "phrenic"
    output_type = "pathway"
    method = "anterior_surface"
    reference = "Baseline implementation"
    required_structures = ["anterior_scalene"]

    def __init__(self, mask_loader: MaskLoader):
        super().__init__(mask_loader)
        config = NERVE_CONFIG["phrenic"]
        self.uncertainty_mm = config["uncertainty_mm"]
        self.anterior_percentile = 10.0

    def _estimate_at_z(self, scalene_mask: np.ndarray, z_level: int) -> Optional[np.ndarray]:
        affine = self.affine
        if affine is None:
            return None
        return get_anterior_surface_at_z(scalene_mask, z_level, affine, percentile=self.anterior_percentile)

    def estimate(self, side: str) -> EstimationResult:
        side = side.lower()
        if side not in ["left", "right"]:
            return self._create_error_result(side, f"Invalid side: {side}")

        missing = self.check_required_structures(side)
        if missing:
            return self._create_error_result(side, f"Missing required structures: {missing}")

        scalene_mask = self.mask_loader.load_mask(f"anterior_scalene_{side}")
        if scalene_mask is None:
            return self._create_error_result(side, f"Failed to load anterior scalene mask for {side}")

        z_range = get_mask_z_range(scalene_mask)
        if z_range is None:
            return self._create_error_result(side, "Empty scalene mask")

        pathway_points = []
        warnings = []

        for z in range(z_range[0], z_range[1] + 1):
            position = self._estimate_at_z(scalene_mask, z)
            if position is not None:
                pathway_points.append(position)

        if len(pathway_points) == 0:
            return self._create_error_result(side, "No valid nerve positions found")

        pathway_voxels = np.array(pathway_points)

        if len(pathway_voxels) >= 4:
            pathway_voxels = smooth_centerline(pathway_voxels, smoothing_factor=0.5)

        if len(pathway_voxels) < 3:
            warnings.append("Short pathway (< 3 points)")

        return self._create_success_result(
            side=side,
            pathway_voxels=pathway_voxels,
            warnings=warnings if warnings else None,
        )
