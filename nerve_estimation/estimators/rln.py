"""되돌이후두신경(RLN) 추정기."""

import numpy as np
from typing import Optional

from .base import BaseNerveEstimator, EstimationResult
from ..mask_loader import MaskLoader
from ..landmarks import get_mask_z_range, get_lateral_border_at_z
from ..utils import smooth_centerline
from ..config import NERVE_CONFIG


class RLNEstimator(BaseNerveEstimator):
    """기관-식도 중간점 방법으로 RLN 위치 추정."""

    nerve_name = "rln"
    output_type = "pathway"
    method = "teg_midpoint"
    reference = "Baseline implementation"
    required_structures = ("trachea", "esophagus")

    def __init__(self, mask_loader: MaskLoader):
        super().__init__(mask_loader)
        config = NERVE_CONFIG["rln"]
        self.uncertainty_mm = config["uncertainty_mm"]

    def _estimate_at_z(
        self,
        trachea_mask: np.ndarray,
        esophagus_mask: np.ndarray,
        z_level: int,
        side: str,
    ) -> Optional[np.ndarray]:
        affine = self.affine
        if affine is None:
            return None

        trachea_border = get_lateral_border_at_z(trachea_mask, z_level, affine, side)
        esophagus_border = get_lateral_border_at_z(esophagus_mask, z_level, affine, side)

        if trachea_border is None or esophagus_border is None:
            return None

        return (trachea_border + esophagus_border) / 2

    def estimate(self, side: str) -> EstimationResult:
        side = side.lower()
        if side not in ["left", "right"]:
            return self._create_error_result(side, f"Invalid side: {side}")

        missing = self.check_required_structures(side)
        if missing:
            return self._create_error_result(side, f"Missing required structures: {missing}")

        trachea_mask = self.mask_loader.load_mask("trachea")
        esophagus_mask = self.mask_loader.load_mask("esophagus")

        if trachea_mask is None:
            return self._create_error_result(side, "Failed to load trachea mask")
        if esophagus_mask is None:
            return self._create_error_result(side, "Failed to load esophagus mask")

        trachea_range = get_mask_z_range(trachea_mask)
        esophagus_range = get_mask_z_range(esophagus_mask)

        if trachea_range is None or esophagus_range is None:
            return self._create_error_result(side, "Empty mask(s)")

        z_min = max(trachea_range[0], esophagus_range[0])
        z_max = min(trachea_range[1], esophagus_range[1])

        if z_min > z_max:
            return self._create_error_result(side, "No overlapping Z range between trachea and esophagus")

        pathway_points = []
        warnings = []

        for z in range(z_min, z_max + 1):
            position = self._estimate_at_z(trachea_mask, esophagus_mask, z, side)
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
