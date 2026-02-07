"""상후두신경 외분지(EBSLN) 추정기."""

import numpy as np

from .base import BaseNerveEstimator, EstimationResult
from ..mask_loader import MaskLoader
from ..landmarks import get_superior_pole, get_mask_z_range, get_center_at_z
from ..utils import get_anatomical_direction, get_spacing
from ..config import NERVE_CONFIG


class EBSLNEstimator(BaseNerveEstimator):
    """갑상선 상극 방법으로 EBSLN 위험 영역 추정."""

    nerve_name = "ebsln"
    output_type = "danger_zone"
    method = "superior_pole_offset"
    reference = "Estrela et al. 2011 (PMID:21537628)"
    required_structures = ("thyroid_gland",)

    def __init__(self, mask_loader: MaskLoader):
        super().__init__(mask_loader)
        config = NERVE_CONFIG["ebsln"]
        self.uncertainty_mm = config["uncertainty_mm"]
        self.superior_offset_mm = config["superior_offset_mm"]

    def estimate(self, side: str) -> EstimationResult:
        side = side.lower()
        if side not in ["left", "right"]:
            return self._create_error_result(side, f"Invalid side: {side}")

        missing = self.check_required_structures(side)
        if missing:
            return self._create_error_result(side, f"Missing required structures: {missing}")

        thyroid_mask = self.mask_loader.load_mask("thyroid_gland")
        if thyroid_mask is None:
            return self._create_error_result(side, "Failed to load thyroid mask")

        affine = self.affine
        if affine is None:
            return self._create_error_result(side, "No affine matrix available")

        trachea_mask = self.mask_loader.load_mask("trachea")
        trachea_midline = None
        if trachea_mask is not None:
            thyroid_z_range = get_mask_z_range(thyroid_mask)
            if thyroid_z_range is not None:
                mid_z = (thyroid_z_range[0] + thyroid_z_range[1]) // 2
                trachea_midline = get_center_at_z(trachea_mask, mid_z)

        superior_pole = get_superior_pole(thyroid_mask, affine, side=side, midline_point=trachea_midline)
        if superior_pole is None:
            return self._create_error_result(side, f"Could not find thyroid superior pole for {side} side")

        superior_axis, superior_sign = get_anatomical_direction(affine, 'superior')
        spacing = get_spacing(affine)
        offset_voxels = self.superior_offset_mm / spacing[superior_axis]

        danger_center = superior_pole.copy()
        danger_center[superior_axis] += superior_sign * offset_voxels

        return self._create_success_result(side=side, center_voxels=danger_center)
