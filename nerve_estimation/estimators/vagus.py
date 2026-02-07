"""미주신경(Vagus) 추정기."""

import numpy as np
from .base import BaseNerveEstimator, EstimationResult
from ..mask_loader import MaskLoader
from ..landmarks import get_center_at_z, get_mask_z_range
from ..utils import get_anatomical_direction, get_spacing, smooth_centerline
from ..config import NERVE_CONFIG


class VagusEstimator(BaseNerveEstimator):

    nerve_name = "vagus"
    output_type = "pathway"
    method = "midpoint_posterior"
    reference = "Inamura et al. 2017"
    required_structures = ("common_carotid_artery", "internal_jugular_vein")

    def __init__(self, mask_loader: MaskLoader):
        super().__init__(mask_loader)
        config = NERVE_CONFIG["vagus"]
        self.uncertainty_mm = config["uncertainty_mm"]
        self.posterior_offset_mm = config["posterior_offset_mm"]

    def estimate(self, side: str) -> EstimationResult:
        side = side.lower()
        if side not in ["left", "right"]:
            return self._create_error_result(side, f"Invalid side: {side}")

        ijv_mask = self.mask_loader.load_mask(f"internal_jugular_vein_{side}")
        if ijv_mask is None:
            return self._create_error_result(side, f"Failed to load IJV mask for {side}")

        cca_mask = self.mask_loader.load_mask(f"common_carotid_artery_{side}")
        ica_mask = self.mask_loader.load_mask(f"internal_carotid_artery_{side}")

        if cca_mask is None and ica_mask is None:
            return self._create_error_result(side, f"Neither CCA nor ICA mask available for {side}")

        ijv_range = get_mask_z_range(ijv_mask)
        if ijv_range is None:
            return self._create_error_result(side, "Empty IJV mask")

        cca_range = get_mask_z_range(cca_mask) if cca_mask is not None else None
        ica_range = get_mask_z_range(ica_mask) if ica_mask is not None else None

        artery_z_min = min(r[0] for r in [cca_range, ica_range] if r is not None)
        artery_z_max = max(r[1] for r in [cca_range, ica_range] if r is not None)

        z_min = max(ijv_range[0], artery_z_min)
        z_max = min(ijv_range[1], artery_z_max)

        z_min, z_max = self.clamp_z_range(z_min, z_max)

        if z_min > z_max:
            return self._create_error_result(side, "No overlapping Z range between carotid and IJV")

        affine = self.affine
        if affine is None:
            return self._create_error_result(side, "No affine matrix available")

        posterior_axis, posterior_sign = get_anatomical_direction(affine, 'posterior')
        spacing = get_spacing(affine)
        offset_voxels = self.posterior_offset_mm / spacing[posterior_axis]

        pathway_points = []
        warnings = []

        for z in range(z_min, z_max + 1):
            ijv_center = get_center_at_z(ijv_mask, z)
            if ijv_center is None:
                continue

            artery_center = None
            if cca_mask is not None:
                artery_center = get_center_at_z(cca_mask, z)
            if artery_center is None and ica_mask is not None:
                artery_center = get_center_at_z(ica_mask, z)
            if artery_center is None:
                continue

            midpoint = (artery_center + ijv_center) / 2
            nerve_pos = midpoint.copy()
            nerve_pos[posterior_axis] += posterior_sign * offset_voxels
            pathway_points.append(nerve_pos)

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
