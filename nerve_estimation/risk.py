"""신경-종양 근접도 위험 계산."""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from scipy.ndimage import distance_transform_edt

from .estimators.base import EstimationResult
from .mask_loader import MaskLoader
from .utils import get_spacing
from .config import RISK_THRESHOLDS


@dataclass
class RiskResult:
    """신경 위험도 평가 결과."""
    nerve: str
    side: str
    min_distance_mm: float
    uncertainty_mm: float
    effective_distance_mm: float
    risk_level: str
    overlap: bool
    overlap_ratio: float
    warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "nerve": self.nerve,
            "side": self.side,
            "min_distance_mm": round(self.min_distance_mm, 2),
            "uncertainty_mm": self.uncertainty_mm,
            "effective_distance_mm": round(self.effective_distance_mm, 2),
            "risk_level": self.risk_level,
            "overlap": self.overlap,
            "overlap_ratio": round(self.overlap_ratio, 4),
        }
        if self.warning:
            result["warning"] = self.warning
        return result


class RiskCalculator:
    """신경-종양 위험도 계산기."""

    def __init__(
        self,
        mask_loader: MaskLoader,
        tumor_mask: Optional[np.ndarray] = None,
    ):
        self.mask_loader = mask_loader
        self._tumor_mask = tumor_mask
        self._tumor_distance_map = None
        self._affine = None

    @property
    def tumor_mask(self) -> Optional[np.ndarray]:
        if self._tumor_mask is None:
            self._tumor_mask = self.mask_loader.load_mask("tumor")
        return self._tumor_mask

    @property
    def affine(self) -> Optional[np.ndarray]:
        if self._affine is None:
            self._affine = self.mask_loader.get_affine()
        return self._affine

    def _compute_distance_map(self) -> Optional[np.ndarray]:
        if self._tumor_distance_map is not None:
            return self._tumor_distance_map

        tumor = self.tumor_mask
        if tumor is None or np.sum(tumor) == 0:
            return None

        affine = self.affine
        if affine is None:
            return None

        spacing = get_spacing(affine)
        outside_tumor = tumor == 0
        self._tumor_distance_map = distance_transform_edt(outside_tumor, sampling=spacing)
        return self._tumor_distance_map

    def _check_overlap(self, nerve_points: np.ndarray) -> tuple:
        tumor = self.tumor_mask
        if tumor is None:
            return False, 0.0

        overlap_count = 0
        for point in nerve_points:
            voxel = np.round(point).astype(int)
            if (0 <= voxel[0] < tumor.shape[0] and
                0 <= voxel[1] < tumor.shape[1] and
                0 <= voxel[2] < tumor.shape[2]):
                if tumor[voxel[0], voxel[1], voxel[2]] > 0:
                    overlap_count += 1

        overlap_ratio = overlap_count / len(nerve_points) if len(nerve_points) > 0 else 0.0
        return overlap_count > 0, overlap_ratio

    def _get_min_distance(self, nerve_points: np.ndarray) -> float:
        distance_map = self._compute_distance_map()
        if distance_map is None:
            return float('inf')

        min_dist = float('inf')
        for point in nerve_points:
            voxel = np.round(point).astype(int)
            if (0 <= voxel[0] < distance_map.shape[0] and
                0 <= voxel[1] < distance_map.shape[1] and
                0 <= voxel[2] < distance_map.shape[2]):
                min_dist = min(min_dist, distance_map[voxel[0], voxel[1], voxel[2]])

        return min_dist

    def _determine_risk_level(self, has_overlap: bool, effective_distance: float) -> str:
        if has_overlap or effective_distance < RISK_THRESHOLDS["high"]:
            return "HIGH"
        elif effective_distance < RISK_THRESHOLDS["moderate"]:
            return "MODERATE"
        return "LOW"

    def calculate_risk(self, estimation_result: EstimationResult) -> Optional[RiskResult]:
        if not estimation_result.success or self.tumor_mask is None:
            return None

        if estimation_result.output_type == "pathway":
            if estimation_result.pathway_voxels is None:
                return None
            nerve_points = estimation_result.pathway_voxels
        elif estimation_result.output_type == "danger_zone":
            if estimation_result.center_voxels is None:
                return None
            nerve_points = estimation_result.center_voxels.reshape(1, 3)
        else:
            return None

        has_overlap, overlap_ratio = self._check_overlap(nerve_points)
        min_distance = self._get_min_distance(nerve_points)
        uncertainty = estimation_result.uncertainty_mm
        effective_distance = max(0.0, min_distance - uncertainty)
        risk_level = self._determine_risk_level(has_overlap, effective_distance)

        warning = None
        if has_overlap:
            warning = "Nerve overlaps with tumor"
        elif effective_distance < RISK_THRESHOLDS["high"]:
            warning = "Nerve very close to tumor"

        return RiskResult(
            nerve=estimation_result.nerve,
            side=estimation_result.side,
            min_distance_mm=min_distance,
            uncertainty_mm=uncertainty,
            effective_distance_mm=effective_distance,
            risk_level=risk_level,
            overlap=has_overlap,
            overlap_ratio=overlap_ratio,
            warning=warning,
        )

    def calculate_all_risks(self, estimation_results: List[EstimationResult]) -> List[RiskResult]:
        if self.tumor_mask is None:
            return []

        risk_results = []
        for result in estimation_results:
            risk = self.calculate_risk(result)
            if risk is not None:
                risk_results.append(risk)

        return risk_results
