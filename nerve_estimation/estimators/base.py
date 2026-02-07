"""신경 추정기 기본 클래스."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np

from ..mask_loader import MaskLoader
from ..landmarks import get_mask_z_range
from ..config import VERTEBRAE


@dataclass
class EstimationResult:
    """신경 추정 결과."""
    nerve: str
    side: str
    success: bool
    output_type: str
    pathway_voxels: Optional[np.ndarray] = None
    pathway_mm: Optional[np.ndarray] = None
    center_voxels: Optional[np.ndarray] = None
    center_mm: Optional[np.ndarray] = None
    radius_mm: Optional[float] = None
    uncertainty_mm: float = 0.0
    method: str = ""
    reference: str = ""
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "nerve": self.nerve,
            "side": self.side,
            "type": self.output_type,
            "uncertainty_mm": self.uncertainty_mm,
            "method": self.method,
            "reference": self.reference,
        }

        if self.output_type == "pathway":
            if self.pathway_voxels is not None:
                result["pathway_voxels"] = self.pathway_voxels.tolist()
            if self.pathway_mm is not None:
                result["pathway_mm"] = self.pathway_mm.tolist()
        elif self.output_type == "danger_zone":
            if self.center_voxels is not None:
                result["center_voxels"] = self.center_voxels.tolist()
            if self.center_mm is not None:
                result["center_mm"] = self.center_mm.tolist()
            if self.radius_mm is not None:
                result["radius_mm"] = self.radius_mm

        if self.warnings:
            result["warnings"] = self.warnings
        if self.error:
            result["error"] = self.error

        return result


class BaseNerveEstimator(ABC):
    """신경 추정기 추상 기본 클래스."""

    nerve_name: str = ""
    output_type: str = "pathway"
    uncertainty_mm: float = 5.0
    method: str = ""
    reference: str = ""
    required_structures: tuple = ()

    def __init__(self, mask_loader: MaskLoader):
        self.mask_loader = mask_loader
        self._affine: Optional[np.ndarray] = None

    @property
    def affine(self) -> Optional[np.ndarray]:
        if self._affine is None:
            self._affine = self.mask_loader.get_affine()
        return self._affine

    def get_vertebral_z_range(self) -> Optional[tuple]:
        if hasattr(self, '_vertebral_z_cache'):
            return self._vertebral_z_cache
        z_min, z_max = None, None
        for v in VERTEBRAE:
            mask = self.mask_loader.load_mask(v)
            if mask is None:
                continue
            r = get_mask_z_range(mask)
            if r is None:
                continue
            z_min = r[0] if z_min is None else min(z_min, r[0])
            z_max = r[1] if z_max is None else max(z_max, r[1])
        self._vertebral_z_cache = (z_min, z_max) if z_min is not None else None
        return self._vertebral_z_cache

    def clamp_z_range(self, z_min: int, z_max: int) -> tuple:
        vr = self.get_vertebral_z_range()
        if vr is not None:
            z_min = max(z_min, vr[0])
            z_max = min(z_max, vr[1])
        return (z_min, z_max)

    def check_required_structures(self, side: str) -> List[str]:
        missing = []
        bilateral = ["common_carotid_artery", "internal_carotid_artery", "internal_jugular_vein", "anterior_scalene"]
        for struct in self.required_structures:
            struct_name = f"{struct}_{side}" if struct in bilateral else struct
            if not self.mask_loader.has_structure(struct_name):
                missing.append(struct_name)
        return missing

    @abstractmethod
    def estimate(self, side: str) -> EstimationResult:
        pass

    def _create_error_result(self, side: str, error: str) -> EstimationResult:
        return EstimationResult(
            nerve=self.nerve_name,
            side=side,
            success=False,
            output_type=self.output_type,
            uncertainty_mm=self.uncertainty_mm,
            method=self.method,
            reference=self.reference,
            error=error,
        )

    def _create_success_result(
        self,
        side: str,
        pathway_voxels: Optional[np.ndarray] = None,
        center_voxels: Optional[np.ndarray] = None,
        warnings: Optional[List[str]] = None,
    ) -> EstimationResult:
        from ..utils import voxel_to_mm

        result = EstimationResult(
            nerve=self.nerve_name,
            side=side,
            success=True,
            output_type=self.output_type,
            uncertainty_mm=self.uncertainty_mm,
            method=self.method,
            reference=self.reference,
            warnings=warnings or [],
        )

        affine = self.affine

        if self.output_type == "pathway" and pathway_voxels is not None:
            result.pathway_voxels = pathway_voxels
            if affine is not None:
                result.pathway_mm = voxel_to_mm(pathway_voxels, affine)
        elif self.output_type == "danger_zone" and center_voxels is not None:
            result.center_voxels = center_voxels
            result.radius_mm = self.uncertainty_mm
            if affine is not None:
                result.center_mm = voxel_to_mm(center_voxels, affine)

        return result
