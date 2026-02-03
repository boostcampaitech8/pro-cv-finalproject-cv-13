"nerve_estimation pipeline: simplified version"

from typing import Dict, Any, Optional, Union
from pathlib import Path

from .pipeline import NerveEstimationPipeline
from .estimators import (
    BaseNerveEstimator,
    EstimationResult,
    VagusEstimator,
    EBSLNEstimator,
    RLNEstimator,
    PhrenicEstimator,
)
from .risk import RiskCalculator, RiskResult
from .mask_loader import MaskLoader
from .export import export_from_json

__version__ = "0.1.0"

__all__ = [
    "run_nerve_estimation",
    "NerveEstimationPipeline",
    "BaseNerveEstimator",
    "EstimationResult",
    "VagusEstimator",
    "EBSLNEstimator",
    "RLNEstimator",
    "PhrenicEstimator",
    "RiskCalculator",
    "RiskResult",
    "MaskLoader",
    "export_from_json",
]


def run_nerve_estimation(
    segmentation_dir: Optional[str] = None,
    normal_structure_dir: Optional[str] = None,
    tumor_path: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """신경 추정 파이프라인 실행."""
    pipeline = NerveEstimationPipeline(
        segmentation_dir=segmentation_dir,
        normal_structure_dir=normal_structure_dir,
        tumor_path=tumor_path,
    )
    return pipeline.run(output_dir=output_dir)
