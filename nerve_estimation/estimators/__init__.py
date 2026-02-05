"""신경 추정기 클래스"""

from .base import BaseNerveEstimator, EstimationResult
from .vagus import VagusEstimator
from .ebsln import EBSLNEstimator
from .rln import RLNEstimator
from .phrenic import PhrenicEstimator

__all__ = [
    "BaseNerveEstimator",
    "EstimationResult",
    "VagusEstimator",
    "EBSLNEstimator",
    "RLNEstimator",
    "PhrenicEstimator",
]

ESTIMATOR_CLASSES = {
    "vagus": VagusEstimator,
    "ebsln": EBSLNEstimator,
    "rln": RLNEstimator,
    "phrenic": PhrenicEstimator,
}
