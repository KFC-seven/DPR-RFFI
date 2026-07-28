"""Paper-facing DPR-RFFI implementation."""

from .model import DPRConfig, DPRRFFI, Prediction, correlation_calibrated_weight
from .perturbations import PerturbationEngine, PerturbationSpec, default_perturbation_specs
from .screening import PerturbationScreeningResult, screen_perturbations

__all__ = [
    "DPRConfig",
    "DPRRFFI",
    "Prediction",
    "correlation_calibrated_weight",
    "PerturbationEngine",
    "PerturbationSpec",
    "PerturbationScreeningResult",
    "default_perturbation_specs",
    "screen_perturbations",
]
