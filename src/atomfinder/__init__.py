"""Atomic column detection in HAADF-STEM images: simulator, detectors,
sub-pixel refinement, matched metrics, and a reproducible benchmark."""

from atomfinder.detect import (
    detect_local_max,
    detect_log,
    detect_ncc,
    gaussian_template,
    peaks_from_heatmap,
)
from atomfinder.metrics import (
    DetectionResult,
    filter_margin,
    match_positions,
    per_species_recall,
)
from atomfinder.net import UNet, predict_heatmap
from atomfinder.real import crop_to_multiple, load_image
from atomfinder.refine import refine_com, refine_gaussian
from atomfinder.sim import (
    PRESETS,
    LatticeSpec,
    SimConfig,
    SimResult,
    column_weight,
    preset_config,
    simulate_image,
)
from atomfinder.train import TrainSettings, train_unet

__all__ = [
    "PRESETS",
    "LatticeSpec",
    "SimConfig",
    "SimResult",
    "column_weight",
    "preset_config",
    "simulate_image",
    "detect_log",
    "detect_local_max",
    "detect_ncc",
    "gaussian_template",
    "peaks_from_heatmap",
    "UNet",
    "predict_heatmap",
    "refine_com",
    "refine_gaussian",
    "match_positions",
    "per_species_recall",
    "filter_margin",
    "DetectionResult",
    "TrainSettings",
    "train_unet",
    "load_image",
    "crop_to_multiple",
]

__version__ = "0.2.1"
