"""Augmentation modulleri."""
from .camo_synth import CamoSynthAugmenter, CamoSynthConfig, synthetic_camouflage
from .lee_filter import LearnableLeeFilter, lee_filter
from .stress import (
    PRESET_STRESS,
    StressConfig,
    StressEvaluator,
    add_cloud_overlay,
    apply_stress,
    simulate_low_light,
)

__all__ = [
    "lee_filter", "LearnableLeeFilter",
    "synthetic_camouflage", "CamoSynthConfig", "CamoSynthAugmenter",
    "add_cloud_overlay", "simulate_low_light",
    "StressConfig", "PRESET_STRESS", "apply_stress", "StressEvaluator",
]
