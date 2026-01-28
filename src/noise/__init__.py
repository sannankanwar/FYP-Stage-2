"""
Noise Pipeline Module for Metalens Wrapped-Phase Simulation.

This module provides modular, configurable noise components for simulating
realistic fabrication and measurement artifacts in metalens phase maps.

Noise Components:
-----------------
- CoordinateWarpGRF: Smooth spatial distortions via Gaussian Random Field displacement
- FabricationGRFPhaseNoise: Spatially-correlated additive phase noise
- StructuredSinusoidArtifact: Periodic sinusoidal phase artifacts
- SensorGrain: High-frequency sensor noise (Gaussian or Poisson)
- DeadPixels: Dead pixel regions with constant phase values
- WrapPhase: Final phase wrapping to [-π, π)

Usage:
------
from src.noise import NoisePipeline, get_default_noise_config

config = get_default_noise_config()
config['fabrication_grf']['enabled'] = True
config['sensor_grain']['enabled'] = True

pipeline = NoisePipeline(config)
phi_noisy, img2_noisy, step_outputs = pipeline.apply(phi_clean, img2_clean)
"""

from .noise_pipeline import (
    NoisePipeline,
    NoiseComponent,
    CoordinateWarpGRF,
    FabricationGRFPhaseNoise,
    StructuredSinusoidArtifact,
    SensorGrain,
    DeadPixels,
    WrapPhase,
    generate_wrapped_phase_image,
    get_default_noise_config,
    phi_to_img2,
    img2_to_phi,
)

__all__ = [
    'NoisePipeline',
    'NoiseComponent',
    'CoordinateWarpGRF',
    'FabricationGRFPhaseNoise',
    'StructuredSinusoidArtifact',
    'SensorGrain',
    'DeadPixels',
    'WrapPhase',
    'generate_wrapped_phase_image',
    'get_default_noise_config',
    'phi_to_img2',
    'img2_to_phi',
]
