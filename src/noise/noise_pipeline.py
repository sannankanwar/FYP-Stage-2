"""
Noise Pipeline for Metalens Wrapped-Phase Simulation.

This module implements modular, configurable noise components for simulating
realistic fabrication and measurement artifacts in wrapped phase maps.

Each noise component is independently togglable and operates on the phase 
tensor φ (B, H, W) in the phase domain. The final output maintains both the 
wrapped phase φ and its 2-channel representation img2 (B, 2, H, W) = [cos(φ), sin(φ)].

Noise Components (applied in canonical order):
----------------------------------------------
1. CoordinateWarpGRF (A2): Smooth spatial distortions via GRF displacement field
2. FabricationGRFPhaseNoise (A1): Spatially-correlated additive phase noise
3. StructuredSinusoidArtifact (B2): Periodic sinusoidal phase artifacts
4. SensorGrain (A3): High-frequency sensor noise (Gaussian/Poisson)
5. DeadPixels (C1): Dead pixel regions with constant phase values
6. WrapPhase: Final phase wrapping to enforce [-π, π)

Configuration:
--------------
Each component has an 'enabled' field to toggle it on/off.
Set 'seed' in the top-level config for reproducibility.
Per-component seeds can override the master seed.

Demo Usage:
-----------
Run this file directly:
    python -m src.noise.noise_pipeline

This will:
1. Generate a clean baseline wrapped phase image
2. Apply each noise component independently (from the clean baseline)
3. Show before/after visualizations for each component
4. Apply all enabled components in pipeline order for composite demo
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod
import math

import torch
import torch.nn.functional as F
import numpy as np

# Import forward model utilities (DO NOT MODIFY)
from src.inversion.forward_model import compute_hyperbolic_phase, wrap_phase, get_2channel_representation


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def phi_to_img2(phi: torch.Tensor) -> torch.Tensor:
    """
    Convert wrapped phase φ (B, H, W) to 2-channel representation (B, 2, H, W).
    
    Args:
        phi: Wrapped phase tensor of shape (B, H, W)
        
    Returns:
        img2: 2-channel tensor (B, 2, H, W) where channel 0=cos(φ), channel 1=sin(φ)
    """
    # Use get_2channel_representation and permute to (B, 2, H, W)
    img2 = get_2channel_representation(phi)  # (B, H, W, 2)
    return img2.permute(0, 3, 1, 2)  # (B, 2, H, W)


def img2_to_phi(img2: torch.Tensor) -> torch.Tensor:
    """
    Convert 2-channel representation (B, 2, H, W) back to wrapped phase φ (B, H, W).
    
    Args:
        img2: 2-channel tensor (B, 2, H, W) where channel 0=cos(φ), channel 1=sin(φ)
        
    Returns:
        phi: Wrapped phase tensor of shape (B, H, W) in [-π, π)
    """
    cos_phi = img2[:, 0, :, :]  # (B, H, W)
    sin_phi = img2[:, 1, :, :]  # (B, H, W)
    return torch.atan2(sin_phi, cos_phi)  # (B, H, W)


def generate_grf_2d(
    H: int, 
    W: int, 
    correlation_length_px: float,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    anisotropic: bool = False,
    anisotropy_ratio: float = 1.0,
) -> torch.Tensor:
    """
    Generate a 2D Gaussian Random Field (GRF) using FFT filtering.
    
    Method: Generate white noise in spatial domain, apply Gaussian filter in 
    frequency domain to create smooth, spatially-correlated noise.
    
    Args:
        H, W: Spatial dimensions
        correlation_length_px: Correlation length in pixels
        generator: Torch random generator for reproducibility
        device: Target device
        dtype: Target dtype
        anisotropic: If True, use different correlation scales for x and y
        anisotropy_ratio: Ratio of correlation lengths (y/x) when anisotropic
        
    Returns:
        GRF tensor of shape (H, W), zero mean, unit variance
    """
    # Generate white noise
    white_noise = torch.randn(H, W, generator=generator, device=device, dtype=dtype)
    
    # Create frequency grids
    freq_y = torch.fft.fftfreq(H, d=1.0, device=device, dtype=dtype)
    freq_x = torch.fft.fftfreq(W, d=1.0, device=device, dtype=dtype)
    freq_yy, freq_xx = torch.meshgrid(freq_y, freq_x, indexing='ij')
    
    # Gaussian filter in frequency domain
    # σ_freq = 1 / (2π * correlation_length)
    if anisotropic and anisotropy_ratio > 1.0:
        sigma_freq_x = 1.0 / (2.0 * math.pi * correlation_length_px)
        sigma_freq_y = 1.0 / (2.0 * math.pi * correlation_length_px * anisotropy_ratio)
        freq_sq = (freq_xx / sigma_freq_x) ** 2 + (freq_yy / sigma_freq_y) ** 2
    else:
        sigma_freq = 1.0 / (2.0 * math.pi * correlation_length_px)
        freq_sq = (freq_xx ** 2 + freq_yy ** 2) / (sigma_freq ** 2)
    
    gaussian_filter = torch.exp(-0.5 * freq_sq)
    
    # Apply filter in frequency domain
    noise_fft = torch.fft.fft2(white_noise)
    filtered_fft = noise_fft * gaussian_filter
    grf = torch.fft.ifft2(filtered_fft).real
    
    # Normalize to zero mean, unit variance
    grf = (grf - grf.mean()) / (grf.std() + 1e-8)
    
    return grf


# ============================================================================
# CONFIGURATION STRUCTURES
# ============================================================================

def get_default_noise_config() -> Dict[str, Any]:
    """
    Returns the default noise pipeline configuration with all components disabled.
    
    Returns:
        Dict containing configuration for all noise components
    """
    return {
        'seed': 42,
        'pipeline_order': [
            'coordinate_warp',
            'fabrication_grf',
            'structured_sinusoid',
            'sensor_grain',
            'dead_pixels',
            'wrap_phase',
        ],
        
        'coordinate_warp': {
            'enabled': False,
            'seed': None,  # Uses master seed if None
            'displacement_std_px': 1.0,
            'correlation_length_px': 5.0,
            'anisotropic': False,
            'anisotropy_ratio': 1.1,
            'interpolation': 'bilinear',  # nearest | bilinear | bicubic
            'padding_mode': 'reflection',  # zeros | reflection | border
        },
        
        'fabrication_grf': {
            'enabled': False,
            'seed': None,
            'amplitude_rad': 0.15,
            'correlation_length_px': 15.0,
            'mean': 0.0,
            'clip_std': 3.0,
        },
        
        'structured_sinusoid': {
            'enabled': False,
            'seed': None,
            'amplitude_rad': 0.1,
            'spatial_frequency_px': 50.0,  # Period in pixels
            'orientation_deg': 45.0,
            'phase_offset_rad': 'random',  # 'random' or 'fixed'
            'fixed_phase_offset_rad': 0.0,
        },
        
        'sensor_grain': {
            'enabled': False,
            'seed': None,
            'noise_type': 'gaussian',  # gaussian | poisson
            'std_rad': 0.2,
            'spatially_correlated': False,
            'correlation_length_px': 2.0,
        },
        
        'dead_pixels': {
            'enabled': False,
            'seed': None,
            'density': 4e-4,  # Target ~4 pixels at 1024x1024
            'region_type': 'pixels',  # pixels | blobs
            'blob_radius_px': [0.5, 1.5],  # [min, max]
            'phase_value_mode': 'fixed',  # random | fixed
            'fixed_phase_rad': 0.0,
            'distribution': 'uniform',  # uniform | clustered
        },
        
        'wrap_phase': {
            'enabled': True,  # Always enabled for wrapped-phase data
        },
    }


# ============================================================================
# BASE CLASS
# ============================================================================

class NoiseComponent(ABC):
    """
    Abstract base class for noise components.
    
    Each noise component implements:
    - __init__(cfg, rng): Initialize with config dict and torch.Generator
    - apply(phi, img2, metadata) -> (phi_out, img2_out): Apply noise transformation
    
    The component operates primarily on φ (wrapped phase) and updates both φ and img2.
    """
    
    def __init__(self, cfg: Dict[str, Any], rng: torch.Generator):
        """
        Initialize noise component.
        
        Args:
            cfg: Component-specific configuration dict
            rng: Torch random generator for reproducibility
        """
        self.cfg = cfg
        self.rng = rng
        self.enabled = cfg.get('enabled', False)
    
    @abstractmethod
    def apply(
        self, 
        phi: torch.Tensor, 
        img2: torch.Tensor, 
        metadata: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply noise to the phase tensor.
        
        Args:
            phi: Wrapped phase tensor (B, H, W)
            img2: 2-channel representation (B, 2, H, W)
            metadata: Dict containing image dimensions, device, etc.
            
        Returns:
            Tuple of (phi_out, img2_out) with noise applied
        """
        pass
    
    def __call__(
        self, 
        phi: torch.Tensor, 
        img2: torch.Tensor, 
        metadata: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Callable interface for applying noise."""
        if not self.enabled:
            return phi, img2
        return self.apply(phi, img2, metadata)


# ============================================================================
# NOISE COMPONENTS
# ============================================================================

class CoordinateWarpGRF(NoiseComponent):
    """
    (A2) Coordinate Warp via Gaussian Random Field.
    
    Generates smooth displacement fields u(x,y) and v(x,y) using GRF and 
    resamples the phase map: φ'(x,y) = φ(x+u, y+v).
    
    This simulates optical aberrations and mechanical distortions.
    """
    
    def apply(
        self, 
        phi: torch.Tensor, 
        img2: torch.Tensor, 
        metadata: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, W = phi.shape
        device = phi.device
        dtype = phi.dtype
        
        cfg = self.cfg
        displacement_std = cfg['displacement_std_px']
        correlation_length = cfg['correlation_length_px']
        anisotropic = cfg.get('anisotropic', False)
        anisotropy_ratio = cfg.get('anisotropy_ratio', 1.0)
        interpolation = cfg.get('interpolation', 'bilinear')
        padding_mode = cfg.get('padding_mode', 'reflection')
        
        # Map interpolation mode to grid_sample mode
        interp_map = {'nearest': 'nearest', 'bilinear': 'bilinear', 'bicubic': 'bicubic'}
        grid_mode = interp_map.get(interpolation, 'bilinear')
        
        # Map padding mode
        pad_map = {'zeros': 'zeros', 'reflection': 'reflection', 'border': 'border', 'wrap': 'reflection'}
        grid_padding = pad_map.get(padding_mode, 'reflection')
        
        # Generate displacement fields for u and v using GRF
        u_field = generate_grf_2d(
            H, W, correlation_length, self.rng, device, dtype,
            anisotropic, anisotropy_ratio
        ) * displacement_std
        
        v_field = generate_grf_2d(
            H, W, correlation_length, self.rng, device, dtype,
            anisotropic, anisotropy_ratio
        ) * displacement_std
        
        # Create base coordinate grid in [-1, 1] for grid_sample
        # Note: grid_sample expects (B, H, W, 2) where last dim is (x, y)
        y_coords = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        x_coords = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Convert displacement from pixels to normalized coordinates
        # Displacement in pixels -> normalized: disp_norm = disp_px * 2 / dim
        u_norm = u_field * (2.0 / W)
        v_norm = v_field * (2.0 / H)
        
        # Add displacement to base grid
        grid_x_warped = grid_x + u_norm
        grid_y_warped = grid_y + v_norm
        
        # Stack and expand for batch
        grid_warped = torch.stack([grid_x_warped, grid_y_warped], dim=-1)  # (H, W, 2)
        grid_warped = grid_warped.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)
        
        # Resample phi using grid_sample (requires (B, C, H, W) input)
        phi_expanded = phi.unsqueeze(1)  # (B, 1, H, W)
        phi_warped = F.grid_sample(
            phi_expanded, grid_warped, 
            mode=grid_mode, padding_mode=grid_padding, align_corners=True
        )
        phi_warped = phi_warped.squeeze(1)  # (B, H, W)
        
        # Update img2 from warped phi
        img2_warped = phi_to_img2(phi_warped)
        
        return phi_warped, img2_warped


class FabricationGRFPhaseNoise(NoiseComponent):
    """
    (A1) Fabrication GRF Phase Noise.
    
    Adds smooth, spatially-correlated phase noise in radians:
    φ <- φ + δφ
    
    This simulates fabrication errors and material non-uniformities.
    """
    
    def apply(
        self, 
        phi: torch.Tensor, 
        img2: torch.Tensor, 
        metadata: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, W = phi.shape
        device = phi.device
        dtype = phi.dtype
        
        cfg = self.cfg
        amplitude = cfg['amplitude_rad']
        correlation_length = cfg['correlation_length_px']
        mean = cfg.get('mean', 0.0)
        clip_std = cfg.get('clip_std', 3.0)
        
        # Generate GRF noise for each batch element
        delta_phi = torch.zeros_like(phi)
        for b in range(B):
            grf = generate_grf_2d(
                H, W, correlation_length, self.rng, device, dtype
            )
            delta_phi[b] = grf
        
        # Scale by amplitude, add mean, and clip
        delta_phi = mean + amplitude * delta_phi
        if clip_std > 0:
            delta_phi = torch.clamp(delta_phi, -clip_std * amplitude, clip_std * amplitude)
        
        # Add noise to phase
        phi_noisy = phi + delta_phi
        
        # Update img2 (do NOT wrap here - wrapping is a separate pipeline step)
        img2_noisy = phi_to_img2(phi_noisy)
        
        return phi_noisy, img2_noisy


class StructuredSinusoidArtifact(NoiseComponent):
    """
    (B2) Structured Sinusoidal Artifact.
    
    Adds a weak sinusoidal phase pattern:
    φ <- φ + A * sin(2π*(x*cos(θ) + y*sin(θ))/period + offset)
    
    This simulates interference patterns and systematic artifacts.
    """
    
    def apply(
        self, 
        phi: torch.Tensor, 
        img2: torch.Tensor, 
        metadata: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, W = phi.shape
        device = phi.device
        dtype = phi.dtype
        
        cfg = self.cfg
        amplitude = cfg['amplitude_rad']
        period = cfg['spatial_frequency_px']  # Period in pixels
        orientation_deg = cfg['orientation_deg']
        phase_offset_mode = cfg.get('phase_offset_rad', 'random')
        fixed_offset = cfg.get('fixed_phase_offset_rad', 0.0)
        
        # Determine phase offset
        if phase_offset_mode == 'random':
            offset = torch.rand(1, generator=self.rng, device=device, dtype=dtype).item() * 2 * math.pi
        else:
            offset = fixed_offset
        
        # Convert orientation to radians
        theta = math.radians(orientation_deg)
        
        # Create coordinate grids in pixel units
        y_coords = torch.arange(H, device=device, dtype=dtype)
        x_coords = torch.arange(W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Compute sinusoidal pattern
        # Wave direction: (cos(θ), sin(θ))
        spatial_phase = 2 * math.pi * (xx * math.cos(theta) + yy * math.sin(theta)) / period
        sinusoid = amplitude * torch.sin(spatial_phase + offset)
        
        # Expand to batch dimension
        sinusoid = sinusoid.unsqueeze(0).expand(B, -1, -1)
        
        # Add to phase
        phi_noisy = phi + sinusoid
        img2_noisy = phi_to_img2(phi_noisy)
        
        return phi_noisy, img2_noisy


class SensorGrain(NoiseComponent):
    """
    (A3) Sensor Grain Noise.
    
    Adds high-frequency sensor noise in the phase domain:
    - Gaussian: φ <- φ + N(0, σ²)
    - Poisson: Scaled Poisson noise approximation
    
    Optionally applies light spatial correlation via low-pass filtering.
    """
    
    def apply(
        self, 
        phi: torch.Tensor, 
        img2: torch.Tensor, 
        metadata: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, W = phi.shape
        device = phi.device
        dtype = phi.dtype
        
        cfg = self.cfg
        noise_type = cfg.get('noise_type', 'gaussian')
        std = cfg.get('std_rad', 0.05)
        spatially_correlated = cfg.get('spatially_correlated', False)
        corr_length = cfg.get('correlation_length_px', 2.0)
        
        if noise_type == 'gaussian':
            noise = torch.randn(B, H, W, generator=self.rng, device=device, dtype=dtype) * std
        elif noise_type == 'poisson':
            # Approximate Poisson noise in phase domain
            # Scale factor to make Poisson variance roughly match desired std
            scale = 10.0 / (std ** 2 + 1e-8)
            poisson_samples = torch.poisson(
                torch.ones(B, H, W, device=device, dtype=dtype) * scale
            )
            noise = (poisson_samples - scale) / math.sqrt(scale) * std
        else:
            noise = torch.zeros(B, H, W, device=device, dtype=dtype)
        
        # Apply light spatial correlation if enabled
        if spatially_correlated and corr_length > 0:
            # Use a simple Gaussian blur kernel
            kernel_size = int(2 * corr_length) * 2 + 1
            kernel_size = max(3, min(kernel_size, 15))  # Clamp kernel size
            sigma = corr_length / 2
            
            # Create Gaussian kernel
            x = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
            kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
            kernel_1d = kernel_1d / kernel_1d.sum()
            kernel_2d = kernel_1d.view(-1, 1) @ kernel_1d.view(1, -1)
            kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
            
            # Apply convolution
            noise = noise.unsqueeze(1)  # (B, 1, H, W)
            padding = kernel_size // 2
            noise = F.conv2d(noise, kernel_2d, padding=padding)
            noise = noise.squeeze(1)  # (B, H, W)
            
            # Renormalize to maintain std
            noise = noise / (noise.std() + 1e-8) * std
        
        phi_noisy = phi + noise
        img2_noisy = phi_to_img2(phi_noisy)
        
        return phi_noisy, img2_noisy


class DeadPixels(NoiseComponent):
    """
    (C1) Dead Pixels.
    
    Creates a mask of dead pixel regions and overwrites phase values with a 
    constant phase. Supports individual pixels or circular blobs.
    
    This simulates sensor defects and stuck pixels.
    """
    
    def apply(
        self, 
        phi: torch.Tensor, 
        img2: torch.Tensor, 
        metadata: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, W = phi.shape
        device = phi.device
        dtype = phi.dtype
        
        cfg = self.cfg
        density = cfg.get('density', 0.01)
        region_type = cfg.get('region_type', 'blobs')
        blob_radius_range = cfg.get('blob_radius_px', [3.0, 8.0])
        phase_mode = cfg.get('phase_value_mode', 'fixed')
        fixed_phase = cfg.get('fixed_phase_rad', 0.0)
        distribution = cfg.get('distribution', 'uniform')
        
        mask = torch.zeros(B, H, W, device=device, dtype=torch.bool)
        phase_values = torch.zeros(B, H, W, device=device, dtype=dtype)
        
        total_pixels = H * W
        target_dead_pixels = int(density * total_pixels)
        
        for b in range(B):
            if region_type == 'pixels':
                # Generate random dead pixel locations
                indices = torch.randperm(total_pixels, generator=self.rng, device=device)[:target_dead_pixels]
                mask_flat = torch.zeros(total_pixels, device=device, dtype=torch.bool)
                mask_flat[indices] = True
                mask[b] = mask_flat.view(H, W)
                
                if phase_mode == 'random':
                    random_phases = torch.rand(total_pixels, generator=self.rng, device=device, dtype=dtype) * 2 * math.pi - math.pi
                    phase_values[b] = random_phases.view(H, W)
                else:
                    phase_values[b] = fixed_phase
                    
            elif region_type == 'blobs':
                # Estimate number of blobs needed to cover target area
                avg_radius = (blob_radius_range[0] + blob_radius_range[1]) / 2
                avg_blob_area = math.pi * avg_radius ** 2
                n_blobs = max(1, int(target_dead_pixels / avg_blob_area))
                
                # Generate blob centers
                if distribution == 'clustered':
                    # Clustered: centers near each other
                    cluster_center_y = torch.rand(1, generator=self.rng, device=device, dtype=dtype).item() * H
                    cluster_center_x = torch.rand(1, generator=self.rng, device=device, dtype=dtype).item() * W
                    spread = min(H, W) / 4
                    
                    centers_y = cluster_center_y + torch.randn(n_blobs, generator=self.rng, device=device, dtype=dtype) * spread
                    centers_x = cluster_center_x + torch.randn(n_blobs, generator=self.rng, device=device, dtype=dtype) * spread
                    centers_y = torch.clamp(centers_y, 0, H - 1)
                    centers_x = torch.clamp(centers_x, 0, W - 1)
                else:
                    # Uniform distribution
                    centers_y = torch.rand(n_blobs, generator=self.rng, device=device, dtype=dtype) * H
                    centers_x = torch.rand(n_blobs, generator=self.rng, device=device, dtype=dtype) * W
                
                # Generate radii
                radii = blob_radius_range[0] + torch.rand(n_blobs, generator=self.rng, device=device, dtype=dtype) * (blob_radius_range[1] - blob_radius_range[0])
                
                # Create coordinate grids
                yy, xx = torch.meshgrid(
                    torch.arange(H, device=device, dtype=dtype),
                    torch.arange(W, device=device, dtype=dtype),
                    indexing='ij'
                )
                
                # Draw blobs
                batch_mask = torch.zeros(H, W, device=device, dtype=torch.bool)
                batch_phase = torch.zeros(H, W, device=device, dtype=dtype)
                
                for i in range(n_blobs):
                    dist_sq = (yy - centers_y[i]) ** 2 + (xx - centers_x[i]) ** 2
                    blob_mask = dist_sq <= radii[i] ** 2
                    
                    if phase_mode == 'random':
                        blob_phase = torch.rand(1, generator=self.rng, device=device, dtype=dtype).item() * 2 * math.pi - math.pi
                    else:
                        blob_phase = fixed_phase
                    
                    # Update phase values first (before mask OR, so each blob can have its own value)
                    batch_phase = torch.where(blob_mask, torch.tensor(blob_phase, device=device, dtype=dtype), batch_phase)
                    batch_mask = batch_mask | blob_mask
                
                mask[b] = batch_mask
                phase_values[b] = batch_phase
        
        # Apply dead pixel mask
        phi_noisy = torch.where(mask, phase_values, phi)
        img2_noisy = phi_to_img2(phi_noisy)
        
        return phi_noisy, img2_noisy


class WrapPhase(NoiseComponent):
    """
    Final Wrap Phase Operation.
    
    Wraps phase to [-π, π) using atan2(sin(φ), cos(φ)).
    This should always be enabled as the final step for wrapped-phase data.
    """
    
    def __init__(self, cfg: Dict[str, Any], rng: torch.Generator):
        super().__init__(cfg, rng)
        # WrapPhase is always enabled for wrapped-phase output
        self.enabled = cfg.get('enabled', True)
    
    def apply(
        self, 
        phi: torch.Tensor, 
        img2: torch.Tensor, 
        metadata: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Wrap using atan2(sin, cos) for numerical stability
        phi_wrapped = torch.atan2(torch.sin(phi), torch.cos(phi))
        img2_wrapped = phi_to_img2(phi_wrapped)
        
        return phi_wrapped, img2_wrapped


# ============================================================================
# NOISE PIPELINE ORCHESTRATOR
# ============================================================================

class NoisePipeline:
    """
    Orchestrator for the noise pipeline.
    
    Manages the creation and execution of noise components in the canonical order,
    with support for enabling/disabling components and tracking per-step outputs.
    """
    
    # Component mapping
    COMPONENT_CLASSES = {
        'coordinate_warp': CoordinateWarpGRF,
        'fabrication_grf': FabricationGRFPhaseNoise,
        'structured_sinusoid': StructuredSinusoidArtifact,
        'sensor_grain': SensorGrain,
        'dead_pixels': DeadPixels,
        'wrap_phase': WrapPhase,
    }
    
    def __init__(self, cfg: Dict[str, Any]):
        """
        Initialize the noise pipeline.
        
        Args:
            cfg: Full noise pipeline configuration dict
        """
        self.cfg = cfg
        self.master_seed = cfg.get('seed', 42)
        self.pipeline_order = cfg.get('pipeline_order', list(self.COMPONENT_CLASSES.keys()))
        
        # Create master generator
        self.master_rng = torch.Generator()
        self.master_rng.manual_seed(self.master_seed)
        
        # Initialize components
        self.components = {}
        for name in self.pipeline_order:
            component_cfg = cfg.get(name, {'enabled': False})
            
            # Determine seed for this component
            component_seed = component_cfg.get('seed')
            if component_seed is None:
                # Derive from master seed using component name hash
                component_seed = self.master_seed + hash(name) % 10000
            
            # Create component-specific generator
            component_rng = torch.Generator()
            component_rng.manual_seed(component_seed)
            
            # Instantiate component
            component_class = self.COMPONENT_CLASSES.get(name)
            if component_class is not None:
                self.components[name] = component_class(component_cfg, component_rng)
    
    def apply(
        self, 
        phi: torch.Tensor, 
        img2: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]:
        """
        Apply the full noise pipeline.
        
        Args:
            phi: Wrapped phase tensor (B, H, W)
            img2: Optional 2-channel representation (B, 2, H, W). 
                  If None, will be computed from phi.
                  
        Returns:
            Tuple of:
            - phi_out: Output wrapped phase (B, H, W)
            - img2_out: Output 2-channel representation (B, 2, H, W)
            - step_outputs: Dict mapping component name to {'before': (phi, img2), 'after': (phi, img2)}
        """
        # Ensure we have both representations
        if img2 is None:
            img2 = phi_to_img2(phi)
        
        # Build metadata
        B, H, W = phi.shape
        metadata = {
            'H': H,
            'W': W,
            'B': B,
            'device': phi.device,
            'dtype': phi.dtype,
        }
        
        step_outputs = {}
        current_phi = phi.clone()
        current_img2 = img2.clone()
        
        for name in self.pipeline_order:
            component = self.components.get(name)
            if component is None or not component.enabled:
                continue
            
            # Store before state
            before_phi = current_phi.clone()
            before_img2 = current_img2.clone()
            
            # Apply component
            current_phi, current_img2 = component(current_phi, current_img2, metadata)
            
            # Store step outputs
            step_outputs[name] = {
                'before_phi': before_phi,
                'before_img2': before_img2,
                'after_phi': current_phi.clone(),
                'after_img2': current_img2.clone(),
            }
        
        return current_phi, current_img2, step_outputs

    def __call__(
        self, 
        phi: torch.Tensor, 
        img2: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Callable interface for the noise pipeline.
        
        Args:
            phi: Wrapped phase tensor (B, H, W)
            img2: Optional 2-channel representation (B, 2, H, W)
            
        Returns:
            Tuple of (phi_out, img2_out)
        """
        phi_out, img2_out, _ = self.apply(phi, img2)
        return phi_out, img2_out
    
    def apply_single_component(
        self, 
        component_name: str, 
        phi: torch.Tensor, 
        img2: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply a single noise component (for ablation studies).
        
        Args:
            component_name: Name of the component to apply
            phi: Wrapped phase tensor (B, H, W)
            img2: Optional 2-channel representation (B, 2, H, W)
            
        Returns:
            Tuple of (phi_out, img2_out)
        """
        if img2 is None:
            img2 = phi_to_img2(phi)
        
        B, H, W = phi.shape
        metadata = {
            'H': H,
            'W': W,
            'B': B,
            'device': phi.device,
            'dtype': phi.dtype,
        }
        
        component = self.components.get(component_name)
        if component is None:
            raise ValueError(f"Unknown component: {component_name}")
        
        # Temporarily enable component
        original_enabled = component.enabled
        component.enabled = True
        
        phi_out, img2_out = component(phi.clone(), img2.clone(), metadata)
        
        # Restore original enabled state
        component.enabled = original_enabled
        
        return phi_out, img2_out


# ============================================================================
# BASELINE PHASE IMAGE GENERATION
# ============================================================================

def generate_wrapped_phase_image(
    xc: float,
    yc: float,
    S: float,
    focal_length: float,
    wavelength: float,
    H: int,
    W: int,
    device: Union[str, torch.device] = 'cpu',
    batch_size: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a baseline wrapped phase image using the forward model.
    
    Args:
        xc, yc: Center coordinates in physical units (micrometers)
        S: Field of view / window size in physical units (micrometers)
        focal_length: Focal length in micrometers
        wavelength: Wavelength in micrometers
        H, W: Image height and width in pixels
        device: Target device ('cpu' or 'cuda')
        batch_size: Batch size
        
    Returns:
        Tuple of:
        - phi: Wrapped phase tensor (B, H, W) in [-π, π)
        - img2: 2-channel representation (B, 2, H, W)
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    # Create normalized grid in [-0.5, 0.5]
    y_norm = torch.linspace(-0.5, 0.5, H, device=device)
    x_norm = torch.linspace(-0.5, 0.5, W, device=device)
    grid_y, grid_x = torch.meshgrid(y_norm, x_norm, indexing='ij')
    
    # Convert to physical coordinates
    X_phys = xc + S * grid_x
    Y_phys = yc + S * grid_y
    
    # Compute phase using forward model
    phi_unwrapped = compute_hyperbolic_phase(X_phys, Y_phys, focal_length, wavelength)
    phi_wrapped = wrap_phase(phi_unwrapped)
    
    # Add batch dimension
    phi_wrapped = phi_wrapped.unsqueeze(0).expand(batch_size, -1, -1)
    
    # Get 2-channel representation
    img2 = phi_to_img2(phi_wrapped)
    
    return phi_wrapped, img2


# ============================================================================
# DEMO / VISUALIZATION
# ============================================================================

def run_demo():
    """
    Run the noise pipeline demo with visualizations.
    
    Demonstrates:
    1. Clean baseline phase image
    2. Each noise component applied independently
    3. Composite noise with all enabled components
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    
    print("=" * 70)
    print("NOISE PIPELINE DEMO")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Example metalens parameters
    H, W = 256, 256
    xc, yc = 0.0, 0.0  # Center at origin (micrometers)
    S = 200.0  # Field of view (micrometers)
    focal_length = 500.0  # Focal length (micrometers)
    wavelength = 0.532  # 532 nm green light (micrometers)
    
    print(f"\nMetalens Parameters:")
    print(f"  Image size: {H}x{W}")
    print(f"  Center: ({xc}, {yc}) µm")
    print(f"  Field of view: {S} µm")
    print(f"  Focal length: {focal_length} µm")
    print(f"  Wavelength: {wavelength} µm ({wavelength * 1000:.0f} nm)")
    
    # Generate baseline
    print("\nGenerating baseline wrapped phase image...")
    phi_clean, img2_clean = generate_wrapped_phase_image(
        xc, yc, S, focal_length, wavelength, H, W, device
    )
    print(f"  phi shape: {phi_clean.shape}")
    print(f"  img2 shape: {img2_clean.shape}")
    print(f"  phi range: [{phi_clean.min().item():.3f}, {phi_clean.max().item():.3f}]")
    
    # -------------------------------------------------------------------------
    # Configure noise components for demo
    # -------------------------------------------------------------------------
    config = get_default_noise_config()
    config['seed'] = 42
    
    # Enable all components with demo parameters
    config['coordinate_warp']['enabled'] = True
    config['coordinate_warp']['displacement_std_px'] = 3.0
    config['coordinate_warp']['correlation_length_px'] = 25.0
    
    config['fabrication_grf']['enabled'] = True
    config['fabrication_grf']['amplitude_rad'] = 0.4
    config['fabrication_grf']['correlation_length_px'] = 20.0
    
    config['structured_sinusoid']['enabled'] = True
    config['structured_sinusoid']['amplitude_rad'] = 0.2
    config['structured_sinusoid']['spatial_frequency_px'] = 40.0
    config['structured_sinusoid']['orientation_deg'] = 30.0
    
    config['sensor_grain']['enabled'] = True
    config['sensor_grain']['noise_type'] = 'gaussian'
    config['sensor_grain']['std_rad'] = 0.1
    
    config['dead_pixels']['enabled'] = True
    config['dead_pixels']['density'] = 0.02
    config['dead_pixels']['region_type'] = 'blobs'
    config['dead_pixels']['blob_radius_px'] = [4.0, 10.0]
    config['dead_pixels']['phase_value_mode'] = 'random'
    
    config['wrap_phase']['enabled'] = True
    
    print("\nNoise Configuration:")
    print(f"  Seed: {config['seed']}")
    for name in config['pipeline_order']:
        if name in config:
            status = "ENABLED" if config[name].get('enabled', False) else "disabled"
            print(f"  {name}: {status}")
    
    # Create pipeline
    pipeline = NoisePipeline(config)
    
    # -------------------------------------------------------------------------
    # Individual Component Demos (each from clean baseline)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("INDIVIDUAL COMPONENT DEMOS")
    print("=" * 70)
    
    component_names = [
        'coordinate_warp',
        'fabrication_grf',
        'structured_sinusoid',
        'sensor_grain',
        'dead_pixels',
    ]
    
    individual_results = {}
    
    for name in component_names:
        print(f"\nApplying {name} to clean baseline...")
        phi_after, img2_after = pipeline.apply_single_component(name, phi_clean, img2_clean)
        individual_results[name] = {
            'before_phi': phi_clean,
            'after_phi': phi_after,
        }
        
        # Compute difference
        diff = (phi_after - phi_clean).abs()
        print(f"  Mean absolute phase change: {diff.mean().item():.4f} rad")
        print(f"  Max absolute phase change: {diff.max().item():.4f} rad")
    
    # -------------------------------------------------------------------------
    # Composite Pipeline Demo
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("COMPOSITE PIPELINE DEMO")
    print("=" * 70)
    
    print("\nApplying full pipeline with all enabled components...")
    phi_composite, img2_composite, step_outputs = pipeline.apply(phi_clean, img2_clean)
    
    print(f"\nComponents applied in order:")
    for name in config['pipeline_order']:
        if name in step_outputs:
            print(f"  ✓ {name}")
    
    composite_diff = (phi_composite - phi_clean).abs()
    print(f"\nTotal effect:")
    print(f"  Mean absolute phase change: {composite_diff.mean().item():.4f} rad")
    print(f"  Max absolute phase change: {composite_diff.max().item():.4f} rad")
    
    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    # Move to CPU for plotting
    phi_clean_np = phi_clean[0].cpu().numpy()
    
    # Color normalization for phase plots
    phase_norm = Normalize(vmin=-np.pi, vmax=np.pi)
    cmap = 'twilight'
    
    # Individual component figures
    for name in component_names:
        phi_after_np = individual_results[name]['after_phi'][0].cpu().numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Noise Component: {name.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        
        im0 = axes[0].imshow(phi_clean_np, cmap=cmap, norm=phase_norm)
        axes[0].set_title('Before (Clean Baseline)')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], label='Phase (rad)', shrink=0.8)
        
        im1 = axes[1].imshow(phi_after_np, cmap=cmap, norm=phase_norm)
        axes[1].set_title(f'After {name.replace("_", " ").title()}')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], label='Phase (rad)', shrink=0.8)
        
        plt.tight_layout()
        plt.savefig(f'noise_demo_{name}.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: noise_demo_{name}.png")
        plt.close()
    
    # Composite figure
    phi_composite_np = phi_composite[0].cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Composite Noise Pipeline (All Components)', fontsize=14, fontweight='bold')
    
    im0 = axes[0].imshow(phi_clean_np, cmap=cmap, norm=phase_norm)
    axes[0].set_title('Before (Clean Baseline)')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], label='Phase (rad)', shrink=0.8)
    
    im1 = axes[1].imshow(phi_composite_np, cmap=cmap, norm=phase_norm)
    axes[1].set_title('After Composite Noise')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], label='Phase (rad)', shrink=0.8)
    
    plt.tight_layout()
    plt.savefig('noise_demo_composite.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: noise_demo_composite.png")
    plt.close()
    
    # Summary figure with all components
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Noise Pipeline Overview', fontsize=16, fontweight='bold')
    
    # Row 1: Clean and individual components
    axes[0, 0].imshow(phi_clean_np, cmap=cmap, norm=phase_norm)
    axes[0, 0].set_title('Clean Baseline')
    axes[0, 0].axis('off')
    
    for i, name in enumerate(component_names[:3]):
        phi_np = individual_results[name]['after_phi'][0].cpu().numpy()
        axes[0, i + 1].imshow(phi_np, cmap=cmap, norm=phase_norm)
        axes[0, i + 1].set_title(name.replace('_', ' ').title())
        axes[0, i + 1].axis('off')
    
    # Row 2: Remaining components and composite
    for i, name in enumerate(component_names[3:]):
        phi_np = individual_results[name]['after_phi'][0].cpu().numpy()
        axes[1, i].imshow(phi_np, cmap=cmap, norm=phase_norm)
        axes[1, i].set_title(name.replace('_', ' ').title())
        axes[1, i].axis('off')
    
    axes[1, 2].imshow(phi_composite_np, cmap=cmap, norm=phase_norm)
    axes[1, 2].set_title('Composite (All)')
    axes[1, 2].axis('off')
    
    # Difference plot
    diff_np = np.abs(phi_composite_np - phi_clean_np)
    im_diff = axes[1, 3].imshow(diff_np, cmap='hot')
    axes[1, 3].set_title('|Composite - Clean|')
    axes[1, 3].axis('off')
    plt.colorbar(im_diff, ax=axes[1, 3], label='|Δφ| (rad)', shrink=0.8)
    
    plt.tight_layout()
    plt.savefig('noise_demo_overview.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: noise_demo_overview.png")
    plt.close()
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    for name in component_names:
        print(f"  - noise_demo_{name}.png")
    print("  - noise_demo_composite.png")
    print("  - noise_demo_overview.png")


if __name__ == '__main__':
    run_demo()
