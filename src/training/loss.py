"""
=============================================================================
Metalens Phase-Map Inversion: Production Loss Module
=============================================================================

PURPOSE
-------
This module provides a clean, modular loss system for training neural networks
that predict metalens parameters (xc, yc, S, f, lambda) from 2-channel wrapped
phase maps (cos, sin).

LOSS MODES (exactly ONE regression mode active at a time)
---------------------------------------------------------
1. UnitStandardizedParamLoss (baseline):
   - For each predicted parameter i: L_i = (pred_i - target_i)^2 / range_i^2
   - Total regression loss = mean over batch of sum over predicted parameters

2. PhaseGradientFlowLoss (phase-gradient based):
   - Trains parameters by matching spatial gradients of reconstructed phase
   - Uses Sobel filters on 2-channel (cos, sin) images
   - L_grad = MSE(grad_pred, grad_target) * (1 / phase_range)

3. KendallUnitStandardizedLoss (per-parameter trainable uncertainty):
   - Wraps unit-standardized terms with learnable log variances
   - L = Σ [ exp(-s_i) * L_i_base + s_i ]
   - Kendall applies ONLY to regression, NOT to physics loss

4. CustomRegressionLoss (placeholder for future work):
   - Safe hook that raises clear error unless callable is provided

PHYSICS LOSS (always additive, never replaces regression)
---------------------------------------------------------
- L_physics = MSE(reconstructed_phase_map, input_phase_map)
- Default weight = 0.1
- PINN-style scheduling: enabled only after physics_start_epoch (default 100)
- Kendall uncertainty NEVER applies to physics loss

LAMBDA (wavelength) HANDLING
----------------------------
- If lambda is PREDICTED: Model outputs 5 values [xc, yc, S, f, lambda]
- If lambda is FIXED: Model outputs 4 values [xc, yc, S, f], lambda provided per-sample
  via batch["lambda_m"] (meters) or batch["lambda_idx"] (index into allowed set)
- Allowed lambda values: [4.05e-7, 4.45e-7, 5.32e-7, 7.25e-7] meters

TRAINING LOOP INTERFACE
-----------------------
loss_module.forward(
    pred_params: Tensor,         # (B, D) where D=4 or 5
    true_params: Tensor,         # (B, D) matching predicted columns
    input_images: Tensor | None, # (B, 2, H, W) for physics loss
    batch: dict | None,          # Contains fixed lambda if not predicted
    epoch: int | None            # For physics scheduling gate
) -> tuple[Tensor, dict]

RETURNED METRICS (stable keys)
------------------------------
Minimum:
- "total_loss"
- "loss_regression"
- "predicted_params" (list of parameter names)

If physics enabled and active:
- "loss_physics"
- "physics_weight"

If Kendall mode:
- "kendall_log_vars" (list of floats per predicted param)

If gradient-flow mode:
- "loss_grad"
- "phase_range"

FACTORY FUNCTION
----------------
build_loss(loss_cfg: dict, data_cfg: dict) -> nn.Module

=============================================================================
"""

from __future__ import annotations

import math
from typing import Callable, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.inversion.forward_model import (
    compute_hyperbolic_phase,
    wrap_phase,
    get_2channel_representation,
)


# =============================================================================
# Constants
# =============================================================================

# Canonical parameter ordering - NEVER assume positional indices
PARAM_ORDER = ["xc", "yc", "S", "f", "lambda"]

# Allowed wavelengths in METERS for fixed-lambda mode
ALLOWED_WAVELENGTHS_M = [4.05e-7, 4.45e-7, 5.32e-7, 7.25e-7]


# =============================================================================
# Type Definitions
# =============================================================================

class ParamRanges(TypedDict):
    """Parameter ranges from data config (in physical units, e.g. micrometers)."""
    xc: tuple[float, float]
    yc: tuple[float, float]
    S: tuple[float, float]
    f: tuple[float, float]
    lambda_: tuple[float, float]  # Note: trailing underscore avoids Python keyword


class LossConfig(TypedDict, total=False):
    """Loss configuration dictionary."""
    mode: str  # "unit_standardized", "gradient_flow", "kendall", "custom"
    physics_enabled: bool
    physics_weight: float
    physics_start_epoch: int
    predicted_params: list[str]
    param_ranges: dict[str, tuple[float, float]]
    phase_range: float
    custom_loss_fn: Callable | None


# =============================================================================
# Helper Functions
# =============================================================================

def extract_param_ranges(data_cfg: dict) -> dict[str, tuple[float, float]]:
    """
    Extract parameter ranges from data config.
    
    Maps config keys to canonical parameter names:
    - xc_range -> xc
    - yc_range -> yc  
    - S_range -> S
    - focal_length_range -> f
    - wavelength_range -> lambda
    """
    ranges = {}
    
    key_map = {
        "xc_range": "xc",
        "yc_range": "yc",
        "S_range": "S",
        "focal_length_range": "f",
        "wavelength_range": "lambda",
    }
    
    for config_key, param_name in key_map.items():
        if config_key in data_cfg:
            val = data_cfg[config_key]
            if isinstance(val, (list, tuple)) and len(val) == 2:
                ranges[param_name] = (float(val[0]), float(val[1]))
    
    return ranges


def build_full_params_for_reconstruction(
    pred_params: torch.Tensor,
    predicted_names: list[str],
    batch: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Construct full physical parameter set for phase map reconstruction.
    
    Args:
        pred_params: (B, D) predicted parameters in order of predicted_names
        predicted_names: list of predicted parameter names (e.g., ["xc","yc","S","f"])
        batch: optional dict containing fixed parameters like lambda_m or lambda_idx
        
    Returns:
        xc, yc, S, focal_length, wavelength: each (B,) tensors in physical units
        
    Raises:
        ValueError: if lambda is required but not available
    """
    B = pred_params.shape[0]
    device = pred_params.device
    
    # Build lookup from predicted names to tensor columns
    param_values = {name: pred_params[:, i] for i, name in enumerate(predicted_names)}
    
    # Extract predicted params
    xc = param_values.get("xc")
    yc = param_values.get("yc")
    S = param_values.get("S")
    focal_length = param_values.get("f")
    wavelength = param_values.get("lambda")
    
    # Validate required params are present
    for name, val in [("xc", xc), ("yc", yc), ("S", S), ("f", focal_length)]:
        if val is None:
            raise ValueError(f"Required parameter '{name}' not found in predicted_names: {predicted_names}")
    
    # Handle lambda: predicted OR fixed from batch
    if wavelength is None:
        if batch is None:
            raise ValueError(
                "Lambda is not predicted but batch is None. "
                "Provide batch['lambda_m'] or batch['lambda_idx'] for fixed wavelength."
            )
        
        if "lambda_m" in batch:
            # Direct wavelength in meters: (B,)
            wavelength = batch["lambda_m"].to(device)
            # Convert to micrometers if needed (forward model uses micrometers)
            if wavelength.max() < 1e-5:  # likely in meters
                wavelength = wavelength * 1e6  # convert m -> um
        elif "lambda_idx" in batch:
            # Index into allowed set
            indices = batch["lambda_idx"].to(device)
            allowed = torch.tensor(ALLOWED_WAVELENGTHS_M, device=device)
            wavelength = allowed[indices] * 1e6  # convert m -> um
        else:
            raise ValueError(
                "Lambda is not predicted and batch does not contain 'lambda_m' or 'lambda_idx'. "
                "Cannot reconstruct phase map."
            )
    
    return xc, yc, S, focal_length, wavelength


def reconstruct_phase_map(
    xc: torch.Tensor,
    yc: torch.Tensor,
    S: torch.Tensor,
    focal_length: torch.Tensor,
    wavelength: torch.Tensor,
    H: int,
    W: int,
) -> torch.Tensor:
    """
    Reconstruct 2-channel phase map from physical parameters.
    
    Args:
        xc, yc, S, focal_length, wavelength: (B,) tensors
        H, W: spatial dimensions
        
    Returns:
        recon_img: (B, 2, H, W) tensor of [cos(phi), sin(phi)]
    """
    B = xc.shape[0]
    device = xc.device
    
    # Create normalized coordinate grids in [-0.5, 0.5]
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-0.5, 0.5, H, device=device),
        torch.linspace(-0.5, 0.5, W, device=device),
        indexing='ij'
    )
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)
    
    # Physical coordinates: use S as scaling "window size"
    X_phys = xc.view(B, 1, 1) + S.view(B, 1, 1) * grid_x
    Y_phys = yc.view(B, 1, 1) + S.view(B, 1, 1) * grid_y
    
    # Compute hyperbolic phase
    phi_unwrapped = compute_hyperbolic_phase(
        X_phys, Y_phys,
        focal_length.view(B, 1, 1),
        wavelength.view(B, 1, 1)
    )
    phi_wrapped = wrap_phase(phi_unwrapped)
    
    # Convert to 2-channel representation: (B, H, W, 2) -> (B, 2, H, W)
    recon_img = get_2channel_representation(phi_wrapped).permute(0, 3, 1, 2)
    
    return recon_img


# =============================================================================
# Base Loss Classes
# =============================================================================

class BaseLoss(nn.Module):
    """Base class for all loss functions with common interface."""
    
    def __init__(
        self,
        predicted_params: list[str],
        param_ranges: dict[str, tuple[float, float]],
    ):
        super().__init__()
        self.predicted_params = predicted_params
        self.num_params = len(predicted_params)
        
        # Validate ranges exist for all predicted params
        for name in predicted_params:
            if name not in param_ranges:
                raise ValueError(
                    f"Missing range for predicted parameter '{name}'. "
                    f"Available ranges: {list(param_ranges.keys())}"
                )
        
        self.param_ranges = param_ranges
        
        # Precompute range magnitudes as buffer
        ranges = torch.tensor([
            param_ranges[name][1] - param_ranges[name][0]
            for name in predicted_params
        ], dtype=torch.float32)
        self.register_buffer("range_magnitudes", ranges)
    
    def _validate_input_dims(self, pred_params: torch.Tensor, true_params: torch.Tensor):
        """Validate input tensor dimensions match config."""
        if pred_params.shape[1] != self.num_params:
            raise ValueError(
                f"pred_params has {pred_params.shape[1]} columns but config specifies "
                f"{self.num_params} predicted params: {self.predicted_params}"
            )
        if true_params.shape[1] != self.num_params:
            raise ValueError(
                f"true_params has {true_params.shape[1]} columns but config specifies "
                f"{self.num_params} predicted params: {self.predicted_params}"
            )


# =============================================================================
# Regression Loss: Unit-Standardized Parameter Loss
# =============================================================================

class UnitStandardizedParamLoss(BaseLoss):
    """
    Unit-standardized parameter regression loss (baseline).
    
    For each predicted parameter i:
        e_i = pred_i - target_i
        range_i = max_i - min_i
        L_i = (e_i^2) * (1 / range_i^2)
    
    Total regression loss = mean over batch of sum over predicted parameters.
    This ensures all parameters contribute equally regardless of their physical scale.
    """
    
    def forward(
        self,
        pred_params: torch.Tensor,
        true_params: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute unit-standardized loss.
        
        Args:
            pred_params: (B, D) predicted parameters
            true_params: (B, D) target parameters
            
        Returns:
            loss: scalar tensor
            metrics: dict with loss values
        """
        self._validate_input_dims(pred_params, true_params)
        
        # Compute per-parameter errors
        errors = pred_params - true_params  # (B, D)
        
        # Scale by inverse range squared
        inv_range_sq = 1.0 / (self.range_magnitudes ** 2 + 1e-8)  # (D,)
        weighted_sq_errors = (errors ** 2) * inv_range_sq  # (B, D)
        
        # Sum over parameters, mean over batch
        loss = weighted_sq_errors.sum(dim=1).mean()
        
        return loss, {
            "loss_regression": loss.item(),
            "total_loss": loss.item(),
            "predicted_params": self.predicted_params,
        }


# =============================================================================
# Regression Loss: Phase Gradient Flow Loss
# =============================================================================

class PhaseGradientFlowLoss(BaseLoss):
    """
    Gradient-flow regression loss (Updated Definition).
    
    Formerly based on spatial phase gradients, this is now a specialized
    parameter-space regression loss.
    
    Definition:
    -----------
    Weighted MSE where each parameter's squared error is scaled by 1 / range.
    (Contrast with UnitStandardizedParamLoss which uses 1 / range^2).
    
    L_i = (pred_i - target_i)^2 * (1 / range_i)
    Total Loss = mean_batch( sum_i(L_i) )
    
    This penalizes errors in larger-range parameters (like xc, yc) significantly
    more heavily relative to their normalized unit interval than standard MSE,
    retaining magnitude sensitivity.
    """
    
    def __init__(
        self,
        predicted_params: list[str],
        param_ranges: dict[str, tuple[float, float]],
        phase_range: float = 2 * math.pi,  # Kept for config compatibility, unused
    ):
        super().__init__(predicted_params, param_ranges)
        # Note: phase_range arg is preserved to avoid breaking existing configs/calls
        # but is no longer used in the calculation.
    
    def forward(
        self,
        pred_params: torch.Tensor,
        true_params: torch.Tensor,
        input_images: torch.Tensor | None = None,
        batch: dict | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute weighted parameter MSE loss.
        
        Args:
            pred_params: (B, D) predicted parameters
            true_params: (B, D) target parameters
            input_images: Unused (kept for interface compatibility)
            batch: Unused
            
        Returns:
            loss: scalar tensor
            metrics: dict with loss values
        """
        self._validate_input_dims(pred_params, true_params)
        
        # Compute per-parameter squared errors
        errors = (pred_params - true_params) ** 2  # (B, D)
        
        # Scale by 1 / range (NOT squared)
        # range_magnitudes is (D,)
        inv_range = 1.0 / (self.range_magnitudes + 1e-8)
        weighted_errors = errors * inv_range  # (B, D)
        
        # Sum over parameters, mean over batch
        loss = weighted_errors.sum(dim=1).mean()
        
        return loss, {
            "loss_regression": loss.item(),
            "loss_grad": loss.item(),  # Aliased for backward compatibility in logs
            "total_loss": loss.item(),
            "predicted_params": self.predicted_params,
        }


# =============================================================================
# Regression Loss: Kendall Uncertainty-Weighted
# =============================================================================

class KendallUnitStandardizedLoss(BaseLoss):
    """
    Kendall uncertainty-weighted regression loss.
    
    Wraps unit-standardized per-parameter terms with learnable log variances.
    For each predicted parameter i, a trainable log variance s_i is maintained.
    
    L = Σ [ exp(-s_i) * L_i_base + s_i ]
    
    where L_i_base is the unit-standardized parameter loss for that parameter.
    
    This allows the model to learn which parameters are more uncertain and
    should be weighted less in the loss.
    
    Note: Kendall applies ONLY to regression loss, NEVER to physics loss.
    """
    
    def __init__(
        self,
        predicted_params: list[str],
        param_ranges: dict[str, tuple[float, float]],
        init_log_var: float = 0.0,
    ):
        super().__init__(predicted_params, param_ranges)
        
        # Trainable log variances: one per predicted parameter
        self.log_vars = nn.Parameter(torch.full((self.num_params,), init_log_var))
    
    def forward(
        self,
        pred_params: torch.Tensor,
        true_params: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute Kendall uncertainty-weighted loss.
        
        Args:
            pred_params: (B, D) predicted parameters
            true_params: (B, D) target parameters
            
        Returns:
            loss: scalar tensor
            metrics: dict with loss values
        """
        self._validate_input_dims(pred_params, true_params)
        
        # Compute per-parameter squared errors
        errors = pred_params - true_params  # (B, D)
        
        # Unit-standardize: scale by inverse range squared
        inv_range_sq = 1.0 / (self.range_magnitudes ** 2 + 1e-8)  # (D,)
        base_losses = (errors ** 2) * inv_range_sq  # (B, D)
        
        # Kendall weighting: exp(-s_i) * L_i + s_i
        precision = torch.exp(-self.log_vars)  # (D,)
        weighted_losses = precision * base_losses + self.log_vars  # (B, D)
        
        # Sum over parameters, mean over batch
        loss = weighted_losses.sum(dim=1).mean()
        
        # Build readable diagnostics
        log_vars_list = self.log_vars.detach().cpu().tolist()
        kendall_diagnostics = {
            f"kendall_logvar_{name}": log_vars_list[i]
            for i, name in enumerate(self.predicted_params)
        }
        
        return loss, {
            "loss_regression": loss.item(),
            "total_loss": loss.item(),
            "predicted_params": self.predicted_params,
            "kendall_log_vars": log_vars_list,
            **kendall_diagnostics,
        }


# =============================================================================
# Regression Loss: Custom Hook
# =============================================================================

class CustomRegressionLoss(BaseLoss):
    """
    Custom regression loss placeholder.
    
    Allows injection of a custom loss function for future experimentation.
    Raises clear error if no callable is provided.
    """
    
    def __init__(
        self,
        predicted_params: list[str],
        param_ranges: dict[str, tuple[float, float]],
        loss_fn: Callable | None = None,
    ):
        super().__init__(predicted_params, param_ranges)
        
        if loss_fn is None:
            raise ValueError(
                "CustomRegressionLoss requires a 'loss_fn' callable. "
                "Signature: loss_fn(pred_params, true_params, range_magnitudes) -> Tensor"
            )
        
        self.loss_fn = loss_fn
    
    def forward(
        self,
        pred_params: torch.Tensor,
        true_params: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute custom loss.
        
        Args:
            pred_params: (B, D) predicted parameters
            true_params: (B, D) target parameters
            
        Returns:
            loss: scalar tensor
            metrics: dict with loss values
        """
        self._validate_input_dims(pred_params, true_params)
        
        loss = self.loss_fn(pred_params, true_params, self.range_magnitudes)
        
        return loss, {
            "loss_regression": loss.item(),
            "total_loss": loss.item(),
            "predicted_params": self.predicted_params,
        }


# =============================================================================
# Regression Loss: Coordinate Bias Loss (User Requested)
# =============================================================================

class CoordinateLoss(BaseLoss):
    """
    Coordinate-biased regression loss.
    
    Hard weights applied to specific parameters to prioritize their learning.
    
    Weights:
        xc, yc, S : 5.0
        f, lambda : 1.0 (or anything else)
    
    This is effectively a weighted UnitStandardizedParamLoss.
    L_i = (pred_i - target_i)^2 * (weight_i / range_i)^2
    """
    
    def __init__(
        self,
        predicted_params: list[str],
        param_ranges: dict[str, tuple[float, float]],
    ):
        super().__init__(predicted_params, param_ranges)
        
        # Define weights
        # Default all to 1.0
        weights = torch.ones(self.num_params)
        
        # Apply 5.0 to prioritized params
        priority_params = ["xc", "yc", "S"]
        
        for i, name in enumerate(predicted_params):
            if name in priority_params:
                weights[i] = 5.0
        
        self.register_buffer("weights", weights)
    
    def forward(
        self,
        pred_params: torch.Tensor,
        true_params: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute coordinate-weighted loss (GradFlow variant).
        
        Formula:
        L = Sum( (pred - target)^2 * Weight / Range )
        """
        self._validate_input_dims(pred_params, true_params)
        
        # Squared errors
        squared_errors = (pred_params - true_params) ** 2
        
        # Scaling: 1 / Range (GradFlow style) instead of 1 / Range^2
        inv_range = 1.0 / (self.range_magnitudes + 1e-8)  # (D,)
        
        # Base GradFlow term per parameter
        base_term = squared_errors * inv_range
        
        # Apply prioritization weights
        # weights: (D,) e.g. [5, 5, 5, 1, 1]
        weighted_term = base_term * self.weights
        
        # Sum over params, mean over batch
        loss = weighted_term.sum(dim=1).mean()
        
        return loss, {
            "loss_regression": loss.item(),
            "loss_coordinate": loss.item(),
            "total_loss": loss.item(),
            "predicted_params": self.predicted_params,
        }


# =============================================================================
# Physics Loss: Phase Map MSE
# =============================================================================

class PhysicsPhaseMapMSELoss(nn.Module):
    """
    Physics loss: MSE between reconstructed 2-channel phase map and input.
    
    This loss enforces physical consistency by requiring the predicted
    parameters to reproduce the input phase map when passed through
    the forward model.
    
    Note: This loss is ALWAYS additive to regression loss and is NEVER
    weighted by Kendall uncertainty.
    """
    
    def __init__(self, predicted_params: list[str]):
        super().__init__()
        self.predicted_params = predicted_params
    
    def forward(
        self,
        pred_params: torch.Tensor,
        input_images: torch.Tensor,
        batch: dict | None = None,
    ) -> torch.Tensor:
        """
        Compute physics loss.
        
        Args:
            pred_params: (B, D) predicted parameters
            input_images: (B, 2, H, W) input phase maps
            batch: optional dict with fixed lambda
            
        Returns:
            loss: scalar tensor
        """
        if input_images is None:
            raise ValueError(
                "PhysicsPhaseMapMSELoss requires input_images but received None."
            )
        
        B, C, H, W = input_images.shape
        
        # Reconstruct phase map from predicted params
        xc, yc, S, f, wl = build_full_params_for_reconstruction(
            pred_params, self.predicted_params, batch
        )
        recon_img = reconstruct_phase_map(xc, yc, S, f, wl, H, W)
        
        # Pixelwise MSE
        loss = F.mse_loss(recon_img, input_images)
        
        return loss


# =============================================================================
# Composite Loss: Combines Regression + Physics with Scheduling
# =============================================================================

class CompositeLoss(nn.Module):
    """
    Composite loss combining regression and physics losses.
    
    Structure:
        total = regression_loss + physics_weight * physics_loss
    
    Physics loss is:
    - Optional (can be disabled)
    - PINN-style scheduled (only active after physics_start_epoch)
    - NEVER weighted by Kendall uncertainty
    
    This is the main loss class to use for training.
    """
    
    def __init__(
        self,
        regression_loss: BaseLoss,
        physics_enabled: bool = False,
        physics_weight: float = 0.1,
        physics_start_epoch: int = 100,
        predicted_params: list[str] | None = None,
    ):
        super().__init__()
        
        self.regression_loss = regression_loss
        self.physics_enabled = physics_enabled
        self.physics_weight = physics_weight
        self.physics_start_epoch = physics_start_epoch
        
        # Get predicted params from regression loss if not provided
        self.predicted_params = predicted_params or regression_loss.predicted_params
        
        # Physics loss module (only created if enabled)
        if physics_enabled:
            self.physics_loss = PhysicsPhaseMapMSELoss(self.predicted_params)
        else:
            self.physics_loss = None
    
    def forward(
        self,
        pred_params: torch.Tensor,
        true_params: torch.Tensor,
        input_images: torch.Tensor | None = None,
        batch: dict | None = None,
        epoch: int | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute composite loss.
        
        Args:
            pred_params: (B, D) predicted parameters
            true_params: (B, D) target parameters
            input_images: (B, 2, H, W) for physics loss
            batch: optional dict with fixed lambda
            epoch: current epoch for physics scheduling
            
        Returns:
            total_loss: scalar tensor
            metrics: dict with all loss values
        """
        # Compute regression loss
        reg_loss, metrics = self.regression_loss(
            pred_params=pred_params,
            true_params=true_params,
            input_images=input_images,
            batch=batch,
        )
        
        total_loss = reg_loss
        
        # Physics loss (if enabled and past start epoch)
        physics_active = False
        if self.physics_enabled and self.physics_loss is not None:
            # Check scheduling gate
            if epoch is None or epoch >= self.physics_start_epoch:
                if input_images is None:
                    raise ValueError(
                        "Physics loss is enabled but input_images is None. "
                        "Provide input_images or disable physics loss."
                    )
                
                phys_loss = self.physics_loss(pred_params, input_images, batch)
                total_loss = reg_loss + self.physics_weight * phys_loss
                
                metrics["loss_physics"] = phys_loss.item()
                metrics["physics_weight"] = self.physics_weight
                physics_active = True
        
        metrics["total_loss"] = total_loss.item()
        metrics["physics_active"] = physics_active
        
        return total_loss, metrics


# =============================================================================
# Factory Function
# =============================================================================

def build_loss(loss_cfg: dict, data_cfg: dict) -> nn.Module:
    """
    Factory function to build loss module from config.
    
    Args:
        loss_cfg: Loss configuration dictionary with keys:
            - mode: str ("unit_standardized", "gradient_flow", "kendall", "custom")
            - predicted_params: list[str] (e.g., ["xc", "yc", "S", "f"])
            - physics_enabled: bool (default False)
            - physics_weight: float (default 0.1)
            - physics_start_epoch: int (default 100)
            - phase_range: float (default 2*pi, for gradient_flow mode)
            - init_log_var: float (default 0.0, for kendall mode)
            - custom_loss_fn: Callable (required for custom mode)
            
        data_cfg: Data configuration with parameter ranges:
            - xc_range: [min, max]
            - yc_range: [min, max]
            - S_range: [min, max]
            - focal_length_range: [min, max]
            - wavelength_range: [min, max]
            
    Returns:
        CompositeLoss module ready for training
        
    Raises:
        ValueError: if required config keys are missing or invalid
    """
    # Extract parameter ranges from data config
    param_ranges = extract_param_ranges(data_cfg)
    
    # Get predicted params (default to 5-param if not specified)
    predicted_params = loss_cfg.get("predicted_params", ["xc", "yc", "S", "f", "lambda"])
    
    # Validate predicted params
    for name in predicted_params:
        if name not in PARAM_ORDER:
            raise ValueError(
                f"Unknown predicted parameter '{name}'. "
                f"Valid parameters: {PARAM_ORDER}"
            )
    
    # Get loss mode
    mode = loss_cfg.get("mode", "unit_standardized")
    
    # Build regression loss based on mode
    if mode == "unit_standardized":
        regression_loss = UnitStandardizedParamLoss(
            predicted_params=predicted_params,
            param_ranges=param_ranges,
        )
    
    elif mode == "coordinate" or mode == "coordinate_loss":
        regression_loss = CoordinateLoss(
            predicted_params=predicted_params,
            param_ranges=param_ranges,
        )

    
    elif mode == "gradient_flow":
        phase_range = loss_cfg.get("phase_range", 2 * math.pi)
        regression_loss = PhaseGradientFlowLoss(
            predicted_params=predicted_params,
            param_ranges=param_ranges,
            phase_range=phase_range,
        )
    
    elif mode == "kendall":
        init_log_var = loss_cfg.get("init_log_var", 0.0)
        regression_loss = KendallUnitStandardizedLoss(
            predicted_params=predicted_params,
            param_ranges=param_ranges,
            init_log_var=init_log_var,
        )
    
    elif mode == "custom":
        custom_fn = loss_cfg.get("custom_loss_fn")
        regression_loss = CustomRegressionLoss(
            predicted_params=predicted_params,
            param_ranges=param_ranges,
            loss_fn=custom_fn,
        )
    
    else:
        raise ValueError(
            f"Unknown loss mode '{mode}'. "
            f"Valid modes: unit_standardized, gradient_flow, kendall, custom"
        )
    
    # Build composite loss with optional physics
    physics_enabled = loss_cfg.get("physics_enabled", False)
    physics_weight = loss_cfg.get("physics_weight", 0.1)
    physics_start_epoch = loss_cfg.get("physics_start_epoch", 100)
    
    composite_loss = CompositeLoss(
        regression_loss=regression_loss,
        physics_enabled=physics_enabled,
        physics_weight=physics_weight,
        physics_start_epoch=physics_start_epoch,
        predicted_params=predicted_params,
    )
    
    return composite_loss


# =============================================================================
# Legacy Compatibility: Deprecated Class Guard
# =============================================================================

class Naive5ParamMSELoss(nn.Module):
    """
    [DEPRECATED/REMOVED]
    
    This loss function was a trap - it compared physical predictions to 
    normalized targets, causing silent training failure.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise RuntimeError(
            "CRITICAL: Naive5ParamMSELoss has been removed for safety.\n"
            "Reason: It causes silent training failure by comparing Physical Preds vs Normalized Targets.\n"
            "Use build_loss() factory with mode='unit_standardized' instead."
        )
    
    def forward(self, *args, **kwargs):
        raise RuntimeError("Naive5ParamMSELoss is disabled.")
