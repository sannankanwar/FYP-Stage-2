import torch
import torch.nn as nn
import numpy as np
from src.inversion.forward_model import compute_hyperbolic_phase, wrap_phase, get_2channel_representation

class Naive5ParamMSELoss(nn.Module):
    """
    Simple MSE Loss on all 5 parameters.
    No weighting, assumes raw or uniformly standardized inputs.
    """
    def __init__(self, normalizer=None):
        super().__init__()
        self.mse = nn.MSELoss()
        self.normalizer = normalizer # used for denormalization if needed for logging only

    def forward(self, pred_params, true_params, input_images=None):
        # input_images unused but kept for interface consistency
        
        # If normalizer is present, it means Trainer is standardizing targets.
        # But for 'Naive' we assume we just minimize the difference directly.
        
        if self.normalizer:
            # Depending on implementation, 'Naive' might implies on *Raw* params.
            # But if the model outputs normalized params, calculating MSE on normalized 
            # params is standard.
            # Let's assume Naive means "Uniform Weighting on Normalized Data".
            true_params_norm = self.normalizer.normalize_tensor(true_params)
            loss = self.mse(pred_params, true_params_norm)
        else:
            loss = self.mse(pred_params, true_params)
            
        return loss, {"total_loss": loss.item()}


class WeightedStandardizedLoss(nn.Module):
    """
    Weighted MSE on Standardized Parameters.
    Allows prioritizing specific parameters (e.g. wavelength) via weights.
    """
    def __init__(self, weights=[1.0, 1.0, 1.0, 10.0, 10.0], normalizer=None):
        """
        weights: list of 5 floats for [xc, yc, fov, wavelength, focal_length]
        """
        super().__init__()
        self.weights = torch.tensor(weights)
        self.normalizer = normalizer

    def forward(self, pred_params, true_params, input_images=None):
        if self.weights.device != pred_params.device:
            self.weights = self.weights.to(pred_params.device)
            
        if self.normalizer:
            true_params_norm = self.normalizer.normalize_tensor(true_params)
            diff = (pred_params - true_params_norm)
        else:
            # If standardizing is off but we want weights, usually means we weight raw values?
            # But raw values have vastly different scales. 
            # This loss is intended for standardized outputs.
            diff = (pred_params - true_params)

        # Weighted MSE: Mean( Weights * (Diff)^2 )
        # Broadcast weights (5,) to (B, 5)
        weighted_sq_diff = self.weights * (diff ** 2)
        loss = torch.mean(weighted_sq_diff)
        
        return loss, {"total_loss": loss.item()}


class WeightedPhysicsLoss(nn.Module):
    """
    Hybrid loss: Weighted Parameter Loss + Physics Reconstruction Loss.
    """
    def __init__(self, 
                 lambda_param=1.0, 
                 lambda_physics=0.1, 
                 param_weights=[1.0, 1.0, 1.0, 10.0, 10.0],
                 fixed_focal_length=100.0, 
                 fixed_wavelength=0.532, 
                 normalizer=None):
        super().__init__()
        self.lambda_param = lambda_param
        self.lambda_physics = lambda_physics
        self.param_loss = WeightedStandardizedLoss(weights=param_weights, normalizer=normalizer)
        self.normalizer = normalizer
        
        self.fixed_focal_length = fixed_focal_length
        self.fixed_wavelength = fixed_wavelength
        self.mse = nn.MSELoss()

    def forward(self, pred_params, true_params, input_images):
        """
        Args:
            pred_params: (B, 5) [xc, yc, fov, wl, fl] (Normalized)
            true_params: (B, 5) [xc, yc, fov, wl, fl] (Real Units)
            input_images: (B, C, H, W)
        """
        # 1. Parameter Component
        loss_param, _ = self.param_loss(pred_params, true_params)

        # 2. Physics Component (Differentiable Reconstruction)
        # We need REAL physical units for the forward model
        if self.normalizer:
            pred_params_phys = self.normalizer.denormalize_tensor(pred_params)
        else:
            pred_params_phys = pred_params

        B, C, H, W = input_images.shape
        device = input_images.device

        xc = pred_params_phys[:, 0]
        yc = pred_params_phys[:, 1]
        fov = pred_params_phys[:, 2]

        # Handle optional dynamic focal_length/wavelength if predicted
        if pred_params_phys.shape[1] >= 5:
            wavelength = pred_params_phys[:, 3]
            focal_length = pred_params_phys[:, 4]
        else:
            # Expand fixed values to match batch size for proper broadcasting
            focal_length = torch.tensor(self.fixed_focal_length, device=device).expand(B)
            wavelength = torch.tensor(self.fixed_wavelength, device=device).expand(B)

        # Create coordinate grids
        # Note: We need to do this in a differentiable way relative to xc, yc, fov
        # Construct grid manually to keep gradients flow clear
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-0.5, 0.5, H, device=device),
            torch.linspace(-0.5, 0.5, W, device=device),
            indexing='ij'
        ) # (H, W)
        
        grid_x = grid_x.unsqueeze(0).expand(B, -1, -1) # (B, H, W)
        grid_y = grid_y.unsqueeze(0).expand(B, -1, -1) # (B, H, W)

        # Physical coordinates
        # X = xc + fov * grid_x_normalized
        X_phys = xc.view(B, 1, 1) + fov.view(B, 1, 1) * grid_x
        Y_phys = yc.view(B, 1, 1) + fov.view(B, 1, 1) * grid_y

        # Broadcast focal_length and wavelength to (B, 1, 1) for physics formula
        focal_length = focal_length.view(B, 1, 1)
        wavelength = wavelength.view(B, 1, 1)

        # Forward Model
        phi_unwrapped = compute_hyperbolic_phase(X_phys, Y_phys, focal_length, wavelength)
        phi_wrapped = wrap_phase(phi_unwrapped)
        reconstructed_image = get_2channel_representation(phi_wrapped) # (B, H, W, 2)
        
        # Permute to (B, C, H, W) to match input
        reconstructed_image = reconstructed_image.permute(0, 3, 1, 2)

        loss_physics = self.mse(reconstructed_image, input_images)

        total_loss = (self.lambda_param * loss_param) + (self.lambda_physics * loss_physics)
        
        return total_loss, {
            "loss_param": loss_param.item(), 
            "loss_physics": loss_physics.item(),
            "total_loss": total_loss.item()
        }
