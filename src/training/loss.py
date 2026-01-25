import torch
import torch.nn as nn
import numpy as np
from src.inversion.forward_model import compute_hyperbolic_phase, wrap_phase, get_2channel_representation


class RawPhysicsLoss(nn.Module):
    """
    Loss function for models with HybridScaledOutput (outputs raw physical values).
    No normalization needed - directly compares raw predictions to raw targets.
    
    Components:
        1. Weighted MSE on raw parameters (scale-aware weighting)
        2. Physics reconstruction loss
    """
    def __init__(self, 
                 lambda_param=1.0, 
                 lambda_physics=0.5,
                 param_weights=None):
        super().__init__()
        self.lambda_param = lambda_param
        self.lambda_physics = lambda_physics
        self.mse = nn.MSELoss()
        
        # Weights inversely proportional to typical magnitude squared
        # This makes each param contribute equally to loss
        if param_weights is None:
            # xc~500, yc~500, S~20, wavelength~0.5, focal_length~50
            param_weights = [1/(500**2), 1/(500**2), 1/(20**2), 1/(0.15**2), 1/(45**2)]
        self.register_buffer('weights', torch.tensor(param_weights, dtype=torch.float32))
    
    def forward(self, pred_params, true_params, input_images):
        """
        Args:
            pred_params: (B, 5) raw physical values [xc, yc, S, wavelength, focal_length]
            true_params: (B, 5) raw physical values from dataset
            input_images: (B, 2, H, W) input phase maps
        """
        # 1. Weighted Parameter Loss (scale-aware)
        diff = pred_params - true_params  # (B, 5)
        weighted_sq_diff = self.weights * (diff ** 2)
        loss_param = weighted_sq_diff.mean()
        
        # 2. Physics Reconstruction Loss
        B, C, H, W = input_images.shape
        device = input_images.device
        
        xc = pred_params[:, 0]
        yc = pred_params[:, 1]
        S = pred_params[:, 2]  # Scaling = window size
        wavelength = pred_params[:, 3]
        focal_length = pred_params[:, 4]
        
        # Create coordinate grids
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-0.5, 0.5, H, device=device),
            torch.linspace(-0.5, 0.5, W, device=device),
            indexing='ij'
        )
        grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
        grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)
        
        # Use S as window size for coordinate grids
        X_phys = xc.view(B, 1, 1) + S.view(B, 1, 1) * grid_x
        Y_phys = yc.view(B, 1, 1) + S.view(B, 1, 1) * grid_y
        
        # Broadcast params to (B, 1, 1) for physics formula
        focal_length = focal_length.view(B, 1, 1)
        wavelength = wavelength.view(B, 1, 1)
        
        phi_unwrapped = compute_hyperbolic_phase(X_phys, Y_phys, focal_length, wavelength)
        phi_wrapped = wrap_phase(phi_unwrapped)
        reconstructed = get_2channel_representation(phi_wrapped).permute(0, 3, 1, 2)
        
        loss_physics = self.mse(reconstructed, input_images)
        
        total_loss = (self.lambda_param * loss_param) + (self.lambda_physics * loss_physics)
        
        return total_loss, {
            "loss_param": loss_param.item(),
            "loss_physics": loss_physics.item(),
            "total_loss": total_loss.item()
        }


class AdaptivePhysicsLoss(nn.Module):
    """
    Kendall's Aleatoric Uncertainty Loss.
    Learns weights dynamically: L = (1/2sigma^2) * MSE + log(sigma)
    """
    def __init__(self, 
                 num_params=5,
                 lambda_param=1.0, 
                 lambda_physics=0.5):
        super().__init__()
        self.lambda_param = lambda_param
        self.lambda_physics = lambda_physics
        self.mse = nn.MSELoss()
        
        # Learnable log variances (one per parameter)
        self.log_vars = nn.Parameter(torch.zeros(num_params))
        
    def forward(self, pred_params, true_params, input_images):
        # 1. Adaptive Parameter Loss
        diff = (pred_params - true_params) ** 2
        precision = torch.exp(-self.log_vars)
        
        # Kendall Loss: sum( precision * diff + log_vars )
        loss_param = (precision * diff + self.log_vars).sum(dim=1).mean()
        
        # 2. Physics Reconstruction Loss
        B, C, H, W = input_images.shape
        device = input_images.device
        
        xc = pred_params[:, 0]
        yc = pred_params[:, 1]
        S = pred_params[:, 2]  # Scaling = window size
        wavelength = pred_params[:, 3]
        focal_length = pred_params[:, 4]
        
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-0.5, 0.5, H, device=device),
            torch.linspace(-0.5, 0.5, W, device=device),
            indexing='ij'
        )
        grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
        grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)
        
        # Use S as window size
        X_phys = xc.view(B, 1, 1) + S.view(B, 1, 1) * grid_x
        Y_phys = yc.view(B, 1, 1) + S.view(B, 1, 1) * grid_y
        
        focal_length = focal_length.view(B, 1, 1)
        wavelength = wavelength.view(B, 1, 1)
        
        phi_unwrapped = compute_hyperbolic_phase(X_phys, Y_phys, focal_length, wavelength)
        phi_wrapped = wrap_phase(phi_unwrapped)
        reconstructed = get_2channel_representation(phi_wrapped).permute(0, 3, 1, 2)
        
        loss_physics = self.mse(reconstructed, input_images)
        
        total_loss = (self.lambda_param * loss_param) + (self.lambda_physics * loss_physics)
        
        return total_loss, {
            "loss_param": loss_param.item(),
            "loss_physics": loss_physics.item(),
            "total_loss": total_loss.item(),
            "sigma_xc": torch.exp(self.log_vars[0]).item(),
            "sigma_S": torch.exp(self.log_vars[2]).item()
        }


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
            pred_params: (B, 5) [xc, yc, S, wl, fl] (Normalized)
            true_params: (B, 5) [xc, yc, S, wl, fl] (Real Units)
            input_images: (B, C, H, W)
        """
        # 1. Parameter Component
        loss_param, _ = self.param_loss(pred_params, true_params)

        # 2. Physics Component (Differentiable Reconstruction)
        if self.normalizer:
            pred_params_phys = self.normalizer.denormalize_tensor(pred_params)
        else:
            pred_params_phys = pred_params

        B, C, H, W = input_images.shape
        device = input_images.device

        xc = pred_params_phys[:, 0]
        yc = pred_params_phys[:, 1]
        S = pred_params_phys[:, 2]  # Scaling = window size

        # Handle optional dynamic focal_length/wavelength if predicted
        if pred_params_phys.shape[1] >= 5:
            wavelength = pred_params_phys[:, 3]
            focal_length = pred_params_phys[:, 4]
        else:
            focal_length = torch.tensor(self.fixed_focal_length, device=device).expand(B)
            wavelength = torch.tensor(self.fixed_wavelength, device=device).expand(B)

        # Create coordinate grids
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-0.5, 0.5, H, device=device),
            torch.linspace(-0.5, 0.5, W, device=device),
            indexing='ij'
        )
        grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
        grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)

        # Use S as window size
        X_phys = xc.view(B, 1, 1) + S.view(B, 1, 1) * grid_x
        Y_phys = yc.view(B, 1, 1) + S.view(B, 1, 1) * grid_y

        focal_length = focal_length.view(B, 1, 1)
        wavelength = wavelength.view(B, 1, 1)

        phi_unwrapped = compute_hyperbolic_phase(X_phys, Y_phys, focal_length, wavelength)
        phi_wrapped = wrap_phase(phi_unwrapped)
        reconstructed_image = get_2channel_representation(phi_wrapped).permute(0, 3, 1, 2)

        loss_physics = self.mse(reconstructed_image, input_images)

        total_loss = (self.lambda_param * loss_param) + (self.lambda_physics * loss_physics)
        
        return total_loss, {
            "loss_param": loss_param.item(), 
            "loss_physics": loss_physics.item(),
            "total_loss": total_loss.item()
        }


class AuxiliaryPhysicsLoss(nn.Module):
    """
    Enhanced physics loss with auxiliary task for fringe density.
    
    Auxiliary Tasks:
    1. Fringe Density Loss: λ/f ratio determines fringe spacing - match FFT peak frequency
    """
    def __init__(self, 
                 lambda_param=1.0, 
                 lambda_physics=0.5,
                 lambda_fringe=0.1,
                 param_weights=[1.0, 1.0, 5.0, 20.0, 20.0],
                 normalizer=None):
        super().__init__()
        self.lambda_param = lambda_param
        self.lambda_physics = lambda_physics
        self.lambda_fringe = lambda_fringe
        self.param_loss = WeightedStandardizedLoss(weights=param_weights, normalizer=normalizer)
        self.normalizer = normalizer
        self.mse = nn.MSELoss()

    def _compute_fringe_density_loss(self, input_images, pred_wavelength, pred_focal_length):
        """
        Compute fringe density loss using FFT.
        Fringe period ∝ λ * f / r, so higher λ*f means lower frequency.
        """
        B, C, H, W = input_images.shape
        
        phase_channel = input_images[:, 0, :, :]
        
        fft = torch.fft.fft2(phase_channel)
        fft_mag = torch.abs(fft)
        fft_shifted = torch.fft.fftshift(fft_mag, dim=(-2, -1))
        
        center_h, center_w = H // 2, W // 2
        
        y_coords = torch.arange(H, device=input_images.device).float() - center_h
        x_coords = torch.arange(W, device=input_images.device).float() - center_w
        Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
        R = torch.sqrt(X**2 + Y**2)
        
        mask = (R > 2) & (R < min(H, W) // 4)
        
        weighted_sum = (fft_shifted * R.unsqueeze(0) * mask.unsqueeze(0)).sum(dim=(-2, -1))
        total_weight = (fft_shifted * mask.unsqueeze(0)).sum(dim=(-2, -1)) + 1e-6
        
        dominant_freq = weighted_sum / total_weight
        
        lambda_f_product = pred_wavelength * pred_focal_length
        theoretical_freq_scale = 1.0 / (lambda_f_product + 1e-6)
        
        dominant_freq_norm = dominant_freq / (dominant_freq.mean() + 1e-6)
        theoretical_freq_norm = theoretical_freq_scale / (theoretical_freq_scale.mean() + 1e-6)
        
        loss = self.mse(dominant_freq_norm, theoretical_freq_norm)
        
        return loss

    def forward(self, pred_params, true_params, input_images):
        """
        Compute total loss with auxiliary components.
        """
        # 1. Parameter Loss
        loss_param, _ = self.param_loss(pred_params, true_params)
        
        # 2. Physics Reconstruction Loss
        if self.normalizer:
            pred_params_phys = self.normalizer.denormalize_tensor(pred_params)
        else:
            pred_params_phys = pred_params
            
        B, C, H, W = input_images.shape
        device = input_images.device
        
        xc = pred_params_phys[:, 0]
        yc = pred_params_phys[:, 1]
        S = pred_params_phys[:, 2]  # Scaling = window size
        wavelength = pred_params_phys[:, 3]
        focal_length = pred_params_phys[:, 4]
        
        # Create coordinate grids
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-0.5, 0.5, H, device=device),
            torch.linspace(-0.5, 0.5, W, device=device),
            indexing='ij'
        )
        grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
        grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)
        
        # Use S as window size
        X_phys = xc.view(B, 1, 1) + S.view(B, 1, 1) * grid_x
        Y_phys = yc.view(B, 1, 1) + S.view(B, 1, 1) * grid_y
        
        phi_unwrapped = compute_hyperbolic_phase(
            X_phys, Y_phys, 
            focal_length.view(B, 1, 1), 
            wavelength.view(B, 1, 1)
        )
        phi_wrapped = wrap_phase(phi_unwrapped)
        reconstructed = get_2channel_representation(phi_wrapped).permute(0, 3, 1, 2)
        
        loss_physics = self.mse(reconstructed, input_images)
        
        # 3. Auxiliary: Fringe Density Loss (for λ/f)
        loss_fringe = self._compute_fringe_density_loss(input_images, wavelength, focal_length)
        
        # Total
        total_loss = (
            self.lambda_param * loss_param +
            self.lambda_physics * loss_physics +
            self.lambda_fringe * loss_fringe
        )
        
        return total_loss, {
            "loss_param": loss_param.item(),
            "loss_physics": loss_physics.item(),
            "loss_fringe": loss_fringe.item(),
            "total_loss": total_loss.item()
        }

