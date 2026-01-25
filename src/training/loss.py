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
                 window_size=100.0,
                 # Default weights scaled inversely with parameter magnitudes
                 param_weights=None):
        super().__init__()
        self.lambda_param = lambda_param
        self.lambda_physics = lambda_physics
        self.window_size = window_size
        self.mse = nn.MSELoss()
        
        # Weights inversely proportional to typical magnitude squared
        # This makes each param contribute equally to loss
        if param_weights is None:
            # xc~500, yc~500, fov~10, wavelength~0.1, focal_length~50
            param_weights = [1/(500**2), 1/(500**2), 1/(10**2), 1/(0.15**2), 1/(45**2)]
        self.register_buffer('weights', torch.tensor(param_weights, dtype=torch.float32))
    
    def forward(self, pred_params, true_params, input_images):
        """
        Args:
            pred_params: (B, 5) raw physical values from model
            true_params: (B, 5) raw physical values from dataset
            input_images: (B, 2, H, W) input phase maps
        """
        # 1. Weighted Parameter Loss (scale-aware)
        diff = pred_params - true_params  # (B, 5)
        weighted_sq_diff = self.weights * (diff ** 2)  # Scale each param appropriately
        loss_param = weighted_sq_diff.mean()
        
        # 2. Physics Reconstruction Loss
        B, C, H, W = input_images.shape
        device = input_images.device
        
        xc = pred_params[:, 0]
        yc = pred_params[:, 1]
        fov = pred_params[:, 2]
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
        
        X_phys = xc.view(B, 1, 1) + self.window_size * grid_x
        Y_phys = yc.view(B, 1, 1) + self.window_size * grid_y
        
        # Broadcast params to (B, 1, 1) for physics formula
        focal_length = focal_length.view(B, 1, 1)
        wavelength = wavelength.view(B, 1, 1)
        fov = fov.view(B, 1, 1)
        
        phi_unwrapped = compute_hyperbolic_phase(X_phys, Y_phys, focal_length, wavelength, theta=fov)
        phi_wrapped = wrap_phase(phi_unwrapped)
        reconstructed = get_2channel_representation(phi_wrapped).permute(0, 3, 1, 2)
        
        loss_physics = self.mse(reconstructed, input_images)
        
        total_loss = (self.lambda_param * loss_param) + (self.lambda_physics * loss_physics)
        
        return total_loss, {
            "loss_param": loss_param.item(),
            "loss_physics": loss_physics.item(),
            "total_loss": total_loss.item()
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
                 window_size=100.0,
                 normalizer=None):
        super().__init__()
        self.lambda_param = lambda_param
        self.lambda_physics = lambda_physics
        self.param_loss = WeightedStandardizedLoss(weights=param_weights, normalizer=normalizer)
        self.normalizer = normalizer
        
        self.fixed_focal_length = fixed_focal_length
        self.fixed_wavelength = fixed_wavelength
        self.window_size = window_size
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
        # X = xc + window_size * grid_x_normalized
        # Window is centered at xc, yc
        X_phys = xc.view(B, 1, 1) + self.window_size * grid_x
        Y_phys = yc.view(B, 1, 1) + self.window_size * grid_y

        # Broadcast focal_length and wavelength to (B, 1, 1) for physics formula
        focal_length = focal_length.view(B, 1, 1)
        wavelength = wavelength.view(B, 1, 1)
        fov = fov.view(B, 1, 1)  # Broadcast fov for grid compatibility

        # Forward Model
        phi_unwrapped = compute_hyperbolic_phase(X_phys, Y_phys, focal_length, wavelength, theta=fov)
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


class AuxiliaryPhysicsLoss(nn.Module):
    """
    Enhanced physics loss with auxiliary tasks to disambiguate degenerate parameters.
    
    Auxiliary Tasks:
    1. Gradient Direction Loss: fov creates horizontal tilt - enforce gradient angle match
    2. Fringe Density Loss: λ/f ratio determines fringe spacing - match FFT peak frequency
    """
    def __init__(self, 
                 lambda_param=1.0, 
                 lambda_physics=0.5,
                 lambda_gradient=0.1,
                 lambda_fringe=0.1,
                 param_weights=[1.0, 1.0, 5.0, 20.0, 20.0],
                 window_size=100.0,
                 normalizer=None):
        super().__init__()
        self.lambda_param = lambda_param
        self.lambda_physics = lambda_physics
        self.lambda_gradient = lambda_gradient
        self.lambda_fringe = lambda_fringe
        self.param_loss = WeightedStandardizedLoss(weights=param_weights, normalizer=normalizer)
        self.normalizer = normalizer
        self.window_size = window_size
        self.mse = nn.MSELoss()
        
        # Sobel kernels for gradient computation
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 4.0)
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 4.0)

    def _compute_gradient_loss(self, input_images, pred_fov_rad):
        """
        Compute gradient direction loss.
        fov creates a horizontal gradient in the phase - gradient angle should correlate with sin(fov).
        """
        B, C, H, W = input_images.shape
        
        # Use the cos channel (channel 0) for gradient computation
        phase_channel = input_images[:, 0:1, :, :]  # (B, 1, H, W)
        
        # Compute gradients using Sobel
        grad_x = torch.nn.functional.conv2d(phase_channel, self.sobel_x, padding=1)
        grad_y = torch.nn.functional.conv2d(phase_channel, self.sobel_y, padding=1)
        
        # Average gradient direction across the image (central region to avoid edges)
        h_start, h_end = H // 4, 3 * H // 4
        w_start, w_end = W // 4, 3 * W // 4
        
        mean_grad_x = grad_x[:, :, h_start:h_end, w_start:w_end].mean(dim=(1, 2, 3))  # (B,)
        mean_grad_y = grad_y[:, :, h_start:h_end, w_start:w_end].mean(dim=(1, 2, 3))  # (B,)
        
        # fov contribution to horizontal gradient: ∂φ/∂x ∝ k0 * sin(θ)
        # Since fov adds k0 * x * sin(θ), gradient in x is k0 * sin(θ)
        # Normalize to get direction indicator
        predicted_tilt = torch.sin(pred_fov_rad)  # (B,)
        
        # The gradient should be proportional to sin(fov)
        # Normalize both to correlate
        eps = 1e-6
        grad_norm = torch.sqrt(mean_grad_x**2 + mean_grad_y**2 + eps)
        normalized_grad_x = mean_grad_x / grad_norm
        
        # Loss: gradient direction should match fov direction
        loss = self.mse(normalized_grad_x, predicted_tilt / (torch.abs(predicted_tilt) + eps))
        
        return loss

    def _compute_fringe_density_loss(self, input_images, pred_wavelength, pred_focal_length):
        """
        Compute fringe density loss using FFT.
        Fringe period ∝ λ * f / r, so higher λ*f means lower frequency.
        """
        B, C, H, W = input_images.shape
        
        # Use cos channel
        phase_channel = input_images[:, 0, :, :]  # (B, H, W)
        
        # Compute 2D FFT magnitude
        fft = torch.fft.fft2(phase_channel)
        fft_mag = torch.abs(fft)
        
        # Shift zero frequency to center
        fft_shifted = torch.fft.fftshift(fft_mag, dim=(-2, -1))
        
        # Find radial average to get dominant frequency
        center_h, center_w = H // 2, W // 2
        
        # Create radial coordinate grid
        y_coords = torch.arange(H, device=input_images.device).float() - center_h
        x_coords = torch.arange(W, device=input_images.device).float() - center_w
        Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
        R = torch.sqrt(X**2 + Y**2)
        
        # Weighted centroid of frequency (excluding DC)
        mask = (R > 2) & (R < min(H, W) // 4)  # Exclude DC and high freq noise
        
        weighted_sum = (fft_shifted * R.unsqueeze(0) * mask.unsqueeze(0)).sum(dim=(-2, -1))
        total_weight = (fft_shifted * mask.unsqueeze(0)).sum(dim=(-2, -1)) + 1e-6
        
        dominant_freq = weighted_sum / total_weight  # (B,)
        
        # Theoretical: freq ∝ 1 / (λ * f)  
        # Normalize both to get relative comparison
        lambda_f_product = pred_wavelength * pred_focal_length
        theoretical_freq_scale = 1.0 / (lambda_f_product + 1e-6)
        
        # Normalize for comparison
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
        fov = pred_params_phys[:, 2]
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
        
        X_phys = xc.view(B, 1, 1) + self.window_size * grid_x
        Y_phys = yc.view(B, 1, 1) + self.window_size * grid_y
        
        phi_unwrapped = compute_hyperbolic_phase(
            X_phys, Y_phys, 
            focal_length.view(B, 1, 1), 
            wavelength.view(B, 1, 1), 
            theta=fov.view(B, 1, 1)  # Broadcast fov for grid compatibility
        )
        phi_wrapped = wrap_phase(phi_unwrapped)
        reconstructed = get_2channel_representation(phi_wrapped).permute(0, 3, 1, 2)
        
        loss_physics = self.mse(reconstructed, input_images)
        
        # 3. Auxiliary: Gradient Direction Loss (for fov)
        fov_rad = fov * torch.pi / 180.0
        loss_gradient = self._compute_gradient_loss(input_images, fov_rad)
        
        # 4. Auxiliary: Fringe Density Loss (for λ/f)
        loss_fringe = self._compute_fringe_density_loss(input_images, wavelength, focal_length)
        
        # Total
        total_loss = (
            self.lambda_param * loss_param +
            self.lambda_physics * loss_physics +
            self.lambda_gradient * loss_gradient +
            self.lambda_fringe * loss_fringe
        )
        
        return total_loss, {
            "loss_param": loss_param.item(),
            "loss_physics": loss_physics.item(),
            "loss_gradient": loss_gradient.item(),
            "loss_fringe": loss_fringe.item(),
            "total_loss": total_loss.item()
        }
