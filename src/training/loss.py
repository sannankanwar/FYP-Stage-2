import torch
import torch.nn as nn
import numpy as np
from src.inversion.forward_model import compute_hyperbolic_phase, wrap_phase, get_2channel_representation

class PhysicsInformedLoss(nn.Module):
    """
    Hybrid loss function combining:
    1. Parameter Loss: MSE(output_params, target_params)
    2. Physics Residual Loss: MSE(Forward(output_params), input_image)
    """
    def __init__(self, lambda_param=1.0, lambda_physics=0.1, 
                 fixed_focal_length=100.0, fixed_wavelength=0.532):
        super().__init__()
        self.lambda_param = lambda_param
        self.lambda_physics = lambda_physics
        self.fixed_focal_length = fixed_focal_length
        self.fixed_wavelength = fixed_wavelength
        self.mse = nn.MSELoss()

    def forward(self, pred_params, true_params, input_images):
        """
        Args:
            pred_params: (B, 3) [xc, yc, fov] or (B, 5) [xc, yc, fov, f, lambda]
            true_params: (B, 3) [xc, yc, fov]
            input_images: (B, C, H, W) where C=2 (cos, sin)
        """
        # 1. Parameter Component
        # If model predicts more params than we have labels for, just match the first 3
        loss_param = self.mse(pred_params[:, :3], true_params[:, :3])

        # 2. Physics Component (Differentiable Reconstruction)
        B, C, H, W = input_images.shape
        device = input_images.device

        xc = pred_params[:, 0]
        yc = pred_params[:, 1]
        fov = pred_params[:, 2]

        # Handle optional dynamic focal_length/wavelength if predicted
        if pred_params.shape[1] >= 5:
            focal_length = pred_params[:, 3]
            wavelength = pred_params[:, 4]
        else:
            focal_length = torch.tensor(self.fixed_focal_length, device=device)
            wavelength = torch.tensor(self.fixed_wavelength, device=device)

        # Create coordinate grids
        # Note: We need to do this in a differentiable way relative to xc, yc, fov
        cols = torch.linspace(-0.5, 0.5, W, device=device).unsqueeze(0).repeat(B, 1) # (B, W)
        rows = torch.linspace(-0.5, 0.5, H, device=device).unsqueeze(0).repeat(B, 1) # (B, H)
        
        # Grid X: xc + fov * col_offset
        # We need a meshgrid per batch item. 
        # Efficient way: (1, 1, H, W) + broadcasting? 
        # Or just manual meshgrid logic flat
        
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
