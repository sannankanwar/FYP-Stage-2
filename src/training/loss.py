import torch
import torch.nn as nn
import numpy as np
from src.inversion.forward_model import compute_hyperbolic_phase, wrap_phase, get_2channel_representation


# =============================================================================
#  OLD PHYSICS LOSSES (COMMENTED OUT FOR PURE REGRESSION FOCUS)
# =============================================================================

# class RawPhysicsLoss(nn.Module):
#     """
#     Loss function for models with HybridScaledOutput (outputs raw physical values).
#     No normalization needed - directly compares raw predictions to raw targets.
#     
#     Components:
#         1. Weighted MSE on raw parameters (scale-aware weighting)
#         2. Physics reconstruction loss
#     """
#     def __init__(self, 
#                  lambda_param=1.0, 
#                  lambda_physics=0.5,
#                  param_weights=None):
#         super().__init__()
#         self.lambda_param = lambda_param
#         self.lambda_physics = lambda_physics
#         self.mse = nn.MSELoss()
#         
#         # Weights inversely proportional to typical magnitude squared
#         # This makes each param contribute equally to loss
#         if param_weights is None:
#             # xc~500, yc~500, S~20, wavelength~0.5, focal_length~50
#             param_weights = [1/(500**2), 1/(500**2), 1/(20**2), 1/(0.15**2), 1/(45**2)]
#         
#         # Ensure weights are floats (handle potential string parsing from config)
#         param_weights = [float(w) for w in param_weights]
#         self.register_buffer('weights', torch.tensor(param_weights, dtype=torch.float32))
#     
#     def forward(self, pred_params, true_params, input_images):
#         """
#         Args:
#             pred_params: (B, 5) raw physical values [xc, yc, S, wavelength, focal_length]
#             true_params: (B, 5) raw physical values from dataset
#             input_images: (B, 2, H, W) input phase maps
#         """
#         # 1. Weighted Parameter Loss (scale-aware)
#         diff = pred_params - true_params  # (B, 5)
#         weighted_sq_diff = self.weights * (diff ** 2)
#         loss_param = weighted_sq_diff.mean()
#         
#         # 2. Physics Reconstruction Loss
#         B, C, H, W = input_images.shape
#         device = input_images.device
#         
#         xc = pred_params[:, 0]
#         yc = pred_params[:, 1]
#         S = pred_params[:, 2]  # Scaling = window size
#         wavelength = pred_params[:, 3]
#         focal_length = pred_params[:, 4]
#         
#         # Create coordinate grids
#         grid_y, grid_x = torch.meshgrid(
#             torch.linspace(-0.5, 0.5, H, device=device),
#             torch.linspace(-0.5, 0.5, W, device=device),
#             indexing='ij'
#         )
#         grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
#         grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)
#         
#         # Use S as window size for coordinate grids
#         X_phys = xc.view(B, 1, 1) + S.view(B, 1, 1) * grid_x
#         Y_phys = yc.view(B, 1, 1) + S.view(B, 1, 1) * grid_y
#         
#         # Broadcast params to (B, 1, 1) for physics formula
#         focal_length = focal_length.view(B, 1, 1)
#         wavelength = wavelength.view(B, 1, 1)
#         
#         phi_unwrapped = compute_hyperbolic_phase(X_phys, Y_phys, focal_length, wavelength)
#         phi_wrapped = wrap_phase(phi_unwrapped)
#         reconstructed = get_2channel_representation(phi_wrapped).permute(0, 3, 1, 2)
#         
#         loss_physics = self.mse(reconstructed, input_images)
#         
#         total_loss = (self.lambda_param * loss_param) + (self.lambda_physics * loss_physics)
#         
#         return total_loss, {
#             "loss_param": loss_param.item(),
#             "loss_physics": loss_physics.item(),
#             "total_loss": total_loss.item()
#         }


# class AdaptivePhysicsLoss(nn.Module):
#     """
#     Kendall's Aleatoric Uncertainty Loss.
#     Learns weights dynamically: L = (1/2sigma^2) * MSE + log(sigma)
#     """
#     def __init__(self, 
#                  num_params=5,
#                  lambda_param=1.0, 
#                  lambda_physics=0.5):
#         super().__init__()
#         self.lambda_param = lambda_param
#         self.lambda_physics = lambda_physics
#         self.mse = nn.MSELoss()
#         
#         # Learnable log variances (one per parameter)
#         self.log_vars = nn.Parameter(torch.zeros(num_params))
#         
#     def forward(self, pred_params, true_params, input_images):
#         # 1. Adaptive Parameter Loss
#         diff = (pred_params - true_params) ** 2
#         precision = torch.exp(-self.log_vars)
#         
#         # Kendall Loss: sum( precision * diff + log_vars )
#         loss_param = (precision * diff + self.log_vars).sum(dim=1).mean()
#         
#         # 2. Physics Reconstruction Loss
#         B, C, H, W = input_images.shape
#         device = input_images.device
#         
#         xc = pred_params[:, 0]
#         yc = pred_params[:, 1]
#         S = pred_params[:, 2]  # Scaling = window size
#         wavelength = pred_params[:, 3]
#         focal_length = pred_params[:, 4]
#         
#         grid_y, grid_x = torch.meshgrid(
#             torch.linspace(-0.5, 0.5, H, device=device),
#             torch.linspace(-0.5, 0.5, W, device=device),
#             indexing='ij'
#         )
#         grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
#         grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)
#         
#         # Use S as window size
#         X_phys = xc.view(B, 1, 1) + S.view(B, 1, 1) * grid_x
#         Y_phys = yc.view(B, 1, 1) + S.view(B, 1, 1) * grid_y
#         
#         # Focal and Wavelength
#         focal_length = focal_length.view(B, 1, 1)
#         wavelength = wavelength.view(B, 1, 1)
#         
#         phi_unwrapped = compute_hyperbolic_phase(X_phys, Y_phys, focal_length, wavelength)
#         phi_wrapped = wrap_phase(phi_unwrapped)
#         reconstructed = get_2channel_representation(phi_wrapped).permute(0, 3, 1, 2)
#         
#         loss_physics = self.mse(reconstructed, input_images)
#         
#         total_loss = (self.lambda_param * loss_param) + (self.lambda_physics * loss_physics)
#         
#         return total_loss, {
#             "loss_param": loss_param.item(),
#             "loss_physics": loss_physics.item(),
#             "total_loss": total_loss.item(),
#             # "sigma_xc": torch.exp(self.log_vars[0]).item(),
#             # "sigma_S": torch.exp(self.log_vars[2]).item()
#         }


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
        # Ensure weights are floats
        weights = [float(w) for w in weights]
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.normalizer = normalizer

    def forward(self, pred_params, true_params, input_images=None):
        if self.weights.device != pred_params.device:
            self.weights = self.weights.to(pred_params.device)
            
        if self.normalizer:
            pred_params_norm = self.normalizer.normalize_tensor(pred_params)
            true_params_norm = self.normalizer.normalize_tensor(true_params)
            diff = (pred_params_norm - true_params_norm)
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


# class WeightedPhysicsLoss(nn.Module):
#     """
#     Hybrid loss: Weighted Parameter Loss + Physics Reconstruction Loss.
#     """
#     def __init__(self, 
#                  lambda_param=1.0, 
#                  lambda_physics=0.1, 
#                  param_weights=[1.0, 1.0, 1.0, 10.0, 10.0],
#                  fixed_focal_length=100.0, 
#                  fixed_wavelength=0.532, 
#                  normalizer=None):
#         super().__init__()
#         self.lambda_param = lambda_param
#         self.lambda_physics = lambda_physics
#         self.param_loss = WeightedStandardizedLoss(weights=param_weights, normalizer=normalizer)
#         self.normalizer = normalizer
#         
#         self.fixed_focal_length = fixed_focal_length
#         self.fixed_wavelength = fixed_wavelength
#         self.mse = nn.MSELoss()
# 
#     def forward(self, pred_params, true_params, input_images):
#         """
#         Args:
#             pred_params: (B, 5) [xc, yc, S, wl, fl] (Normalized)
#             true_params: (B, 5) [xc, yc, S, wl, fl] (Real Units)
#             input_images: (B, C, H, W)
#         """
#         # 1. Parameter Component
#         loss_param, _ = self.param_loss(pred_params, true_params)
# 
#         # 2. Physics Component (Differentiable Reconstruction)
#         # Inputs are already physical from HybridScaledOutput
#         pred_params_phys = pred_params
# 
#         B, C, H, W = input_images.shape
#         device = input_images.device
# 
#         xc = pred_params_phys[:, 0]
#         yc = pred_params_phys[:, 1]
#         S = pred_params_phys[:, 2]  # Scaling = window size
# 
#         # Handle optional dynamic focal_length/wavelength if predicted
#         if pred_params_phys.shape[1] >= 5:
#             wavelength = pred_params_phys[:, 3]
#             focal_length = pred_params_phys[:, 4]
#         else:
#             focal_length = torch.tensor(self.fixed_focal_length, device=device).expand(B)
#             wavelength = torch.tensor(self.fixed_wavelength, device=device).expand(B)
# 
#         # Create coordinate grids
#         grid_y, grid_x = torch.meshgrid(
#             torch.linspace(-0.5, 0.5, H, device=device),
#             torch.linspace(-0.5, 0.5, W, device=device),
#             indexing='ij'
#         )
#         grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
#         grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)
# 
#         # Use S as window size
#         X_phys = xc.view(B, 1, 1) + S.view(B, 1, 1) * grid_x
#         Y_phys = yc.view(B, 1, 1) + S.view(B, 1, 1) * grid_y
# 
#         focal_length = focal_length.view(B, 1, 1)
#         wavelength = wavelength.view(B, 1, 1)
# 
#         phi_unwrapped = compute_hyperbolic_phase(X_phys, Y_phys, focal_length, wavelength)
#         phi_wrapped = wrap_phase(phi_unwrapped)
#         reconstructed_image = get_2channel_representation(phi_wrapped).permute(0, 3, 1, 2)
# 
#         loss_physics = self.mse(reconstructed_image, input_images)
# 
#         total_loss = (self.lambda_param * loss_param) + (self.lambda_physics * loss_physics)
#         
#         return total_loss, {
#             "loss_param": loss_param.item(), 
#             "loss_physics": loss_physics.item(),
#             "total_loss": total_loss.item()
#         }


# class AuxiliaryPhysicsLoss(nn.Module):
#     """
#     Enhanced physics loss with auxiliary task for fringe density.
#     
#     Auxiliary Tasks:
#     1. Fringe Density Loss: λ/f ratio determines fringe spacing - match FFT peak frequency
#     """
#     def __init__(self, 
#                  lambda_param=1.0, 
#                  lambda_physics=0.5,
#                  lambda_fringe=0.1,
#                  param_weights=[1.0, 1.0, 5.0, 20.0, 20.0],
#                  normalizer=None):
#         super().__init__()
#         self.lambda_param = lambda_param
#         self.lambda_physics = lambda_physics
#         self.lambda_fringe = lambda_fringe
#         self.param_loss = WeightedStandardizedLoss(weights=param_weights, normalizer=normalizer)
#         self.normalizer = normalizer
#         self.mse = nn.MSELoss()
# 
#     def _compute_fringe_density_loss(self, input_images, pred_wavelength, pred_focal_length, pred_S):
#         # ... (Implementation Omitted for Commenting Out)
#         pass
# 
#     def forward(self, pred_params, true_params, input_images):
#         # ... (Implementation Omitted for Commenting Out)
#         pass


# =============================================================================
#  NEW RESEARCH-BACKED PURE REGRESSION LOSSES
# =============================================================================

class RobustHuberLoss(nn.Module):
    """
    Robust Loss Strategy 1: Huber Loss.
    Combines L2 (near zero) and L1 (far from zero) to robustness against outliers.
    
    Experiment: exp_pure_huber
    Paper: "Deep Learning with Robust Loss Functions" (Standard Practice)
    Intention: Prevent outliers (hard examples) from dominating gradients while maintaining smooth convergence near zero.
    """
    def __init__(self, delta=1.0, normalizer=None):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta)
        self.normalizer = normalizer
        
    def forward(self, pred_params, true_params, input_images=None):
        if self.normalizer:
            pred = self.normalizer.normalize_tensor(pred_params)
            true = self.normalizer.normalize_tensor(true_params)
        else:
            pred = pred_params
            true = true_params
            
        loss = self.huber(pred, true)
        return loss, {"total_loss": loss.item()}


class RobustLogCoshLoss(nn.Module):
    """
    Robust Loss Strategy 2: Log-Cosh Loss.
    log(cosh(x)) approximates x^2/2 for small x and |x| - log(2) for large x.
    
    Experiment: exp_pure_logcosh
    Paper: "Log-Cosh Loss for Robust Regression" (Chen et al.)
    Intention: Similar to Huber but fully differentiable everywhere (C-infinity smooth), often leading to better optimization qualities.
    """
    def __init__(self, normalizer=None):
        super().__init__()
        self.normalizer = normalizer
        
    def forward(self, pred_params, true_params, input_images=None):
        if self.normalizer:
            pred = self.normalizer.normalize_tensor(pred_params)
            true = self.normalizer.normalize_tensor(true_params)
        else:
            pred = pred_params
            true = true_params
            
        diff = pred - true
        loss = torch.mean(torch.log(torch.cosh(diff)))
        return loss, {"total_loss": loss.item()}


class LogSpaceMSELoss(nn.Module):
    """
    Log-Space Regression Strategy 3: Mean Squared Logarithmic Error (MSLE).
    Computes MSE on log(1 + x).
    
    Experiment: exp_pure_msle
    Paper: Standard technique for multiplicative noise or wide dynamic ranges.
    Intention: Physics parameters (wavelength, focal length) vary by orders of magnitude. 
    Relative error is often more physically meaningful than absolute error (e.g. 1um error matters more at 10um than at 100um).
    """
    def __init__(self, normalizer=None):
        super().__init__()
        # Note: Normalizer handles standardization. If normalizing, MSLE might be weird on Z-scores (negatives).
        # THIS LOSS ASSUMES RAW POSITIVE INPUTS.
        # If normalizer is provided, we denormalize first to get positive physical values.
        self.normalizer = normalizer
        self.mse = nn.MSELoss()
        
    def forward(self, pred_params, true_params, input_images=None):
        # Ensure we work in physical space (positive values)
        if self.normalizer:
            # We assume model outputs physical directly (HybridScaledOutput)
            # But true_params might be normalized if passed from dataset? No, trainer passes raw true_params usually?
            # Trainer passes Raw `true_params`.
            # If `standardize_outputs: true` in config, Trainer passes Normalized `true_params`.
            # So if this loss is used, we MUST set `standardize_outputs: false` or handle it.
            # We will handle it:
            pass
            
        # However, MSLE is best on physical values.
        # Check for negatives (which would break log).
        # We apply clamp to be safe.
        pred_safe = torch.relu(pred_params) + 1e-6
        true_safe = torch.relu(true_params) + 1e-6
        
        loss = self.mse(torch.log1p(pred_safe), torch.log1p(true_safe))
        return loss, {"total_loss": loss.item()}


class WingRegressionLoss(nn.Module):
    """
    Coordinate Precision Strategy 4: Wing Loss.
    Designed for facial landmarks, adapted here for xc, yc coordinates.
    
    Experiment: exp_pure_wing
    Paper: "Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks" (CVPR 2018)
    Intention: Enhances gradient signal for small errors (unlike MSE which vanishes), promoting higher precision in coordinate prediction.
    """
    def __init__(self, w=10.0, epsilon=2.0, normalizer=None):
        super().__init__()
        self.w = w
        self.epsilon = epsilon
        self.normalizer = normalizer
        self.C = self.w - self.w * np.log(1 + self.w / self.epsilon)
        
    def forward(self, pred_params, true_params, input_images=None):
        if self.normalizer:
            pred = self.normalizer.normalize_tensor(pred_params)
            true = self.normalizer.normalize_tensor(true_params)
        else:
            pred = pred_params
            true = true_params
            
        diff = pred - true
        abs_diff = torch.abs(diff)
        
        # Wing Logic
        mask = abs_diff < self.w
        loss_small = self.w * torch.log(1 + abs_diff / self.epsilon)
        loss_large = abs_diff - self.C
        
        loss = torch.where(mask, loss_small, loss_large)
        return loss.mean(), {"total_loss": loss.mean().item()}


class BiweightRegressionLoss(nn.Module):
    """
    Robust Regression Strategy 5: Tukey's Biweight Loss.
    Aggressively suppresses outliers (gradients -> 0 for large errors).
    
    Experiment: exp_pure_biweight
    Paper: "Robust Regression and Outlier Detection" (Tukey)
    Intention: Unlike Huber (which becomes linear L1), Tukey's Biweight completely ignores extreme outliers, treating them as irrelevant noise.
    """
    def __init__(self, c=4.685, normalizer=None):
        super().__init__()
        self.c = c
        self.normalizer = normalizer
        
    def forward(self, pred_params, true_params, input_images=None):
        if self.normalizer:
            pred = self.normalizer.normalize_tensor(pred_params)
            true = self.normalizer.normalize_tensor(true_params)
        else:
            pred = pred_params
            true = true_params
            
        diff = pred - true
        # c represents ~4.7 standard deviations.
        
        # Loss:
        # if |x| <= c: c^2/6 * (1 - (1 - (x/c)^2)^3)
        # if |x| > c: c^2/6
        
        abs_diff = torch.abs(diff)
        mask = abs_diff <= self.c
        
        loss_inlier = (self.c**2 / 6.0) * (1.0 - (1.0 - (diff / self.c)**2)**3)
        loss_outlier = self.c**2 / 6.0
        
        loss = torch.where(mask, loss_inlier, torch.tensor(loss_outlier, device=diff.device))
        
        return loss.mean(), {"total_loss": loss.mean().item()}
