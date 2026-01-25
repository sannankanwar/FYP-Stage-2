"""
Scaled Output Layers for Bounded Regression.

These layers map raw model outputs to specific parameter ranges using
appropriate activation functions (tanh for symmetric, sigmoid for asymmetric).
"""
import torch
import torch.nn as nn


class HybridScaledOutput(nn.Module):
    """
    Hybrid output layer using tanh for symmetric params, sigmoid for asymmetric.
    
    Outputs raw physical values - no normalization needed in training.
    
    Parameters and their ranges:
        - xc: [-500, 500] μm (symmetric → tanh)
        - yc: [-500, 500] μm (symmetric → tanh)
        - fov: [1, 20] degrees (asymmetric → sigmoid)
        - wavelength: [0.4, 0.7] μm (asymmetric → sigmoid)
        - focal_length: [10, 100] μm (asymmetric → sigmoid)
    """
    def __init__(self, 
                 xc_range=(-500, 500),
                 yc_range=(-500, 500),
                 fov_range=(1, 20),
                 wavelength_range=(0.4, 0.7),
                 focal_length_range=(10, 100)):
        super().__init__()
        
        # Store ranges for tanh (center + half_range)
        self.xc_center = (xc_range[0] + xc_range[1]) / 2
        self.xc_half = (xc_range[1] - xc_range[0]) / 2
        
        self.yc_center = (yc_range[0] + yc_range[1]) / 2
        self.yc_half = (yc_range[1] - yc_range[0]) / 2
        
        # Store ranges for sigmoid (min + scale)
        self.fov_min = fov_range[0]
        self.fov_scale = fov_range[1] - fov_range[0]
        
        self.wl_min = wavelength_range[0]
        self.wl_scale = wavelength_range[1] - wavelength_range[0]
        
        self.fl_min = focal_length_range[0]
        self.fl_scale = focal_length_range[1] - focal_length_range[0]
    
    def forward(self, x):
        """
        Args:
            x: (B, 5) raw logits from model
        Returns:
            (B, 5) scaled outputs in physical units
        """
        # Symmetric params: tanh
        xc = self.xc_center + self.xc_half * torch.tanh(x[:, 0])
        yc = self.yc_center + self.yc_half * torch.tanh(x[:, 1])
        
        # Asymmetric params: sigmoid
        fov = self.fov_min + self.fov_scale * torch.sigmoid(x[:, 2])
        wl = self.wl_min + self.wl_scale * torch.sigmoid(x[:, 3])
        fl = self.fl_min + self.fl_scale * torch.sigmoid(x[:, 4])
        
        return torch.stack([xc, yc, fov, wl, fl], dim=1)


class ScaledTanhOutput(nn.Module):
    """
    Pure tanh output scaled to parameter ranges.
    All params mapped from tanh [-1,1] to [min, max].
    """
    def __init__(self, param_ranges):
        """
        Args:
            param_ranges: list of (min, max) tuples for each param
        """
        super().__init__()
        mins = torch.tensor([r[0] for r in param_ranges], dtype=torch.float32)
        maxs = torch.tensor([r[1] for r in param_ranges], dtype=torch.float32)
        centers = (mins + maxs) / 2
        half_ranges = (maxs - mins) / 2
        
        self.register_buffer('centers', centers)
        self.register_buffer('half_ranges', half_ranges)
    
    def forward(self, x):
        return self.centers + self.half_ranges * torch.tanh(x)


class ScaledSigmoidOutput(nn.Module):
    """
    Pure sigmoid output scaled to parameter ranges.
    All params mapped from sigmoid [0,1] to [min, max].
    """
    def __init__(self, param_ranges):
        super().__init__()
        mins = torch.tensor([r[0] for r in param_ranges], dtype=torch.float32)
        scales = torch.tensor([r[1] - r[0] for r in param_ranges], dtype=torch.float32)
        
        self.register_buffer('mins', mins)
        self.register_buffer('scales', scales)
    
    def forward(self, x):
        return self.mins + self.scales * torch.sigmoid(x)
