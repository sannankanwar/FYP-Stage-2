import numpy as np
import torch

"""
This file contains the core physical model for the metalens phase maps. 
It supports both NumPy arrays (for data generation) and PyTorch tensors (for differentiable loss).
"""

def compute_hyperbolic_phase(X, Y, focal_length, wavelength):
    """
    Computes the unwrapped hyperbolic phase for a given coordinate grid.
    Supports both numpy and torch inputs.
    """
    # Check if inputs are torch tensors
    is_torch = isinstance(X, torch.Tensor) or isinstance(focal_length, torch.Tensor)
    
    if is_torch:
        pi = torch.pi
        sqrt = torch.sqrt
    else:
        pi = np.pi
        sqrt = np.sqrt

    k0 = 2.0 * pi / wavelength
    R = sqrt(X**2 + Y**2)
    return k0 * (sqrt(R**2 + focal_length**2) - focal_length)

def wrap_phase(unwrapped_phase):
    """
    Wraps the phase into the [-pi, pi] range using the complex exponential method.
    """
    if isinstance(unwrapped_phase, torch.Tensor):
        return torch.angle(torch.exp(1j * unwrapped_phase)).float()
    else:
        return np.angle(np.exp(1j * unwrapped_phase)).astype(np.float32)

def get_2channel_representation(phi_wrapped):
    """
    Converts a wrapped phase map into a stack of [cos, sin] channels.
    """
    if isinstance(phi_wrapped, torch.Tensor):
        return torch.stack(
            [torch.cos(phi_wrapped), torch.sin(phi_wrapped)],
            dim=-1
        )
    else:
        return np.stack(
            [np.cos(phi_wrapped), np.sin(phi_wrapped)],
            axis=-1
        ).astype(np.float32)
