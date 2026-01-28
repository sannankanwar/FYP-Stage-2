import torch
import pytest
from src.inversion.forward_model import compute_hyperbolic_phase, wrap_phase, get_2channel_representation
import numpy as np

def test_forward_model_shapes():
    B, H, W = 2, 64, 64
    xc = torch.zeros(B)
    yc = torch.zeros(B)
    f = torch.ones(B) * 10.0
    wl = torch.ones(B) * 0.5
    
    # Grid construction (simplified simulation of what dataset does)
    grid_y, grid_x = torch.meshgrid(torch.linspace(-50,50,H), torch.linspace(-50,50,W), indexing='ij')
    # Physical phase calculation
    # X, Y need to be broadcast against batch if we aren't careful, 
    # but compute_hyperbolic_phase expects compatible broadcasting.
    
    # Let's test single item first as the func might not handle batch dims natively without broadcast inputs
    # Looking at code: X^2 + Y^2. If X is (H,W) and f is (B,), we get (B,H,W).
    
    f_expanded = f.view(B, 1, 1)
    wl_expanded = wl.view(B, 1, 1)
    
    phi = compute_hyperbolic_phase(grid_x.unsqueeze(0), grid_y.unsqueeze(0), f_expanded, wl_expanded)
    assert phi.shape == (B, H, W)
    assert not torch.isnan(phi).any()

    wrapped = wrap_phase(phi)
    assert wrapped.shape == (B, H, W)
    assert wrapped.min() >= -torch.pi - 1e-5
    assert wrapped.max() <= torch.pi + 1e-5

    channels = get_2channel_representation(wrapped)
    # The function returns (..., 2)
    assert channels.shape == (B, H, W, 2) 
    
    # Verify cos^2 + sin^2 = 1
    norm = channels[..., 0]**2 + channels[..., 1]**2
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)
