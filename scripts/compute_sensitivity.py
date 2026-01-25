"""
Compute Physics Sensitivity (Jacobian-based) for Loss Weighting.
Measures the mean change in Phase Map per unit change in each parameter.

Sensitivity S_i = mean( | d(Phase) / d(param_i) | )
Recommended Weight w_i = 1 / S_i
"""
import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.inversion.forward_model import compute_hyperbolic_phase, wrap_phase

def compute_sensitivity(N=256, samples=100, device='cpu'):
    print(f"Computing sensitivities using {samples} samples at {N}x{N} resolution...")
    
    # 1. Define Parameter Ranges (Physical Units)
    ranges = {
        'xc': (-500, 500),
        'yc': (-500, 500),
        'S': (1, 40),  # Scaling = window size in micrometers
        'wavelength': (0.4, 0.7),
        'focal_length': (10, 100)
    }
    
    # 2. Generate Random Base Parameters
    # Shape: (samples, 1, 1) to broadcast to image
    xc = torch.empty(samples, 1, 1).uniform_(*ranges['xc']).to(device)
    yc = torch.empty(samples, 1, 1).uniform_(*ranges['yc']).to(device)
    S = torch.empty(samples, 1, 1).uniform_(*ranges['S']).to(device)
    wl = torch.empty(samples, 1, 1).uniform_(*ranges['wavelength']).to(device)
    fl = torch.empty(samples, 1, 1).uniform_(*ranges['focal_length']).to(device)
    
    # Enable gradients
    xc.requires_grad = True
    yc.requires_grad = True
    S.requires_grad = True
    wl.requires_grad = True
    fl.requires_grad = True
    
    # 3. Create Grid (normalized coordinates)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-0.5, 0.5, N, device=device),
        torch.linspace(-0.5, 0.5, N, device=device),
        indexing='ij'
    )
    grid_x = grid_x.unsqueeze(0)  # (1, H, W)
    grid_y = grid_y.unsqueeze(0)
    
    # Physical coordinates using S as window size
    X_phys = xc + S * grid_x
    Y_phys = yc + S * grid_y
    
    # 4. Forward Pass (Phase Computation)
    # Unwrapped phase captures the magnitude of change best
    phi = compute_hyperbolic_phase(X_phys, Y_phys, fl, wl)
    
    # 5. Compute Gradients
    # Use Hutchinson's estimator (random vector) to avoid spatial cancellation
    # J^T v where v ~ N(0, 1). E[|grad|^2] ~ Sum(J_ij^2)
    v = torch.randn_like(phi)
    phi.backward(gradient=v)
    
    # 6. Aggregate Sensitivities
    # Sensitivity = Mean( |dPhi/dParam| ) per pixel
    sensitivities = {
        'xc': xc.grad.abs().mean().item() / (N*N),
        'yc': yc.grad.abs().mean().item() / (N*N),
        'S': S.grad.abs().mean().item() / (N*N),
        'wavelength': wl.grad.abs().mean().item() / (N*N),
        'focal_length': fl.grad.abs().mean().item() / (N*N)
    }
    
    # Normalize per pixel
    sensitivities['xc'] /= (N*N)
    sensitivities['yc'] /= (N*N)
    sensitivities['S'] /= (N*N)
    sensitivities['wavelength'] /= (N*N)
    sensitivities['focal_length'] /= (N*N)
    
    # 7. Compute Weights (Inverse Sensitivity)
    weights = {k: 1.0 / (v + 1e-12) for k, v in sensitivities.items()}
    
    return sensitivities, weights

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
        
    S, W = compute_sensitivity(N=256, samples=100, device=device)
    
    print("\n=== Physics Sensitivity Analysis ===")
    print(f"{'Parameter':<15} | {'Sensitivity (dPhi/dp)':<25} | {'Recommended Weight':<20}")
    print("-" * 65)
    
    order = ['xc', 'yc', 'S', 'wavelength', 'focal_length']
    for k in order:
        print(f"{k:<15} | {S[k]:<25.6e} | {W[k]:<20.4e}")
        
    print("\nCopy these weights to your config for Strategy #3!")

