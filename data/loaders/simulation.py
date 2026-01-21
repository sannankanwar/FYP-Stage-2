import numpy as np
from src.inversion.forward_model import (
    compute_hyperbolic_phase, 
    wrap_phase, 
    get_2channel_representation
)

"""
Inverse Metalens CNN Regressor - Data Generation
This module handles sampling and dataset creation by leveraging the physics 
engine in src.inversion.forward_model.

generate_single_sample generates a single sample using the centralized forward model.
generate_dataset generates a dataset of phase maps and corresponding [xc, yc, fov] labels. It is random sampling.
generate_grid_dataset generates a deterministic grid-based dataset for evaluation.
"""

# Global physical parameters
FOCAL_LENGTH = 100.0     # micrometers
WAVELENGTH = 0.532       # micrometers

def generate_single_sample(N,
                           xc,
                           yc,
                           fov,
                           focal_length=FOCAL_LENGTH,
                           wavelength=WAVELENGTH,
                           noise_std=0.0):
    """
    Generate one synthetic sample using the centralized forward model.
    """
    # Coordinate grids in physical units
    x_coords = np.linspace(xc - fov / 2.0, xc + fov / 2.0, N, dtype=np.float32)
    y_coords = np.linspace(yc - fov / 2.0, yc + fov / 2.0, N, dtype=np.float32)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)

    # 1. Physics: Compute unwrapped phase
    phi_unwrapped = compute_hyperbolic_phase(X_grid, Y_grid, focal_length, wavelength)

    # 2. Augmentation: Add noise in phase domain
    if noise_std > 0.0:
        phi_unwrapped = phi_unwrapped + np.random.normal(
            loc=0.0,
            scale=noise_std,
            size=phi_unwrapped.shape
        ).astype(np.float32)

    # 3. Processing: Wrap and format
    phi_wrapped = wrap_phase(phi_unwrapped)
    input_sample = get_2channel_representation(phi_wrapped)

    target = np.array([xc, yc, fov], dtype=np.float32)

    return input_sample, target


def generate_dataset(N,
                     num_samples,
                     xc_range=(-500.0, 500.0),
                     yc_range=(-500.0, 500.0),
                     fov_range=(10.0, 80.0),
                     focal_length=FOCAL_LENGTH,
                     wavelength=WAVELENGTH,
                     noise_std=0.0,
                     seed=None):
    """
    Generate a dataset of phase maps and corresponding [xc, yc, fov] labels.
    """
    if seed is not None:
        np.random.seed(seed)

    X = np.zeros((num_samples, N, N, 2), dtype=np.float32)
    y = np.zeros((num_samples, 3), dtype=np.float32)

    for i in range(num_samples):
        xc = np.random.uniform(*xc_range)
        yc = np.random.uniform(*yc_range)
        fov = np.random.uniform(*fov_range)

        inp, tgt = generate_single_sample(
            N=N,
            xc=xc,
            yc=yc,
            fov=fov,
            focal_length=focal_length,
            wavelength=wavelength,
            noise_std=noise_std
        )

        X[i] = inp
        y[i] = tgt

    return X, y


def generate_grid_dataset(xc_count,
                          yc_count,
                          fov,
                          N=128,
                          xc_range=(-500.0, 500.0),
                          yc_range=(-500.0, 500.0),
                          focal_length=FOCAL_LENGTH,
                          wavelength=WAVELENGTH,
                          noise_std=0.0):
    """
    Generate a deterministic grid-based dataset for evaluation.
    
    Args:
        xc_count (int): Number of steps along x-coordinate.
        yc_count (int): Number of steps along y-coordinate.
        fov (float): Field of View (fixed for the grid).
        N (int): Image resolution (side length).
        xc_range, yc_range: Range of coordinates to scan.
        
    Returns:
        X: (xc_count * yc_count, N, N, 2)
        y: (xc_count * yc_count, 3)
        metadata: { 'xc_steps': [...], 'yc_steps': [...] }
    """
    xc_steps = np.linspace(xc_range[0], xc_range[1], xc_count, dtype=np.float32)
    yc_steps = np.linspace(yc_range[0], yc_range[1], yc_count, dtype=np.float32)

    num_samples = xc_count * yc_count
    X = np.zeros((num_samples, N, N, 2), dtype=np.float32)
    y = np.zeros((num_samples, 3), dtype=np.float32)

    for i, xc in enumerate(xc_steps):
        for j, yc in enumerate(yc_steps):
            idx = i * yc_count + j
            
            inp, tgt = generate_single_sample(
                N=N,
                xc=xc,
                yc=yc,
                fov=fov,
                focal_length=focal_length,
                wavelength=wavelength,
                noise_std=noise_std
            )

            X[idx] = inp
            y[idx] = tgt

    metadata = {
        "xc_steps": xc_steps,
        "yc_steps": yc_steps,
        "fov": fov,
        "grid_shape": (xc_count, yc_count)
    }

    return X, y, metadata
