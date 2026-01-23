
import numpy as np
from torch.utils.data import Dataset
import torch
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
                           noise_std=0.0,
                           window_size=100.0):
    """
    Generate one synthetic sample using the centralized forward model.
    fov is now treated as incident angle in degrees.
    window_size is the physical width of the patch in micrometers.
    """
    # Coordinate grids in physical units
    x_coords = np.linspace(xc - window_size / 2.0, xc + window_size / 2.0, N, dtype=np.float32)
    y_coords = np.linspace(yc - window_size / 2.0, yc + window_size / 2.0, N, dtype=np.float32)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)

    # 1. Physics: Compute unwrapped phase (passing fov as theta)
    phi_unwrapped = compute_hyperbolic_phase(X_grid, Y_grid, focal_length, wavelength, theta=fov)

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

    # If using PyTorch
    try:
        import torch
        if isinstance(xc, torch.Tensor):
            return input_sample, torch.stack([xc, yc, fov, torch.tensor(wavelength), torch.tensor(focal_length)])
    except ImportError:
        pass # torch not available, proceed with numpy

    return input_sample, np.array([xc, yc, fov, wavelength, focal_length], dtype=np.float32)


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
                          xc_range=(-500.0, 500.0),
                          yc_range=(-500.0, 500.0),
                          fov_range=(1.0, 20.0),
                          wavelength_range=(0.4, 0.7),
                           focal_length_range=(10.0, 100.0),
                          window_size=100.0,
                          N=128,
                          noise_std=0.0,
                          grid_strategy="mean",
                          offset=False,
                          seed=None):
    """
    Generate a deterministic grid-based dataset for evaluation.
    
    Args:
        xc_count, yc_count (int): Grid steps.
        grid_strategy (str): "mean" (use center of ranges) or "random_fixed" (randomize once).
        offset (bool): If True, shift grid by 0.5 * step_size (Validation Set).
    """
    if seed is not None:
        np.random.seed(seed)
        
    # 1. Determine "Constant" Parameters based on strategy
    if grid_strategy == "random_fixed":
        fov = np.random.uniform(*fov_range)
        wavelength = np.random.uniform(*wavelength_range)
        focal_length = np.random.uniform(*focal_length_range)
    else: # "mean"
        fov = (fov_range[0] + fov_range[1]) / 2.0
        wavelength = (wavelength_range[0] + wavelength_range[1]) / 2.0
        focal_length = (focal_length_range[0] + focal_length_range[1]) / 2.0

    # 2. Generate Grid Steps
    x_step = (xc_range[1] - xc_range[0]) / xc_count
    y_step = (yc_range[1] - yc_range[0]) / yc_count
    
    xc_steps = np.linspace(xc_range[0], xc_range[1], xc_count, dtype=np.float32)
    yc_steps = np.linspace(yc_range[0], yc_range[1], yc_count, dtype=np.float32)
    
    if offset:
        # Shift by half step to interleave with training grid
        # Careful not to go out of bounds? 
        # For validation, testing interpolation is key.
        xc_steps += (x_step / 2.0)
        yc_steps += (y_step / 2.0)

    num_samples = xc_count * yc_count
    X = np.zeros((num_samples, N, N, 2), dtype=np.float32)
    y = np.zeros((num_samples, 5), dtype=np.float32) # 5 Params

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
                noise_std=noise_std,
                window_size=window_size
            )

            X[idx] = inp
            y[idx] = tgt # (5,)

    metadata = {
        "xc_steps": xc_steps,
        "yc_steps": yc_steps,
        "fov": fov,
        "wavelength": wavelength,
        "focal_length": focal_length,
        "grid_shape": (xc_count, yc_count)
    }

    return X, y, metadata

class OnTheFlyDataset(Dataset):
    """
    Dataset that generates samples on the fly to avoid memory issues with large datasets.
    """
    def __init__(self, config, length=1000):
        self.config = config
        self.length = length
        
        # Extract params
        self.N = config.get("resolution", 256) # Default to 256 if not set, be careful with HighRes
        self.xc_range = tuple(config.get("xc_range", [-500.0, 500.0]))
        self.yc_range = tuple(config.get("yc_range", [-500.0, 500.0]))
        self.fov_range = tuple(config.get("fov_range", [10.0, 80.0]))
        self.wavelength_range = tuple(config.get("wavelength_range", [0.4, 0.7]))
        self.focal_length_range = tuple(config.get("focal_length_range", [10.0, 100.0]))
        self.window_size = config.get("window_size", 100.0)
        self.noise_std = config.get("noise_std", 0.0)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Randomize parameters
        xc = np.random.uniform(*self.xc_range)
        yc = np.random.uniform(*self.yc_range)
        fov = np.random.uniform(*self.fov_range)
        wavelength = np.random.uniform(*self.wavelength_range)
        focal_length = np.random.uniform(*self.focal_length_range)
        
        inp, tgt = generate_single_sample(
            N=self.N,
            xc=xc,
            yc=yc,
            fov=fov,
            focal_length=focal_length,
            wavelength=wavelength,
            noise_std=self.noise_std,
            window_size=self.window_size
        )
        
        # inp is (H, W, 2), convert to (2, H, W) for PyTorch
        inp = np.transpose(inp, (2, 0, 1))
        
        return torch.from_numpy(inp), torch.from_numpy(tgt)

class GridDataset(Dataset):
    """
    Fixed grid-based dataset for training (even epochs) and validation.
    Wraps generate_grid_dataset.
    """
    def __init__(self, config, steps=10, offset=False, seed=None):
        self.config = config
        self.N = config.get("resolution", 256)
        
        xc_range = tuple(config.get("xc_range", [-500.0, 500.0]))
        yc_range = tuple(config.get("yc_range", [-500.0, 500.0]))
        fov_range = tuple(config.get("fov_range", [1.0, 20.0]))
        wavelength_range = tuple(config.get("wavelength_range", [0.4, 0.7]))
        focal_length_range = tuple(config.get("focal_length_range", [10.0, 100.0]))
        grid_strategy = config.get("grid_strategy", "mean")
        window_size = config.get("window_size", 100.0)
        noise_std = config.get("noise_std", 0.0)
        
        self.X, self.y, self.metadata = generate_grid_dataset(
            xc_count=steps,
            yc_count=steps,
            xc_range=xc_range,
            yc_range=yc_range,
            fov_range=fov_range,
            wavelength_range=wavelength_range,
            focal_length_range=focal_length_range,
            N=self.N,
            noise_std=noise_std,
            grid_strategy=grid_strategy,
            window_size=window_size,
            offset=offset,
            seed=seed
        )
        
        # X: (N_samples, H, W, 2) -> (N_samples, 2, H, W)
        self.X = np.transpose(self.X, (0, 3, 1, 2))
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])
