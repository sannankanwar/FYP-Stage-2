
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
                           S,
                           focal_length=FOCAL_LENGTH,
                           wavelength=WAVELENGTH,
                           noise_std=0.0):
    """
    Generate one synthetic sample using the centralized forward model.
    
    Args:
        N: Grid resolution (NxN pixels)
        xc, yc: Center coordinates of the observation window (micrometers)
        S: Scaling - physical window size (micrometers). S=20 means 20μm × 20μm.
        focal_length: Lens focal length (micrometers)
        wavelength: Light wavelength (micrometers)
        noise_std: Optional phase noise standard deviation
    """
    # Coordinate grids in physical units using S as window size
    x_coords = np.linspace(xc - S / 2.0, xc + S / 2.0, N, dtype=np.float32)
    y_coords = np.linspace(yc - S / 2.0, yc + S / 2.0, N, dtype=np.float32)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)

    # Physics: Compute unwrapped phase (ideal hyperbolic formula)
    phi_unwrapped = compute_hyperbolic_phase(X_grid, Y_grid, focal_length, wavelength)

    # Augmentation: Add noise in phase domain
    if noise_std > 0.0:
        phi_unwrapped = phi_unwrapped + np.random.normal(
            loc=0.0,
            scale=noise_std,
            size=phi_unwrapped.shape
        ).astype(np.float32)

    # Processing: Wrap and format
    phi_wrapped = wrap_phase(phi_unwrapped)
    input_sample = get_2channel_representation(phi_wrapped)

    # If using PyTorch
    try:
        import torch
        if isinstance(xc, torch.Tensor):
            return input_sample, torch.stack([xc, yc, S, torch.tensor(wavelength), torch.tensor(focal_length)])
    except ImportError:
        pass # torch not available, proceed with numpy

    return input_sample, np.array([xc, yc, S, wavelength, focal_length], dtype=np.float32)


def generate_dataset(N,
                     num_samples,
                     xc_range=(-500.0, 500.0),
                     yc_range=(-500.0, 500.0),
                     S_range=(1.0, 40.0),
                     focal_length=FOCAL_LENGTH,
                     wavelength=WAVELENGTH,
                     noise_std=0.0,
                     seed=None):
    """
    Generate a dataset of phase maps and corresponding [xc, yc, S] labels.
    """
    if seed is not None:
        np.random.seed(seed)

    X = np.zeros((num_samples, N, N, 2), dtype=np.float32)
    y = np.zeros((num_samples, 3), dtype=np.float32)

    for i in range(num_samples):
        xc = np.random.uniform(*xc_range)
        yc = np.random.uniform(*yc_range)
        S = np.random.uniform(*S_range)

        inp, tgt = generate_single_sample(
            N=N,
            xc=xc,
            yc=yc,
            S=S,
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
                          S_range=(1.0, 40.0),
                          wavelength_range=(0.4, 0.7),
                          focal_length_range=(10.0, 100.0),
                          N=128,
                          noise_std=0.0,
                          grid_strategy="mean",
                          offset=False,
                          seed=None):
    """
    Generate a deterministic grid-based dataset for evaluation.
    
    Args:
        xc_count, yc_count (int): Grid steps.
        S_range: Range for scaling parameter (window size in micrometers)
        grid_strategy (str): 
            - "mean": Fix S/wl/fl to center of ranges (only test xc/yc)
            - "random_fixed": Randomize S/wl/fl once (still only tests xc/yc)
            - "random_all": Randomize ALL 5 parameters per sample (true 5-param eval)
        offset (bool): If True, shift grid by 0.5 * step_size (Validation Set).
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Strategy: random_all means we ignore the grid and just sample randomly
    if grid_strategy == "random_all":
        num_samples = xc_count * yc_count
        X = np.zeros((num_samples, N, N, 2), dtype=np.float32)
        y = np.zeros((num_samples, 5), dtype=np.float32)
        
        for idx in range(num_samples):
            xc = np.random.uniform(*xc_range)
            yc = np.random.uniform(*yc_range)
            S = np.random.uniform(*S_range)
            wavelength = np.random.uniform(*wavelength_range)
            focal_length = np.random.uniform(*focal_length_range)
            
            inp, tgt = generate_single_sample(
                N=N,
                xc=xc,
                yc=yc,
                S=S,
                focal_length=focal_length,
                wavelength=wavelength,
                noise_std=noise_std
            )
            
            X[idx] = inp
            y[idx] = tgt
        
        metadata = {
            "xc_steps": None,
            "yc_steps": None,
            "S": "random",
            "wavelength": "random",
            "focal_length": "random",
            "grid_shape": (xc_count, yc_count),
            "strategy": "random_all"
        }
        
        return X, y, metadata
        
    # Original strategies: mean or random_fixed
    if grid_strategy == "random_fixed":
        S = np.random.uniform(*S_range)
        wavelength = np.random.uniform(*wavelength_range)
        focal_length = np.random.uniform(*focal_length_range)
    else: # "mean"
        S = (S_range[0] + S_range[1]) / 2.0
        wavelength = (wavelength_range[0] + wavelength_range[1]) / 2.0
        focal_length = (focal_length_range[0] + focal_length_range[1]) / 2.0

    # Generate Grid Steps
    x_step = (xc_range[1] - xc_range[0]) / xc_count
    y_step = (yc_range[1] - yc_range[0]) / yc_count
    
    xc_steps = np.linspace(xc_range[0], xc_range[1], xc_count, dtype=np.float32)
    yc_steps = np.linspace(yc_range[0], yc_range[1], yc_count, dtype=np.float32)
    
    if offset:
        xc_steps += (x_step / 2.0)
        yc_steps += (y_step / 2.0)

    num_samples = xc_count * yc_count
    X = np.zeros((num_samples, N, N, 2), dtype=np.float32)
    y = np.zeros((num_samples, 5), dtype=np.float32)

    for i, xc in enumerate(xc_steps):
        for j, yc in enumerate(yc_steps):
            idx = i * yc_count + j
            
            inp, tgt = generate_single_sample(
                N=N,
                xc=xc,
                yc=yc,
                S=S,
                focal_length=focal_length,
                wavelength=wavelength,
                noise_std=noise_std
            )

            X[idx] = inp
            y[idx] = tgt

    metadata = {
        "xc_steps": xc_steps,
        "yc_steps": yc_steps,
        "S": S,
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
        self.N = config.get("resolution", 256)
        self.xc_range = tuple(config.get("xc_range", [-500.0, 500.0]))
        self.yc_range = tuple(config.get("yc_range", [-500.0, 500.0]))
        self.S_range = tuple(config.get("S_range", [1.0, 40.0]))
        self.wavelength_range = tuple(config.get("wavelength_range", [0.4, 0.7]))
        self.focal_length_range = tuple(config.get("focal_length_range", [10.0, 100.0]))
        self.noise_std = config.get("noise_std", 0.0)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Randomize parameters
        xc = np.random.uniform(*self.xc_range)
        yc = np.random.uniform(*self.yc_range)
        S = np.random.uniform(*self.S_range)
        wavelength = np.random.uniform(*self.wavelength_range)
        focal_length = np.random.uniform(*self.focal_length_range)
        
        inp, tgt = generate_single_sample(
            N=self.N,
            xc=xc,
            yc=yc,
            S=S,
            focal_length=focal_length,
            wavelength=wavelength,
            noise_std=self.noise_std
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
        S_range = tuple(config.get("S_range", [1.0, 40.0]))
        wavelength_range = tuple(config.get("wavelength_range", [0.4, 0.7]))
        focal_length_range = tuple(config.get("focal_length_range", [10.0, 100.0]))
        grid_strategy = config.get("grid_strategy", "mean")
        noise_std = config.get("noise_std", 0.0)
        
        self.X, self.y, self.metadata = generate_grid_dataset(
            xc_count=steps,
            yc_count=steps,
            xc_range=xc_range,
            yc_range=yc_range,
            S_range=S_range,
            wavelength_range=wavelength_range,
            focal_length_range=focal_length_range,
            N=self.N,
            noise_std=noise_std,
            grid_strategy=grid_strategy,
            offset=offset,
            seed=seed
        )
        
        # X: (N_samples, H, W, 2) -> (N_samples, 2, H, W)
        self.X = np.transpose(self.X, (0, 3, 1, 2))
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])

