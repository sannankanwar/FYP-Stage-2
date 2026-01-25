#!/usr/bin/env python
"""
Generate residual phase maps and R²/MSE progression plots for experiment comparison.
"""
import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import load_config
from src.models.factory import get_model
from src.utils.normalization import ParameterNormalizer
from src.inversion.forward_model import compute_hyperbolic_phase, wrap_phase
from data.loaders.simulation import generate_single_sample


def load_model_from_checkpoint(checkpoint_path, config):
    """Load model from checkpoint."""
    model = get_model(config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def compute_residual_phase(pred_params, true_params, resolution=1024, window_size=100.0):
    """
    Compute residual between predicted and true phase maps.
    Returns: pred_phase, true_phase, residual
    """
    # Generate true phase map
    true_inp, _ = generate_single_sample(
        N=resolution,
        xc=true_params[0],
        yc=true_params[1],
        fov=true_params[2],
        wavelength=true_params[3],
        focal_length=true_params[4],
        window_size=window_size
    )
    
    # Generate predicted phase map
    pred_inp, _ = generate_single_sample(
        N=resolution,
        xc=pred_params[0],
        yc=pred_params[1],
        fov=pred_params[2],
        wavelength=pred_params[3],
        focal_length=pred_params[4],
        window_size=window_size
    )
    
    # Extract cos channel (or reconstruct phase)
    true_phase = np.arctan2(true_inp[:,:,1], true_inp[:,:,0])
    pred_phase = np.arctan2(pred_inp[:,:,1], pred_inp[:,:,0])
    
    # Residual
    residual = true_phase - pred_phase
    # Wrap residual to [-pi, pi]
    residual = np.angle(np.exp(1j * residual))
    
    return pred_phase, true_phase, residual


def plot_residual_phase_map(exp_dir, output_path):
    """
    Generate residual phase map for a single experiment at final epoch.
    """
    # Find best model
    checkpoint_path = os.path.join(exp_dir, "checkpoints", "best_model.pth")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(exp_dir, "checkpoints", "latest_checkpoint.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found in {exp_dir}")
        return None
    
    # Load config
    config_path = os.path.join(exp_dir, "config.yaml")
    if os.path.exists(config_path):
        config = load_config(config_path)
    else:
        # Fallback: infer from experiment name
        exp_name = os.path.basename(exp_dir)
        print(f"Warning: No config.yaml found for {exp_name}")
        return None
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint.get('config', config)
    model = get_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Setup normalizer
    ranges = {
        'xc': tuple(config.get("xc_range", [-500, 500])),
        'yc': tuple(config.get("yc_range", [-500, 500])),
        'fov': tuple(config.get("fov_range", [1, 20])),
        'wavelength': tuple(config.get("wavelength_range", [0.4, 0.7])),
        'focal_length': tuple(config.get("focal_length_range", [10, 100])),
    }
    normalizer = ParameterNormalizer(ranges)
    
    # Generate a test sample
    resolution = config.get("resolution", 1024)
    window_size = config.get("window_size", 100.0)
    
    # Fixed test parameters
    true_xc, true_yc = 100.0, -150.0
    true_fov = 10.0
    true_wl = 0.55
    true_fl = 50.0
    
    true_params = np.array([true_xc, true_yc, true_fov, true_wl, true_fl])
    
    # Generate input
    inp, _ = generate_single_sample(
        N=resolution,
        xc=true_xc, yc=true_yc, fov=true_fov,
        wavelength=true_wl, focal_length=true_fl,
        window_size=window_size
    )
    inp_tensor = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        pred_norm = model(inp_tensor)
        pred_denorm = normalizer.denormalize_tensor(pred_norm)
        pred_params = pred_denorm.squeeze().numpy()
    
    # Compute residual phase
    pred_phase, true_phase, residual = compute_residual_phase(
        pred_params, true_params, resolution, window_size
    )
    
    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # True phase
    im0 = axes[0].imshow(true_phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[0].set_title('True Phase')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    
    # Predicted phase
    im1 = axes[1].imshow(pred_phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[1].set_title('Predicted Phase')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    # Residual
    im2 = axes[2].imshow(residual, cmap='RdBu', vmin=-np.pi, vmax=np.pi)
    axes[2].set_title('Residual (True - Pred)')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    # Parameter comparison
    param_names = ['xc', 'yc', 'fov', 'λ', 'f']
    x = np.arange(len(param_names))
    width = 0.35
    axes[3].bar(x - width/2, true_params, width, label='True', color='blue', alpha=0.7)
    axes[3].bar(x + width/2, pred_params, width, label='Pred', color='orange', alpha=0.7)
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(param_names)
    axes[3].legend()
    axes[3].set_title('Parameter Comparison')
    
    exp_name = os.path.basename(exp_dir)
    fig.suptitle(f'{exp_name} - Residual Phase Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved: {output_path}")
    return {'true': true_params, 'pred': pred_params}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs_3", help="Output directory with experiments")
    args = parser.parse_args()
    
    output_dir = args.output_dir
    
    # Find all experiment directories
    exp_dirs = sorted(glob(os.path.join(output_dir, "exp4_*")))
    
    if not exp_dirs:
        print(f"No exp4_* directories found in {output_dir}")
        return
    
    print(f"Found {len(exp_dirs)} experiments")
    
    # Generate residual phase maps
    for exp_dir in exp_dirs:
        exp_name = os.path.basename(exp_dir)
        output_path = os.path.join(exp_dir, "residual_phase_map.png")
        plot_residual_phase_map(exp_dir, output_path)


if __name__ == "__main__":
    main()
