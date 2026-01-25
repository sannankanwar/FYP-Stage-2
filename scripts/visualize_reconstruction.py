import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys

# Add src to path
sys.path.append(os.getcwd())

from src.utils.config import load_config
from src.models.factory import get_model
from src.utils.normalization import ParameterNormalizer
from data.loaders.simulation import generate_single_sample, generate_grid_dataset
from src.inversion.forward_model import compute_hyperbolic_phase, wrap_phase

def visualize_reconstruction(experiment_dir, num_samples=5):
    # 1. Load Config and Model
    config_path = os.path.join(experiment_dir, "checkpoints", "latest_checkpoint.pth")
    if not os.path.exists(config_path):
        print(f"Error: Checkpoint not found at {config_path}")
        return

    checkpoint = torch.load(config_path, map_location='cpu')
    config = checkpoint['config']
    
    model_config = config['model']
    model_config['output_dim'] = 5  # Ensure 5-param
    model = get_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. Setup Normalizer
    normalizer = None
    if config.get("standardize_outputs", False):
        # We need the ranges from data.yaml usually, but Trainer stores them.
        # For simplicity, let's assume standard ranges or extract from config if possible.
        # Trainer.py saves these ranges into the checkpoint config usually if we 
        # updated it. Let's try to reconstruct it.
        # Default ranges if missing:
        ranges = {
            'xc': [-500.0, 500.0],
            'yc': [-500.0, 500.0],
            'S': [1.0, 40.0],
            'wavelength': [0.4, 0.7],
            'focal_length': [10.0, 100.0]
        }
        # Override with config if present
        data_cfg = load_config("configs/data.yaml")
        ranges['S'] = data_cfg.get('S_range', ranges['S'])
        ranges['wavelength'] = data_cfg.get('wavelength_range', ranges['wavelength'])
        ranges['focal_length'] = data_cfg.get('focal_length_range', ranges['focal_length'])
        
        normalizer = ParameterNormalizer(ranges)

    # 3. Generate 5 Samples
    N = config['model'].get("resolution", 256)
    
    
    # We generate a few distinct samples: (xc, yc, S, wavelength, focal_length)
    samples = []
    # Sample 1: Center, small window
    samples.append((0.0, 0.0, 20.0, 0.55, 50.0))
    # Sample 2: Offset, medium window
    samples.append((200.0, -100.0, 30.0, 0.45, 30.0))
    # Sample 3: Large window
    samples.append((-300.0, 300.0, 40.0, 0.65, 80.0))
    # Sample 4: Small window, low wavelength
    samples.append((50.0, 50.0, 15.0, 0.41, 20.0))
    # Sample 5: High focal length
    samples.append((-150.0, -250.0, 25.0, 0.58, 95.0))
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    plt.subplots_adjust(hspace=0.4)

    for i in range(num_samples):
        xc, yc, S, wl, fl = samples[i]
        
        # Generate Ground Truth Phase using S as window size
        x_coords = np.linspace(xc - S / 2.0, xc + S / 2.0, N, dtype=np.float32)
        y_coords = np.linspace(yc - S / 2.0, yc + S / 2.0, N, dtype=np.float32)
        X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
        
        phi_gt_unwrapped = compute_hyperbolic_phase(X_grid, Y_grid, fl, wl)
        phi_gt = wrap_phase(phi_gt_unwrapped)
        
        # Prepare input for model
        input_data, _ = generate_single_sample(N, xc, yc, S, fl, wl)
        # input is (H, W, 2), convert to (B, 2, H, W)
        input_tensor = torch.from_numpy(np.transpose(input_data, (2, 0, 1))).unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            pred_norm = model(input_tensor)
            if normalizer:
                pred = normalizer.denormalize_tensor(pred_norm).squeeze(0).numpy()
            else:
                pred = pred_norm.squeeze(0).numpy()
        
        p_xc, p_yc, p_S, p_wl, p_fl = pred
        
        # Reconstruct Phase from Prediction using predicted S
        x_pred = np.linspace(p_xc - p_S / 2.0, p_xc + p_S / 2.0, N, dtype=np.float32)
        y_pred = np.linspace(p_yc - p_S / 2.0, p_yc + p_S / 2.0, N, dtype=np.float32)
        X_pred, Y_pred = np.meshgrid(x_pred, y_pred)
        phi_pred_unwrapped = compute_hyperbolic_phase(X_pred, Y_pred, p_fl, p_wl)
        phi_pred = wrap_phase(phi_pred_unwrapped)
        
        # Error Map
        error_map = np.abs(phi_gt - phi_pred)
        # Handle wrap around error? (pi - (-pi) = 2pi error but physically 0)
        error_map = np.minimum(error_map, 2*np.pi - error_map)

        # Plotting
        ax_gt = axes[i, 0]
        im_gt = ax_gt.imshow(phi_gt, extent=[-S/2, S/2, -S/2, S/2], cmap='twilight')
        ax_gt.set_title(f"GT Phase\n$\\lambda$={wl*1000:.0f}nm, f={fl:.1f}$\\mu$m, S={S:.1f}$\\mu$m")
        plt.colorbar(im_gt, ax=ax_gt)

        ax_pred = axes[i, 1]
        im_pred = ax_pred.imshow(phi_pred, extent=[-p_S/2, p_S/2, -p_S/2, p_S/2], cmap='twilight')
        ax_pred.set_title(f"Pred Phase\n$\lambda$={p_wl*1000:.0f}nm, f={p_fl:.1f}$\mu$m, S={p_S:.1f}$\mu$m")
        plt.colorbar(im_pred, ax=ax_pred)

        ax_err = axes[i, 2]
        im_err = ax_err.imshow(error_map, extent=[-window_size/2, window_size/2, -window_size/2, window_size/2], cmap='hot')
        ax_err.set_title("Absolute Error Map")
        plt.colorbar(im_err, ax=ax_err)

    save_dir = os.path.join(experiment_dir, "visualizations")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "reconstruction_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Reconstruction comparison saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, required=True)
    args = parser.parse_args()
    visualize_reconstruction(args.experiment_dir)
