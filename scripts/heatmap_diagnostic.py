
import argparse
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.config import load_config
from src.models.factory import get_model
from data.loaders.simulation import generate_single_sample

def main():
    parser = argparse.ArgumentParser(description="Generate Difficulty Heatmap")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="diagnostic_heatmaps")
    parser.add_argument("--resolution", type=int, default=10, help="Grid samples per dim")
    args = parser.parse_args()
    
    conf = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(conf).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    
    # 2. Define Grid
    # Fixed Params
    f_fixed = 50.0  # approximate mean
    wl_fixed = 0.55 # approximate mean
    
    # Varying Params
    steps = args.resolution
    xc_grid = np.linspace(-100, 100, steps)
    yc_grid = np.linspace(-100, 100, steps)
    S_grid = np.linspace(5, 50, steps)
    
    # Container for error tensor: (steps, steps, steps) -> (xc, yc, S)
    error_volume = np.zeros((steps, steps, steps))
    samples_per_point = 5
    img_res = conf.get("resolution", 256)
    
    print(f"Scanning {steps}x{steps}x{steps} = {steps**3} grid points...")
    
    with torch.no_grad():
        for i, xc in enumerate(tqdm(xc_grid)):
            for j, yc in enumerate(yc_grid):
                for k, S in enumerate(S_grid):
                    
                    batch_loss = 0.0
                    
                    for _ in range(samples_per_point):
                        # Generate
                        inp_np, tgt_np = generate_single_sample(
                            N=img_res, xc=xc, yc=yc, S=S, 
                            focal_length=f_fixed, wavelength=wl_fixed, noise_std=0.0
                        )
                        
                        # Forward
                        inp_t = torch.from_numpy(inp_np).unsqueeze(0).permute(0,3,1,2).to(device)
                        tgt_t = torch.tensor([xc, yc, S, wl_fixed, f_fixed]).to(device).unsqueeze(0)
                        
                        pred = model(inp_t)
                        
                        # Loss (MSE on params)
                        # We focus on xc, yc, S error specifically? Or all?
                        # Let's measure ALL param error, but f/wl are fixed so their error contributes too
                        loss = torch.mean((pred - tgt_t)**2).item()
                        batch_loss += loss
                        
                    error_volume[i, j, k] = batch_loss / samples_per_point
    
    # 3. Visualization (Slices)
    # Slices at low, mid, high S
    indices = [0, steps//2, steps-1]
    s_values = [S_grid[i] for i in indices]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (s_idx, s_val) in enumerate(zip(indices, s_values)):
        ax = axes[idx]
        # Slice: (xc, yc) at fixed S
        # Matrix shape: [xc, yc]
        # Transpose for plotting? imshow origin is usually top-left.
        # xc horizontal, yc vertical?
        slice_data = error_volume[:, :, s_idx].T # Transpose to map xc->x, yc->y
        
        im = ax.imshow(
            slice_data, 
            origin='lower', 
            extent=[xc_grid[0], xc_grid[-1], yc_grid[0], yc_grid[-1]],
            cmap='magma', 
            interpolation='nearest'
        )
        ax.set_title(f"Mean MSE @ S={s_val:.1f}")
        ax.set_xlabel("xc")
        ax.set_ylabel("yc")
        plt.colorbar(im, ax=ax, fraction=0.046)
        
    plt.suptitle("Difficulty Heatmap (MSE)")
    save_path = os.path.join(args.output_dir, "heatmap_slices.png")
    plt.savefig(save_path)
    print(f"Saved heatmap to {save_path}")
    
    # Save raw data
    np.save(os.path.join(args.output_dir, "error_volume.npy"), error_volume)

if __name__ == "__main__":
    main()
