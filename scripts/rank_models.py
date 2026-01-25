"""
Rank Models based on systematic Grid Evaluation.
Generates a ranking table based on Total Physics Residual.
"""
import torch
import numpy as np
import pandas as pd
import os
import glob
import sys
import argparse
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import load_config
from src.models.factory import get_model
from data.loaders.simulation import generate_single_sample

def evaluate_model_on_grid(model, config, device='cpu', grid_resolution=3):
    """
    Evaluate model on a structured grid of parameters.
    grid_resolution: number of points per parameter dim (e.g. 3 -> low, mid, high)
    
    3^5 = 243 test cases.
    """
    model.eval()
    
    # Define grid points
    # xc, yc: [-500, 0, 500]
    # fov: [1, 10, 20]
    # wl: [0.4, 0.55, 0.7]
    # fl: [10, 55, 100]
    
    xcs = np.linspace(-500, 500, grid_resolution)
    ycs = np.linspace(-500, 500, grid_resolution)
    fovs = np.linspace(1, 20, grid_resolution)
    wls = np.linspace(0.4, 0.7, grid_resolution)
    fls = np.linspace(10, 100, grid_resolution)
    
    # Create mesh grid
    mesh = np.array(np.meshgrid(xcs, ycs, fovs, wls, fls)).T.reshape(-1, 5)
    
    total_residual = 0.0
    residuals_per_param = np.zeros(5)
    
    resolution = config.get("resolution", 256)
    window_size = config.get("window_size", 100.0)
    
    # Batch processing could be faster, but let's do loop for simplicity/memory
    # or small batches.
    
    batch_size = 16
    num_batches = int(np.ceil(len(mesh) / batch_size))
    
    with torch.no_grad():
        for i in range(num_batches):
            batch_params = mesh[i*batch_size : (i+1)*batch_size]
            
            # Generate Inputs
            inputs = []
            for p in batch_params:
                inp, _ = generate_single_sample(
                    N=resolution, xc=p[0], yc=p[1], fov=p[2],
                    wavelength=p[3], focal_length=p[4], window_size=window_size
                )
                inputs.append(inp.transpose(2,0,1))
            
            input_tensor = torch.tensor(np.array(inputs), dtype=torch.float32).to(device)
            
            # Predict
            pred_raw = model(input_tensor)
            
            # Hybrid models output raw. Check for standardization config just in case.
            # We assume NO standardization for Phase B.
            preds = pred_raw.cpu().numpy()
            
            # Compute Errors
            # MSE per sample per param
            diff = (preds - batch_params) ** 2
            residuals_per_param += diff.sum(axis=0)
            total_residual += diff.sum()
            
    # Normalize
    N = len(mesh)
    mean_mse_per_param = residuals_per_param / N
    mean_total_mse = total_residual / N
    
    return mean_total_mse, mean_mse_per_param

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
        
    exp_dirs = sorted(glob.glob(os.path.join(args.output_dir, "exp*")))
    
    results = []
    
    print(f"Ranking {len(exp_dirs)} models using Grid Search (3^5=243 patterns)...")
    
    for d in tqdm(exp_dirs):
        exp_name = os.path.basename(d)
        
        # Load Config
        # Try checkpoint first
        ckpt_path = os.path.join(d, "checkpoints", "best_model.pth")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(d, "checkpoints", "latest_checkpoint.pth")
            
        if not os.path.exists(ckpt_path):
            print(f"Skipping {exp_name}: No checkpoint")
            continue
            
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            if 'config' in checkpoint:
                config = checkpoint['config']
            else:
                # Fallback to file
                 config = load_config(os.path.join(d, "config.yaml"))
        except:
            print(f"Skipping {exp_name}: Config load failed")
            continue
            
        # Load Model
        try:
            model = get_model(config)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
        except Exception as e:
            print(f"Skipping {exp_name}: Model load failed {e}")
            continue
            
        # Evaluate
        score, param_scores = evaluate_model_on_grid(model, config, device)
        
        res = {
            'Model': exp_name,
            'Total_MSE': score,
            'MSE_xc': param_scores[0],
            'MSE_yc': param_scores[1],
            'MSE_fov': param_scores[2],
            'MSE_wl': param_scores[3],
            'MSE_fl': param_scores[4],
        }
        results.append(res)
        
    # Create DataFrame
    df = pd.DataFrame(results)
    if df.empty:
        print("No results generated.")
        return
        
    df = df.sort_values("Total_MSE") # Ascending (lower is better)
    df['Rank'] = range(1, len(df) + 1)
    
    # Reorder columns
    cols = ['Rank', 'Model', 'Total_MSE', 'MSE_xc', 'MSE_yc', 'MSE_fov', 'MSE_wl', 'MSE_fl']
    df = df[cols]
    
    print("\n=== Model Ranking ===")
    print(df.to_string(index=False))
    
    csv_path = os.path.join(args.output_dir, "model_ranking.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved ranking to {csv_path}")

if __name__ == "__main__":
    main()
