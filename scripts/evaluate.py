import argparse
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Ensure src is importable
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.factory import get_model
from data.loaders.simulation import generate_grid_dataset
from src.utils.model_utils import replace_activation

def evaluate_grid(checkpoint_path, output_dir, device="cpu", steps=50):
    """
    Evaluates the model on a dense grid of (xc, yc) coordinates and plots a heatmap of the error.
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False) # weights_only=False to load legacy full checkpoints if any
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # 1. Reconstruct Model
    # We try to use the config saved in the checkpoint if available, otherwise we might need arguments.
    # The Trainer saves 'config' in the checkpoint.
    config = checkpoint.get('config')
    if not config:
        print("Error: Checkpoint does not contain configuration 'config'. Cannot reconstruct model.")
        return

    print(f"Model Config: Name={config.get('name')}, Activation={config.get('activation', 'ReLU (Default)')}")
    
    # Instantiate Model
    model = get_model(config)
    
    # Load Weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # 2. Generate Grid Dataset
    print(f"Generating Evaluation Grid ({steps}x{steps})...")
    # Define range relative to what we trained on? Or standard [-500, 500].
    # Let's use standard range for consistent comparison.
    xc_range = (-500.0, 500.0)
    yc_range = (-500.0, 500.0)
    fixed_fov = 45.0 # Evaluate at a fixed FOV in the middle of range [10, 80]
    
    X, y, metadata = generate_grid_dataset(
        xc_count=steps,
        yc_count=steps,
        fov=fixed_fov,
        xc_range=xc_range,
        yc_range=yc_range,
        N=config.get("resolution", 256)
    )
    
    # X: (N_samples, H, W, 2) -> Need (N_samples, 2, H, W)
    X = np.transpose(X, (0, 3, 1, 2))
    
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)
    
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 3. Predict & Compute Error
    print("Running Inference...")
    predictions = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(loader):
            inputs = inputs.to(device)
            preds = model(inputs)
            predictions.append(preds.cpu())
            
    predictions = torch.cat(predictions, dim=0) # (N_samples, 3)
    
    # Calculate Square Error per sample (Sum of squared errors for xc, yc, fov or just params?)
    # Users usually care about spatial error (xc, yc) mostly. Let's do total parameter MSE.
    mse_per_sample = torch.sum((predictions - y_tensor)**2, dim=1).numpy()
    
    # Reshape to grid
    grid_shape = metadata['grid_shape'] # (xc_count, yc_count)
    mse_grid = mse_per_sample.reshape(grid_shape)
    
    # 4. Plot Heatmap
    plt.figure(figsize=(10, 8))
    
    # Create Heatmap
    # We flip Y axis for visualization to match standard Cartesian coordinates if needed,
    # but imshow default usually puts (0,0) top left. 
    # Let's use extent to map to physical units.
    extent = [xc_range[0], xc_range[1], yc_range[0], yc_range[1]]
    
    im = plt.imshow(mse_grid.T, origin='lower', extent=extent, cmap='viridis', aspect='auto', interpolation='nearest')
    plt.colorbar(im, label='MSE Loss')
    
    plt.title(f"Loss Heatmap (FOV={fixed_fov})\n{config.get('experiment_name', 'Unknown Experiment')}")
    plt.xlabel('XC (micrometers)')
    plt.ylabel('YC (micrometers)')
    
    output_plot_path = os.path.join(output_dir, "evaluation_heatmap.png")
    plt.savefig(output_plot_path)
    print(f"Heatmap saved to {output_plot_path}")
    
    # 5. Save aggregate stats
    avg_loss = np.mean(mse_per_sample)
    print(f"Average Grid Loss: {avg_loss:.4f}")
    
    stats_path = os.path.join(output_dir, "evaluation_stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"Average MSE: {avg_loss}\n")
        f.write(f"Max MSE: {np.max(mse_per_sample)}\n")
        f.write(f"Min MSE: {np.min(mse_per_sample)}\n")

def main():
    parser = argparse.ArgumentParser(description="Evaluate model on a dense grid and generate heatmap.")
    parser.add_argument("--experiment_dir", type=str, required=True, help="Path to experiment directory (e.g., outputs/experiment1)")
    parser.add_argument("--checkpoint", type=str, default="best_model.pth", help="Name of checkpoint file (default: best_model.pth)")
    parser.add_argument("--steps", type=int, default=50, help="Grid resolution (steps x steps)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    checkpoint_path = os.path.join(args.experiment_dir, "checkpoints", args.checkpoint)
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        # Try looking directly if user passed full path
        if os.path.exists(args.experiment_dir) and args.experiment_dir.endswith(".pth"):
             checkpoint_path = args.experiment_dir
             # Infer output dir
             args.experiment_dir = os.path.dirname(os.path.dirname(checkpoint_path))
        else:
            sys.exit(1)
            
    evaluate_grid(checkpoint_path, args.experiment_dir, device=args.device, steps=args.steps)

if __name__ == "__main__":
    main()
