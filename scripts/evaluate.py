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

from src.utils.config import load_config
from src.models.factory import get_model
from data.loaders.simulation import generate_grid_dataset
from src.utils.model_utils import replace_activation
from src.utils.normalization import ParameterNormalizer

def visualize_sample(input_tensor, true_params, pred_params, loss, title, output_path):
    """
    Visualizes a single sample: Phase Map + Parameter Info
    input_tensor: (2, H, W)
    """
    # Compute Phase from (Cos, Sin)
    # input_tensor is numpy array or tensor on cpu
    if isinstance(input_tensor, torch.Tensor):
        input_tensor = input_tensor.numpy()
        
    cos_phi = input_tensor[0]
    sin_phi = input_tensor[1]
    phase = np.arctan2(sin_phi, cos_phi)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(phase, cmap='hsv', origin='lower') # HSV is good for phase
    plt.colorbar(label='Phase (rad)')
    plt.title(f"{title}\nLoss: {loss:.6f}")
    
    # Text info
    if len(true_params) == 5:
        info_text = (
            f"TRUE: xc={true_params[0]:.2f}, yc={true_params[1]:.2f}, fov={true_params[2]:.2f}\n"
            f"      wl={true_params[3]*1e9:.1f}nm, f={true_params[4]*1e6:.1f}um\n"
            f"PRED: xc={pred_params[0]:.2f}, yc={pred_params[1]:.2f}, fov={pred_params[2]:.2f}\n"
            f"      wl={pred_params[3]*1e9:.1f}nm, f={pred_params[4]*1e6:.1f}um"
        )
    else:
        info_text = (
            f"TRUE: xc={true_params[0]:.2f}, yc={true_params[1]:.2f}, fov={true_params[2]:.2f}\n"
            f"PRED: xc={pred_params[0]:.2f}, yc={pred_params[1]:.2f}, fov={pred_params[2]:.2f}"
        )
    plt.figtext(0.5, 0.05, info_text, ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_scatter(y_true, y_pred, output_dir, title="Parameter Scatter Plots"):
    """
    Plots True vs Predicted scatter plots for each parameter (xc, yc, fov).
    y_true, y_pred: (N, 3) arrays
    """
    num_params = y_true.shape[1]
    if num_params == 5:
        params = ['xc', 'yc', 'fov', 'wavelength', 'focal_length']
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    else:
        params = ['xc', 'yc', 'fov']
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, param in enumerate(params):
        ax = axes[i]
        true_vals = y_true[:, i]
        pred_vals = y_pred[:, i]
        
        # Calculate R2 or correlation
        correlation = np.corrcoef(true_vals, pred_vals)[0, 1]
        mse = np.mean((true_vals - pred_vals)**2)
        
        ax.scatter(true_vals, pred_vals, alpha=0.5, s=1)
        
        # Plot identity line
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
        
        ax.set_xlabel(f'True {param}')
        ax.set_ylabel(f'Predicted {param}')
        ax.set_title(f'{param} (R={correlation:.3f}, MSE={mse:.2f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.suptitle(title)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "scatter_plots.png")
    plt.savefig(output_path)
    print(f"Scatter plots saved to {output_path}")
    plt.close()

def get_subdirectories(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def select_experiments(base_dir="outputs"):
    if not os.path.exists(base_dir):
        print(f"Directory '{base_dir}' not found.")
        return []
        
    experiments = sorted(get_subdirectories(base_dir))
    if not experiments:
        print("No experiments found.")
        return []
        
    print("\nAvailable Experiments:")
    for i, exp in enumerate(experiments):
        print(f"{i+1}. {exp}")
        
    selection = input("\nEnter experiment numbers (comma-separated, e.g., 1,3) or 'all': ").strip()
    
    selected_exps = []
    if selection.lower() == 'all':
        return [os.path.join(base_dir, exp) for exp in experiments]
    
    try:
        indices = [int(x.strip()) - 1 for x in selection.split(',')]
        for idx in indices:
            if 0 <= idx < len(experiments):
                selected_exps.append(os.path.join(base_dir, experiments[idx]))
            else:
                print(f"Warning: Index {idx+1} out of range. Skipping.")
    except ValueError:
        print("Invalid input. Please enter numbers.")
        
    return selected_exps

def select_resolution():
    print("\nSelect Grid Resolution:")
    print("1. 10x10 (Fast)")
    print("2. 25x25 (medium)")
    print("3. 50x50 (Detailed)")
    print("4. Custom")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == '1': return 10
    if choice == '2': return 25
    if choice == '3': return 50
    if choice == '4':
        try:
            return int(input("Enter custom resolution (e.g. 20): ").strip())
        except ValueError:
            print("Invalid number. Defaulting to 25.")
            return 25
    
    print("Invalid choice. Defaulting to 25.")
    return 25

def evaluate_grid(checkpoint_path, output_dir, device="cpu", steps=25):
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
    # Instantiate Model
    # Handle potentially nested model config (e.g. from exp13 yaml where it's under 'model')
    model_conf = config.copy()
    if 'model' in config:
        print("Flattening 'model' section from config...")
        model_conf.update(config['model'])
    
    # Map 'type' to 'name' if 'name' is missing (common alias in our yamls)
    if 'name' not in model_conf and 'type' in model_conf:
        model_conf['name'] = model_conf['type']
        
    print(f"Model Factory Config: {model_conf}")
    model = get_model(model_conf)
    
    # Load Weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # If resolution is missing in config, try to check configs/data.yaml or default to 1024 (project default)
    resolution = config.get("resolution")
    if not resolution:
        # Try loading data.yaml
        data_yaml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "data.yaml")
        if os.path.exists(data_yaml_path):
             try:
                 data_conf = load_config(data_yaml_path)
                 resolution = data_conf.get("resolution", 1024)
                 print(f"Loaded resolution {resolution} from {data_yaml_path}")
             except:
                 resolution = 1024
        else:
            resolution = 1024
            
    print(f"Using Model Resolution: {resolution}x{resolution}")

    # 2. Generate Grid Dataset
    print(f"Generating Evaluation Grid ({steps}x{steps})...")
    # Pull ranges from config or defaults
    xc_range = tuple(config.get("xc_range", (-500.0, 500.0)))
    yc_range = tuple(config.get("yc_range", (-500.0, 500.0)))
    fov_range = tuple(config.get("fov_range", (1.0, 20.0))) # Support 5p range
    wavelength_range = tuple(config.get("wavelength_range", (400e-9, 700e-9)))
    focal_length_range = tuple(config.get("focal_length_range", (10e-6, 100e-6)))

    X, y, metadata = generate_grid_dataset(
        xc_count=steps,
        yc_count=steps,
        xc_range=xc_range,
        yc_range=yc_range,
        fov_range=fov_range,
        wavelength_range=wavelength_range,
        focal_length_range=focal_length_range,
        N=resolution,
        grid_strategy="mean" # Consistent with training anchor
    )
    
    # Initialize Normalizer if needed
    normalizer = None
    if config.get("standardize_outputs", False):
        print("Detected Standardized Model: Initializing Normalizer...")
        # Normalizer needs ranges. We use the evaluation ranges here since that's what we are feeding in?
        # Ideally, we should use the ranges the model was TRAINED on. 
        # But for OnTheFly, the training ranges are in the config.
        # Let's try to pull from config if available, else use defaults.
        
        norm_ranges = {
            'xc': tuple(config.get("xc_range", (-500.0, 500.0))),
            'yc': tuple(config.get("yc_range", (-500.0, 500.0))),
            'fov': tuple(config.get("fov_range", (10.0, 80.0))),
            'wavelength': tuple(config.get("wavelength_range", (400e-9, 700e-9))),
            'focal_length': tuple(config.get("focal_length_range", (10e-6, 100e-6)))
        }
        normalizer = ParameterNormalizer(norm_ranges)
    
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
    
    # Denormalize if needed
    if normalizer:
        print("Denormalizing predictions...")
        # Only denormalize the first 3 (xc, yc, fov)
        # Assuming predictions might have more columns later? 
        # But normalizer works on shape (B, 3) usually.
        # The model output is (B, 3).
        predictions = normalizer.denormalize_tensor(predictions)
    
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
    
    plt.title(f"Loss Heatmap (FOV={metadata['fov']:.1f})\n{config.get('experiment_name', 'Unknown Experiment')}")
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

    # 6. Scatter Plots
    print("Generating Scatter Plots...")
    plot_scatter(y, predictions.numpy(), output_dir)
    
    # 7. Find and Save Best/Worst Cases

    # 6. Find and Save Best/Worst Cases
    min_idx = np.argmin(mse_per_sample)
    max_idx = np.argmax(mse_per_sample)
    
    print(f"Saving Best Case (Index {min_idx}, Loss {mse_per_sample[min_idx]:.6f})...")
    visualize_sample(
        X[min_idx], 
        y[min_idx], 
        predictions[min_idx].numpy(), 
        mse_per_sample[min_idx],
        "Best Case Prediction",
        os.path.join(output_dir, "best_case.png")
    )
    
    print(f"Saving Worst Case (Index {max_idx}, Loss {mse_per_sample[max_idx]:.6f})...")
    visualize_sample(
        X[max_idx], 
        y[max_idx], 
        predictions[max_idx].numpy(), 
        mse_per_sample[max_idx],
        "Worst Case Prediction",
        os.path.join(output_dir, "worst_case.png")
    )

def main():
    parser = argparse.ArgumentParser(description="Evaluate model on a dense grid and generate heatmap.")
    parser.add_argument("--experiment_dir", type=str, help="Path to experiment directory (optional, for non-interactive mode)")
    parser.add_argument("--checkpoint", type=str, default="best_model.pth", help="Name of checkpoint file (default: best_model.pth)")
    parser.add_argument("--steps", type=int, default=25, help="Grid resolution (steps x steps). Default 25.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Check if arguments provided for non-interactive mode
    if args.experiment_dir:
        checkpoint_path = os.path.join(args.experiment_dir, "checkpoints", args.checkpoint)
        # Check direct path logic...
        if not os.path.exists(checkpoint_path):
             if os.path.exists(args.experiment_dir) and args.experiment_dir.endswith(".pth"):
                 checkpoint_path = args.experiment_dir
                 args.experiment_dir = os.path.dirname(os.path.dirname(checkpoint_path))
        
        evaluate_grid(checkpoint_path, args.experiment_dir, device=args.device, steps=args.steps)
    else:
        # Interactive Mode
        print("=== Interactive Evaluation Mode ===")
        selected_exps = select_experiments()
        if not selected_exps:
            print("No experiments selected. Exiting.")
            return
            
        resolution = select_resolution()
        
        print(f"\nProcessing {len(selected_exps)} experiments with resolution {resolution}x{resolution}...")
        
        for exp_dir in selected_exps:
            print(f"\n--- Evaluating {os.path.basename(exp_dir)} ---")
            checkpoint_path = os.path.join(exp_dir, "checkpoints", args.checkpoint)
            
            if not os.path.exists(checkpoint_path):
                # Try latest if best not found? 
                latest_path = os.path.join(exp_dir, "checkpoints", "latest_checkpoint.pth")
                if os.path.exists(latest_path):
                     print(f"Note: '{args.checkpoint}' not found, using 'latest_checkpoint.pth' instead.")
                     checkpoint_path = latest_path
                else:
                    print(f"Skipping {exp_dir}: Checkpoint not found.")
                    continue
            
            evaluate_grid(checkpoint_path, exp_dir, device=args.device, steps=resolution)
            
        print("\nAll evaluations completed.")

if __name__ == "__main__":
    main()
