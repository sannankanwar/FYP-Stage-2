import argparse
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import json
from tqdm import tqdm

# Ensure src is importable
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import load_config
from src.models.factory import get_model
from data.loaders.simulation import generate_grid_dataset
from src.utils.model_utils import process_predictions
from src.utils.normalization import ParameterNormalizer

def evaluate_metrics(checkpoint_path, output_json_path, device="cpu", steps=25):
    print(f"Loading checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    config = checkpoint.get('config')
    if not config:
        print("Error: Checkpoint does not contain configuration 'config'.")
        return

    model_conf = config.copy()
    if 'model' in config:
        model_conf.update(config['model'])
    
    if 'name' not in model_conf and 'type' in model_conf:
        model_conf['name'] = model_conf['type']
        
    model = get_model(model_conf)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    resolution = config.get("resolution")
    if not resolution:
        resolution = 1024
            
    # Generate Grid
    xc_range = tuple(config.get("xc_range", (-500.0, 500.0)))
    yc_range = tuple(config.get("yc_range", (-500.0, 500.0)))
    S_range = tuple(config.get("S_range", (1.0, 40.0)))
    wavelength_range = tuple(config.get("wavelength_range", (0.4, 0.7)))
    focal_length_range = tuple(config.get("focal_length_range", (10.0, 100.0)))

    print(f"Generating Grid ({steps}x{steps})...")
    X, y, metadata = generate_grid_dataset(
        xc_count=steps,
        yc_count=steps,
        xc_range=xc_range,
        yc_range=yc_range,
        S_range=S_range,
        wavelength_range=wavelength_range,
        focal_length_range=focal_length_range,
        N=resolution,
        grid_strategy="mean"
    )
    
    normalizer = None
    if config.get("standardize_outputs", False):
        norm_ranges = {
            'xc': xc_range,
            'yc': yc_range,
            'S': S_range,
            'wavelength': wavelength_range,
            'focal_length': focal_length_range
        }
        normalizer = ParameterNormalizer(norm_ranges)
    
    X = np.transpose(X, (0, 3, 1, 2))
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)
    
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    
    predictions = []
    with torch.no_grad():
        for inputs, targets in tqdm(loader):
            inputs = inputs.to(device)
            preds = model(inputs)
            predictions.append(preds.cpu())
            
    predictions = torch.cat(predictions, dim=0)
    predictions = process_predictions(model, predictions, normalizer, config)
    
    # Calculate Per-Parameter Metrics
    # y shape: (N, 5)
    # params: ['xc', 'yc', 'S', 'wavelength', 'focal_length']
    param_names = ['xc', 'yc', 'S', 'wavelength', 'focal_length']
    metrics = {}
    
    y_np = y
    pred_np = predictions.numpy()
    
    for i, param in enumerate(param_names):
        true_vals = y_np[:, i]
        pred_vals = pred_np[:, i]
        
        mse = np.mean((true_vals - pred_vals)**2)
        correlation = np.corrcoef(true_vals, pred_vals)[0, 1]
        
        metrics[param] = {
            "mse": float(mse),
            "correlation": float(correlation)
        }
        
    metrics['overall_mse'] = float(np.mean((y_np - pred_np)**2))
    metrics['experiment'] = os.path.basename(os.path.dirname(os.path.dirname(checkpoint_path)))
    
    with open(output_json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Metrics saved to {output_json_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs='+', required=True, help="List of experiment directories")
    parser.add_argument("--output_file", type=str, default="combined_metrics.json")
    parser.add_argument("--device", type=str, default="cpu") # Use CPU to be safe/background friendly
    
    args = parser.parse_args()
    
    all_metrics = []
    
    for exp_dir in args.experiments:
        checkpoint_path = os.path.join(exp_dir, "checkpoints", "best_model.pth")
        if not os.path.exists(checkpoint_path):
            print(f"Skipping {exp_dir}, no best_model.pth")
            continue
            
        temp_json = os.path.join(exp_dir, "temp_metrics.json")
        evaluate_metrics(checkpoint_path, temp_json, device=args.device)
        
        if os.path.exists(temp_json):
            with open(temp_json, 'r') as f:
                all_metrics.append(json.load(f))
            os.remove(temp_json)
            
    with open(args.output_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)

if __name__ == "__main__":
    main()
