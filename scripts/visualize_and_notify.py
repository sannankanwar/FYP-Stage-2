
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import subprocess
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.factory import get_model
from src.utils.config import load_config
from data.loaders.simulation import generate_single_sample, get_2channel_representation
from src.inversion.forward_model import compute_hyperbolic_phase, wrap_phase

def get_latest_checkpoint(exp_dir):
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    if not os.path.exists(ckpt_dir):
        return None
    
    # Try best_model first
    best = os.path.join(ckpt_dir, "best_model.pth")
    if os.path.exists(best):
        return best
    
    # Try latest_checkpoint
    latest = os.path.join(ckpt_dir, "latest_checkpoint.pth")
    if os.path.exists(latest):
        return latest
        
    return None

def load_model_from_checkpoint(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config')
    
    if not config:
        print("Config not found in checkpoint.")
        return None, None
        
    # Reconstruct model config
    model_conf = config.copy()
    if 'model' in config:
        model_conf.update(config['model'])
    if 'name' not in model_conf and 'type' in model_conf:
        model_conf['name'] = model_conf['type']
        
    model = get_model(model_conf)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config

def generate_test_cases():
    # 5 Specific Cases
    # [xc, yc, S, wl, f]
    cases = [
        {"name": "Center", "params": [0.0, 0.0, 20.0, 0.532, 50.0]},
        {"name": "Offset 30", "params": [30.0, 30.0, 20.0, 0.532, 50.0]},
        {"name": "Offset 75", "params": [75.0, 75.0, 20.0, 0.532, 50.0]},
        {"name": "Offset 250", "params": [250.0, 250.0, 20.0, 0.532, 50.0]},
        {"name": "Custom Params", "params": [250.0, 250.0, 40.0, 0.650, 90.0]}
    ]
    return cases

def visualize_case(model, case, output_dir, device, resolution=1024):
    name = case["name"]
    params = case["params"]
    xc, yc, S, wl, f = params
    
    # 1. Generate Truth (using simulation loader directly)
    # generate_single_sample returns (H,W,2), target(5,)
    inp, tgt = generate_single_sample(resolution, xc, yc, S, f, wl)
    
    # Extract True Phase for Plot
    # inp is (H,W,2) -> atan2(sin, cos)
    true_phase = np.arctan2(inp[..., 1], inp[..., 0])
    
    # 2. Run Inference
    # Model expects (1, 2, H, W)
    inp_tensor = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        pred_params = model(inp_tensor).cpu().numpy()[0]
        
    # 3. Generate Predicted Phase
    # Use forward model physics
    pxc, pyc, pS, pwl, pf = pred_params
    
    x_coords = np.linspace(pxc - pS/2.0, pxc + pS/2.0, resolution)
    y_coords = np.linspace(pyc - pS/2.0, pyc + pS/2.0, resolution)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
    
    phi_unwrapped = compute_hyperbolic_phase(X_grid, Y_grid, pf, pwl)
    pred_phase = wrap_phase(phi_unwrapped) # numpy version
    
    # 4. Residual (Circular Difference)
    # diff = angle(exp(i*(true - pred)))
    residual = np.angle(np.exp(1j * (true_phase - pred_phase)))
    
    # 5. Plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Truth
    im0 = axes[0].imshow(true_phase, cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[0].set_title(f"True Phase\nxc={xc:.1f}, yc={yc:.1f}")
    plt.colorbar(im0, ax=axes[0])
    
    # Pred
    im1 = axes[1].imshow(pred_phase, cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[1].set_title(f"Pred Phase\nxc={pxc:.1f}, yc={pyc:.1f}")
    plt.colorbar(im1, ax=axes[1])
    
    # Residual
    im2 = axes[2].imshow(residual, cmap='coolwarm', vmin=-np.pi, vmax=np.pi)
    axes[2].set_title(f"Residual Phase\nMSE(params)={np.mean((np.array(params)-pred_params)**2):.4f}")
    plt.colorbar(im2, ax=axes[2])

    # Text Info
    text = (
        f"CASE: {name}\n"
        f"TRUE: {params}\n"
        f"PRED: {np.round(pred_params, 3)}"
    )
    axes[3].axis('off')
    axes[3].text(0.1, 0.5, text, fontsize=12, va='center')

    plt.tight_layout()
    filename = name.lower().replace(" ", "_").replace(":", "") + ".png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close()
    
    return save_path, text

def plot_scatter_evaluation(model, output_dir, device, resolution=256, samples=100):
    true_vals = []
    pred_vals = []
    
    print(f"Generating {samples} random samples for scatter plot...")
    
    # We can rely on generate_single_sample but randomized
    for _ in range(samples):
        # Sample randomly from typical ranges
        xc = np.random.uniform(-100, 100) # Using config ranges would be better but this is a proxy
        yc = np.random.uniform(-100, 100)
        S = np.random.uniform(5, 40)
        wl = np.random.uniform(0.4, 0.7)
        f = np.random.uniform(10, 100)
        
        inp, tgt = generate_single_sample(resolution, xc, yc, S, f, wl)
        
        inp_tensor = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).float().to(device)
        with torch.no_grad():
             pred = model(inp_tensor).cpu().numpy()[0]
             
        true_vals.append([xc, yc, S, wl, f])
        pred_vals.append(pred)
        
    true_vals = np.array(true_vals)
    pred_vals = np.array(pred_vals)
    
    labels = ['xc', 'yc', 'S', 'wl', 'f']
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    for i, ax in enumerate(axes):
        t = true_vals[:, i]
        p = pred_vals[:, i]
        
        ax.scatter(t, p, alpha=0.6)
        
        # Identity
        mn = min(t.min(), p.min())
        mx = max(t.max(), p.max())
        ax.plot([mn, mx], [mn, mx], 'r--')
        
        mse = np.mean((t-p)**2)
        ax.set_title(f"{labels[i]} (MSE={mse:.4f})")
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")
        
    plt.tight_layout()
    save_path = os.path.join(output_dir, "scatter_summary.png")
    plt.savefig(save_path)
    plt.close()
    
    return save_path

def notify_user(image_path, title, body=""):
    notify_script = os.path.join(os.path.dirname(__file__), "notify.py")
    
    # Source secrets if available? 
    # Actually, subprocess inherits env vars, so if run from queue script which sourced secrets, we are good.
    # Otherwise, need to source manually?
    
    # We assume secrets are in env
    cmd = [
        "python", notify_script,
        title,
        body,
        "--image-file", image_path
    ]
    
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    
    exp_dir = args.experiment_dir
    exp_name = os.path.basename(exp_dir)
    print(f"Visualizing Experiment: {exp_name}")
    
    ckpt = get_latest_checkpoint(exp_dir)
    if not ckpt:
        print("No checkpoint found.")
        return
        
    model, config = load_model_from_checkpoint(ckpt, args.device)
    if not model:
        print("Failed to load model.")
        return
        
    # Create Vis Dir
    vis_dir = os.path.join(exp_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # 1. 5 Test Cases
    cases = generate_test_cases()
    for case in cases:
        print(f"Running Case: {case['name']}")
        try:
            path, text = visualize_case(model, case, vis_dir, args.device, resolution=config.get("resolution", 1024))
            notify_user(path, f"[{exp_name}] Case: {case['name']}", text)
        except Exception as e:
            print(f"Failed case {case['name']}: {e}")
            
    # 2. Scatter Plot
    try:
        scatter_path = plot_scatter_evaluation(model, vis_dir, args.device, samples=args.samples)
        notify_user(scatter_path, f"[{exp_name}] Scatter Evaluation", f"Random {args.samples} samples validation.")
    except Exception as e:
        print(f"Failed scatter plot: {e}")

if __name__ == "__main__":
    main()
