
import os
import sys
import argparse
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import numba
from scipy.optimize import differential_evolution

# Add root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.factory import get_model
from src.inversion.forward_model import compute_hyperbolic_phase, wrap_phase
from data.loaders.simulation import OnTheFlyDataset

# ===================== Numba Accelerated Cost Function =====================

@numba.jit(nopython=True, cache=True, fastmath=True)
def compute_phase_diff_cost(params, X_flat, Y_flat, target_phase_flat, lambda_m):
    """
    Computes absolute phase difference cost for a batch of pixels.
    Params: [scale, focal_length, x0, y0]
    Wavelength (lambda_m) is fixed for this optimization loop or could be optimized too.
    """
    scale, focal_length, x0, y0 = params
    
    # Numba loop for memory efficiency on large images
    n = len(X_flat)
    cost = 0.0
    
    # Precompute constants
    k_const = -np.pi / (lambda_m * focal_length)
    
    for i in range(n):
        # Scale coordinates
        # Logic: The 'design' coordinates are fixed (X_flat), but the physics 'effective' coordinates shift/scale.
        # X_eff = X_design / scale + x0
        # If scale > 1, the pattern zooms in (features get larger? No, pattern covers less area).
        # Let's match user logic: L_zoomed = L / scale.
        
        # User snippet logic: 
        # xs = idx_range * (L/scale)/N + x0
        # Here X_flat is effectively pre-scaled grid. Let's apply transformation.
        x = X_flat[i] / scale + x0
        y = Y_flat[i] / scale + y0
        
        r_sq = x*x + y*y
        
        # Phase: -(π * R²) / (λ * f)  (This is the user's specific approx formula, slightly different from hyperbolic sqrt)
        # Reverting to EXACT Hyperbolic formula if focal length is small: 
        # phi = (2pi/lam) * (sqrt(r^2 + f^2) - f)
        
        # NOTE: User snippet used approx: phase_sim = (-np.pi * r_sq) / (lambda_m * focal_length)
        # This corresponds to Fresnel approx. 
        # Let's use the EXACT formula from our `src/inversion/forward_model.py` to be consistent with our ground truth generation.
        # But we need to JIT it.
        
        val = np.sqrt(r_sq + focal_length*focal_length) - focal_length
        phase_sim = (2.0 * np.pi / lambda_m) * val
        
        # Wrap
        # arctan2(sin, cos) is safest
        phase_sim = np.arctan2(np.sin(phase_sim), np.cos(phase_sim))
        
        # Diff
        diff = phase_sim - target_phase_flat[i]
        
        # Wrap diff to [-pi, pi]
        # (diff + pi) % 2pi - pi
        diff = diff + np.pi
        diff = diff - 2.0*np.pi * np.floor(diff / (2.0*np.pi))
        diff = diff - np.pi
        
        cost += np.abs(diff)
        
    return cost / n # Mean Absolute Error

# ===================== Global Cost Function for Multiprocessing =====================

def cost_func_global(p, X_flat, Y_flat, target_flat):
    # p = [xc, yc, scale, wl, f]
    
    # Unpack DE vector
    xc, yc, scale, wl, f = p
    
    # Call Numba JIT function
    # params order in JIT: [scale, focal_length, x0, y0]
    
    jit_params = np.array([scale, f, xc, yc], dtype=np.float64)
    return compute_phase_diff_cost(jit_params, X_flat, Y_flat, target_flat, wl)

# ===================== Main Script =====================

def main():
    parser = argparse.ArgumentParser(description="Refine Metalens Parameters using Differential Evolution")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Experiment 9 (or similar) checkpoint")
    parser.add_argument("--sample_id", type=int, default=0, help="Index of sample from validation set to refine")
    parser.add_argument("--max_iter", type=int, default=50, help="Max generations for DE")
    parser.add_argument("--pop_size", type=int, default=15, help="Population size for DE")
    parser.add_argument("--config", type=str, help="Path to config file (optional, uses checkpoint's if not provided)")
    parser.add_argument("--output_dir", type=str, default="outputs/refinement_de", help="Directory to save results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Model & Config
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False) # weights_only=False required for full pickle
    
    # Extract config
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    elif 'config' in ckpt:
        config = ckpt['config']
    else:
        raise ValueError("No config found in checkpoint and none provided via --config")

    # Flatten config
    def flatten_config(cfg):
        flat = cfg.copy()
        if 'model' in cfg: flat.update(cfg['model'])
        if 'data' in cfg: flat.update(cfg['data'])
        return flat
    
    flat_config = flatten_config(config)
    
    # 2. Load Validation Data
    print("Loading Validation Dataset...")
    # Force validation mode to get deterministic samples if possible (or seed)
    flat_config['seed'] = 42 # Ensure we get same sample 
    val_dataset = OnTheFlyDataset(flat_config, length=max(args.sample_id + 1, 100))
    
    # Get Sample
    input_tensor, target_params = val_dataset[args.sample_id]
    
    # Input is [Cos, Sin]. Reconstruct Phase Map.
    cos_map = input_tensor[0].numpy()
    sin_map = input_tensor[1].numpy()
    gt_phase_wrapped = np.arctan2(sin_map, cos_map)
    H, W = gt_phase_wrapped.shape
    
    print(f"Loaded Sample {args.sample_id}. Shape: {H}x{W}")
    print(f"Ground Truth Params: {target_params.numpy()}") # [xc, yc, fwd_model_params...]
    
    # 3. Model Prediction (Initial Guess)
    model = get_model(flat_config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    with torch.no_grad():
        input_batch = input_tensor.unsqueeze(0) # 1, C, H, W
        pred_params = model(input_batch).squeeze(0).numpy()
        
    print(f"Initial Prediction: {pred_params}")
    
    # Param Mapping (Experiment Specific)
    # Usually: [xc, yc, fov_scale, wavelength, focal_length] or similar order.
    # Check config for output keys? Assuming standard 5-param model based on convo history.
    # Order usually: xc, yc, scale, wavelength, focal_length (based on `simulation.py`)
    # Let's assume standard order:
    # 0: xc
    # 1: yc
    # 2: fov_scale (or S)
    # 3: wavelength
    # 4: focal_length
    
    # NOTE: user snippet optimizes [scale, focal_length, x0, y0]. Wavelength fixed?
    # Our model predicts 5 params. Let's refine ALL 5.
    
    xc_init, yc_init, scale_init, wl_init, f_init = pred_params
    
    # Bounds (Relaxed around prediction +/- 20%)
    bounds = [
        (pred_params[0] - 5.0, pred_params[0] + 5.0), # xc (um)
        (pred_params[1] - 5.0, pred_params[1] + 5.0), # yc (um)
        (pred_params[2] * 0.8, pred_params[2] * 1.2), # scale
        (pred_params[3] * 0.9, pred_params[3] * 1.1), # wavelength
        (pred_params[4] * 0.8, pred_params[4] * 1.2)  # focal_length
    ]
    
    # 4. Prepare Numba Optimization
    # Create coordinate grid centered at 0 (simulation coordinates)
    # The simulation assumes 0,0 is center of lens.
    # The 'xc, yc' are offsets of the aperture in the simulation space.
    
    # Grid in microns
    L = flat_config.get('physical_size', 100.0) # um 
    x = np.linspace(-L/2, L/2, W)
    y = np.linspace(-L/2, L/2, H)
    X, Y = np.meshgrid(x, y)
    
    X_flat = X.ravel().astype(np.float64)
    Y_flat = Y.ravel().astype(np.float64)
    target_flat = gt_phase_wrapped.ravel().astype(np.float64)

    # 5. Run Differential Evolution
    print("\nStarting Differential Evolution...")
    print(f"Max Iters: {args.max_iter}, Pop Size: {args.pop_size}")
    
    # We pass 'args' via closure? No, must be global or passed as args.
    # scipy differential_evolution allows passing 'args' tuple to cost_func.
    # cost_func(x, *args)
    
    global_args = (X_flat, Y_flat, target_flat)
    
    res = differential_evolution(
        cost_func_global,
        bounds=bounds,
        args=global_args,
        maxiter=args.max_iter,
        popsize=args.pop_size,
        polish=True, # L-BFGS-B at end
        disp=True,
        workers=-1 # All cores
    )
    
    refined_params = res.x
    final_cost = res.fun
    
    print("\n" + "="*30)
    print("Refinement Complete")
    print("="*30)
    print(f"Initial Cost: {cost_func_global(pred_params, X_flat, Y_flat, target_flat):.5f} (approx)")
    print(f"Final Cost:   {final_cost:.5f}")
    
    print("\nParams (Init -> Refined):")
    names = ['xc', 'yc', 'scale', 'wl', 'f']
    for n, p0, p1 in zip(names, pred_params, refined_params):
        print(f"  {n}: {p0:.4f} -> {p1:.4f} (Diff: {p1-p0:.4f})")
        
    print(f"\nGT Params: {target_params.numpy()}")
    
    # 6. Visualize
    # Recompute maps for plotting
    def compute_full_map(p):
        xc, yc, scale, wl, f = p
        # Vectorized numpy (no jit needed for plot)
        X_eff = X / scale + xc
        Y_eff = Y / scale + yc
        R_sq = X_eff**2 + Y_eff**2
        k = 2*np.pi/wl
        phase = k * (np.sqrt(R_sq + f**2) - f)
        return np.angle(np.exp(1j * phase))
        
    init_map = compute_full_map(pred_params)
    refined_map = compute_full_map(refined_params)
    
    diff_init = wrap_phase(init_map - gt_phase_wrapped)
    diff_refined = wrap_phase(refined_map - gt_phase_wrapped)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Maps
    axes[0,0].imshow(gt_phase_wrapped, cmap='hsv')
    axes[0,0].set_title("Ground Truth Phase")
    
    axes[0,1].imshow(init_map, cmap='hsv')
    axes[0,1].set_title("Predicted Phase (Exp9)")
    
    axes[0,2].imshow(refined_map, cmap='hsv')
    axes[0,2].set_title(f"Refined Phase (DE)\nCost: {final_cost:.4f}")
    
    # Row 2: Residuals
    # Show diff wrapped to [-pi, pi]
    axes[1,0].axis('off')
    
    im1 = axes[1,1].imshow(diff_init, cmap='RdBu', vmin=-np.pi, vmax=np.pi)
    axes[1,1].set_title(f"Initial Residual\nRMS: {np.std(diff_init):.4f}")
    plt.colorbar(im1, ax=axes[1,1])
    
    im2 = axes[1,2].imshow(diff_refined, cmap='RdBu', vmin=-np.pi, vmax=np.pi)
    axes[1,2].set_title(f"Refined Residual\nRMS: {np.std(diff_refined):.4f}")
    plt.colorbar(im2, ax=axes[1,2])
    
    out_path = os.path.join(args.output_dir, f"refine_sample_{args.sample_id}.png")
    plt.savefig(out_path)
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    main()
