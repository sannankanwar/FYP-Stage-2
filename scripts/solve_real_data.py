
import os
import sys
import argparse
import yaml
import time
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# Add root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.factory import get_model
from src.inversion.forward_model import compute_hyperbolic_phase, wrap_phase, get_2channel_representation
# Reuse the GPU forward model from refine_de_gpu
from scripts.refine_de_gpu import batch_forward_model 

# ===================== Hardware Setup =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Wrappers (Redefined here or imported? Inline is safer for script isolation)
GLOBAL_TARGET_TENSOR = None
GLOBAL_N = 1024 # Target resolution after crop

def objective_function_vectorized(x):
    """
    SciPy wrapper handling both Vectorized DE (Batch) and L-BFGS-B Polish (Single).
    """
    # Check if batch (DE) or single (Polish)
    if x.ndim == 1:
        # Single vector (5,) -> Reshape to (1, 5)
        x_input = x[np.newaxis, :]
        is_single = True
    else:
        # Batch (5, Pop) -> Transpose to (Pop, 5)
        x_input = x.T
        is_single = False
        
    params = torch.tensor(x_input, dtype=torch.float32, device=device)
    
    # We optimize 5 parameters: [xc, yc, S, wl, f]
    costs = batch_forward_model(params, GLOBAL_N, None, GLOBAL_TARGET_TENSOR)
    
    if is_single:
        return float(costs[0]) # Return float for minimize
    return costs # Return array for differential_evolution

def load_and_process_csv(filepath, target_size=1024):
    """
    Load CSV, Center Crop.
    """
    print(f"Loading {filepath}...")
    # CSV is 2D array in radians? Or flat? User snippet said "loadtxt delimiter=,"
    try:
        data_raw = np.loadtxt(filepath, delimiter=',')
    except Exception as e:
        print(f"Error loading numpy: {e}. Trying pandas...")
        data_raw = pd.read_csv(filepath, header=None).values
        
    H, W = data_raw.shape
    if H < target_size or W < target_size:
        raise ValueError(f"Input data ({H}x{W}) is smaller than target ({target_size}x{target_size}).")

    row_start = (H - target_size) // 2
    row_end = row_start + target_size
    col_start = (W - target_size) // 2
    col_end = col_start + target_size

    data_crop = data_raw[row_start:row_end, col_start:col_end]
    
    # Wrap to [-pi, pi] for consistency (User snippet did this: angle(exp(1j*data)))
    # The input CSV might be unwrapped or wrapped.
    # User snippet: data = np.angle(np.exp(1j * data))
    data_wrapped = np.angle(np.exp(1j * data_crop))
    
    return data_wrapped # (1024, 1024)

def main():
    parser = argparse.ArgumentParser(description="Solve Real Data using GPU DE")
    parser.add_argument("--input_file", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Exp9 Checkpoint")
    parser.add_argument("--config", type=str, help="Config path")
    parser.add_argument("--output_dir", type=str, default="outputs/real_data_solutions")
    parser.add_argument("--pop_size", type=int, default=50)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--crop_size", type=int, default=1024)
    args = parser.parse_args()

    filename = os.path.basename(args.input_file).replace('.csv', '')
    out_dir = os.path.join(args.output_dir, filename)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Processing {filename} on {device}")

    # 1. Load Data
    phase_map = load_and_process_csv(args.input_file, args.crop_size)
    
    # Convert to 2-channel tensor [Cos, Sin] for Model Input
    # phase_map is (N, N)
    # Model expects (2, N, N) [Cos, Sin]?? NO.
    # Data Loader simulation.py: get_2channel_representation returns (H, W, 2) -> Transpose to (2, H, W)
    # So we do:
    cos_map = np.cos(phase_map)
    sin_map = np.sin(phase_map)
    input_tensor = torch.stack([
        torch.from_numpy(cos_map), 
        torch.from_numpy(sin_map)
    ], dim=0).float() # (2, H, W)
    
    # Prepare Global Target for DE
    global GLOBAL_TARGET_TENSOR
    global GLOBAL_N
    GLOBAL_TARGET_TENSOR = input_tensor.to(device)
    GLOBAL_N = args.crop_size
    
    # 2. Checkpoint & Model
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    elif 'config' in ckpt:
        config = ckpt['config']
    else:
        # Fallback config if missing
        config = {'model': {'name': 'spectral_resnet', 'activation': 'silu'}}
        
    def flatten_config(cfg):
        flat = cfg.copy()
        if 'model' in cfg: flat.update(cfg['model'])
        return flat
    
    flat_config = flatten_config(config)
    flat_config['resolution'] = args.crop_size # Force resolution match
    
    # 3. Predict Initial Guess
    model = get_model(flat_config)
    # Handle state dict (strict=False if needed due to resolution change in FNO?)
    # FNO is resolution invariant usually.
    try:
        model.load_state_dict(ckpt['model_state_dict'])
    except:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print("Loaded model with strict=False")
        
    model.to(device) # Exp9 model on GPU? Or CPU? Forward model is GPU.
    model.eval()
    
    # Model on CPU? 
    # Exp9 model might be heavy. Let's run prediction on Device if possible.
    model.to(device)
    input_batch = input_tensor.unsqueeze(0).to(device) # (1, 2, H, W)
    
    with torch.no_grad():
        pred_params = model(input_batch).squeeze(0).cpu().numpy()
        
    print(f"Initial Prediction: {pred_params}")
    # [xc, yc, S, wl, f]
    
    # 4. Bounds (Relaxed and Widened for Real Data)
    # The prediction might be out of distribution (e.g. if model trained on S<40 but real crop is S=100)
    # We must allow the optimizer to explore a much larger space.
    
    bounds = [
        (pred_params[0]-500, pred_params[0]+500), # xc (Allow large shift)
        (pred_params[1]-500, pred_params[1]+500), # yc
        (10.0, 500.0),                             # S (Scale/FOV). 1024 pixels could be 10um or 500um.
        (0.4, 0.7),                                # Wavelength (Physics constrained 400-700nm)
        (10.0, 500.0)                              # Focal Length
    ]
    
    print(f"Bounds: {bounds}")
    
    # 5. Run DE
    start_time = time.time()
    
    # Init population around prediction (but variance must be high enough to match bounds)
    n_params = 5
    total_pop = args.pop_size * n_params
    init_pop = np.zeros((total_pop, n_params))
    
    # Random init within FULL bounds
    for i in range(n_params):
        low, high = bounds[i]
        init_pop[:, i] = np.random.uniform(low, high, total_pop)
        
    # Inject Prediction
    init_pop[0] = pred_params
    
    # Strategy: 'best1bin' is greedy. 'rand1bin' explores more. 
    # Tol: set extremely low to prevent early stop.
    
    result = differential_evolution(
        objective_function_vectorized,
        bounds,
        strategy='best1bin', # or rand1bin
        maxiter=args.max_iter,
        popsize=args.pop_size,
        tol=1e-6,        # Force continue
        atol=1e-6,       # Force continue
        mutation=(0.5, 1.0),
        recombination=0.7,
        vectorized=True,
        workers=1,
        polish=True,
        init=init_pop,
        disp=True
    )
    
    end_time = time.time()
    print(f"Completed in {end_time - start_time:.2f}s. Cost: {result.fun:.5f}")
    
    # 6. Save Results
    best_params = result.x
    
    # Save CSV of params
    df = pd.DataFrame([best_params], columns=['xc', 'yc', 'S', 'wl', 'f'])
    df['cost'] = result.fun
    df.to_csv(os.path.join(out_dir, 'solution.csv'), index=False)
    
    # Visualization
    # Reconstruct
    xc, yc, S, wl, f = best_params
    linspace = torch.linspace(-0.5, 0.5, GLOBAL_N, device=device)
    Y_base, X_base = torch.meshgrid(linspace, linspace, indexing='ij')
    X_phys = X_base * S + xc
    Y_phys = Y_base * S + yc
    R2 = X_phys**2 + Y_phys**2
    phase = (2.0 * np.pi / wl) * (torch.sqrt(R2 + f**2) - f)
    refined_map = torch.atan2(torch.sin(phase), torch.cos(phase)).cpu().numpy()
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(phase_map, cmap='hsv')
    ax[0].set_title("Measured (Center Crop)")
    ax[1].imshow(refined_map, cmap='hsv')
    ax[1].set_title(f"Solved (Cost: {result.fun:.4f})")
    
    diff = wrap_phase(refined_map - phase_map)
    im = ax[2].imshow(diff, cmap='RdBu', vmin=-np.pi, vmax=np.pi)
    ax[2].set_title("Residual")
    plt.colorbar(im, ax=ax[2])
    
    plt.savefig(os.path.join(out_dir, 'result.png'))
    plt.close()

if __name__ == "__main__":
    main()
