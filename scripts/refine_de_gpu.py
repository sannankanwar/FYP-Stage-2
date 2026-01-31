
import os
import sys
import argparse
import yaml
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# Add root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.factory import get_model
from src.inversion.forward_model import compute_hyperbolic_phase, wrap_phase
from data.loaders.simulation import OnTheFlyDataset

# ===================== Hardware Setup =====================

# Auto-detect GPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def batch_forward_model(params_batch, N_pixels, fixed_lambda, measured_tensor):
    """
    Computes the cost for the WHOLE population at once using GPU tensors.
    
    Args:
        params_batch: (pop_size, 5) tensor containing [xc, yc, S, wl, f] 
                      OR (pop_size, 4) if wl is fixed? 
                      User snippet had 4, but our model predicts 5. 
                      Let's support 5 params: [xc, yc, S, wl, f].
        N_pixels: Image resolution (e.g., 128)
        fixed_lambda: Scalar or tensor. If params_batch contains wl, we use that.
        measured_tensor: (2, N, N) tensor of the target image on GPU
    
    Returns:
        costs: (pop_size,) array of errors
    """
    # 1. Unpack parameters (Shape: [Pop_Size, 1, 1] for broadcasting)
    xc = params_batch[:, 0].view(-1, 1, 1)
    yc = params_batch[:, 1].view(-1, 1, 1)
    S  = params_batch[:, 2].view(-1, 1, 1)
    # wl = params_batch[:, 3].view(-1, 1, 1) # If we optimize wavelength
    # f  = params_batch[:, 4].view(-1, 1, 1)
    
    # Check if we are optimizing 4 or 5 params.
    # The DE vector will come in as shape (Pop, NumParams).
    # If 5 params: [xc, yc, S, wl, f]
    wl = params_batch[:, 3].view(-1, 1, 1)
    f  = params_batch[:, 4].view(-1, 1, 1)
    
    pop_size = params_batch.shape[0]

    # 2. Create Normalized Grid [-0.5, 0.5]
    # We create one base grid and reuse it. Assuming Square Lens/Window.
    linspace = torch.linspace(-0.5, 0.5, N_pixels, device=device)
    Y_base, X_base = torch.meshgrid(linspace, linspace, indexing='ij') # (N, N)
    
    # 3. Compute Physical Coordinates
    # X_phys = xc + X_base * S
    # The 'S' parameter is the Physical Size of the window being viewed.
    # This maps the discrete grid `[-0.5, 0.5]` to `[xc - S/2, xc + S/2]`.
    X_phys = X_base.unsqueeze(0) * S + xc
    Y_phys = Y_base.unsqueeze(0) * S + yc
    
    # 4. Compute Radius Squared
    R2 = X_phys**2 + Y_phys**2
    
    # 5. Compute Phase (The Forward Model)
    # Ideal Hyperbolic: phi = (2pi/lambda) * (sqrt(R^2 + f^2) - f)
    # (User snippet used Fresnel approx: -pi * R^2 / (lambda * f))
    # We use Exact to match training ground truth.
    
    sqrt = torch.sqrt(R2 + f**2)
    phase = (2.0 * torch.pi / wl) * (sqrt - f)
    
    # 6. Compute 2-Channel Representation (Cos, Sin)
    sin_sim = torch.sin(phase)
    cos_sim = torch.cos(phase)
    
    # 7. Compute Batch Cost (MSE)
    # measured_tensor shape: (2, N, N)
    target_cos = measured_tensor[0].unsqueeze(0) # (1, N, N)
    target_sin = measured_tensor[1].unsqueeze(0) # (1, N, N)
    
    # Error = Mean((sin_pred - sin_true)^2 + (cos_pred - cos_true)^2)
    diff_sin = sin_sim - target_sin
    diff_cos = cos_sim - target_cos
    
    # Sum over spatial dims (1,2), then divide by N*N
    # Or just mean.
    cost = torch.mean(diff_sin**2 + diff_cos**2, dim=(1, 2))
    
    return cost.cpu().numpy() # SciPy expects NumPy arrays back

# ===================== Global Wrappers for SciPy =====================

# These must be set before optimization runs
GLOBAL_TARGET_TENSOR = None
GLOBAL_N = 128

def objective_function_vectorized(x):
    """
    SciPy wrapper.
    x shape is (NumParams, PopSize) coming from SciPy if vectorized=True.
    We need to transpose it to (PopSize, NumParams) for PyTorch.
    """
    # Transpose: (5, Pop) -> (Pop, 5)
    params = torch.tensor(x.T, dtype=torch.float32, device=device)
    
    # We don't have 'fixed_lambda' because we are optimizing it (param 3)
    return batch_forward_model(params, GLOBAL_N, None, GLOBAL_TARGET_TENSOR)


def main():
    parser = argparse.ArgumentParser(description="GPU-Accelerated DE Refinement")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sample_id", type=int, default=0)
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--pop_size", type=int, default=50, help="Candidates per parameter (Total Pop = pop_size * 5?) No, scipy popsize is multiplier")
    # Note: SciPy popsize arg is a multiplier. Total pop = popsize * len(x).
    # If 5 params and popsize=20 -> 100 candidates.
    parser.add_argument("--config", type=str, help="Config path")
    parser.add_argument("--output_dir", type=str, default="outputs/refinement_de_gpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"🚀 Computation Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    # 1. Load Model & Config (Same as CPU script)
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    elif 'config' in ckpt:
        config = ckpt['config']
    else:
        raise ValueError("No config found.")

    def flatten_config(cfg):
        flat = cfg.copy()
        if 'model' in cfg: flat.update(cfg['model'])
        if 'data' in cfg: flat.update(cfg['data'])
        return flat
    
    flat_config = flatten_config(config)
    
    # 2. Load Validation Data
    print("Loading Dataset...")
    flat_config['seed'] = 42
    dataset = OnTheFlyDataset(flat_config, length=max(args.sample_id + 1, 100))
    
    input_tensor, target_params = dataset[args.sample_id]
    
    # Prepare Target on GPU
    # input_tensor is (2, H, W) [Cos, Sin]
    global GLOBAL_TARGET_TENSOR
    global GLOBAL_N
    
    GLOBAL_TARGET_TENSOR = input_tensor.to(device)
    GLOBAL_N = input_tensor.shape[-1]
    
    print(f"Target Loaded. Shape: {GLOBAL_TARGET_TENSOR.shape} on {device}")
    print(f"GT Params: {target_params.numpy()}")

    # 3. Predict Initial Guess
    model = get_model(flat_config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    with torch.no_grad():
        pred_params = model(input_tensor.unsqueeze(0)).squeeze(0).numpy()
        
    print(f"Initial Prediction: {pred_params}")
    # [xc, yc, S, wl, f]
    
    # 4. Bounds & Initialization
    # We want to center vector around prediction.
    # Bounds = Pred +/- 20% (or fixed range if safer)
    
    bounds = []
    # xc, yc: +/- 20um around prediction? Or just large window?
    # Prediction could be way off. Let's trust prediction within +/- 20um.
    bounds.append((pred_params[0]-20, pred_params[0]+20))
    bounds.append((pred_params[1]-20, pred_params[1]+20))
    
    # Scale: +/- 20%
    bounds.append((pred_params[2]*0.8, pred_params[2]*1.2))
    
    # Wavelength: +/- 10% (Physics usually narrows this down)
    bounds.append((pred_params[3]*0.9, pred_params[3]*1.1))
    
    # Focal Length: +/- 20%
    bounds.append((pred_params[4]*0.8, pred_params[4]*1.2))
    
    # Manual Population Initialization (Inject Prediction)
    # SciPy init array shape: (pop_size * len(x), len(x))??
    # No, SciPy 'init' can be (TotalPop, len(x)).
    # TotalPop = popsize * len(x).
    # If popsize=20, len=5 -> 100 candidates.
    
    n_params = 5
    total_pop = args.pop_size * n_params
    
    # Initialize random population within bounds
    init_pop = np.zeros((total_pop, n_params))
    for i in range(n_params):
        low, high = bounds[i]
        init_pop[:, i] = np.random.uniform(low, high, total_pop)
        
    # Inject Prediction at index 0 (Strategy: current best)
    init_pop[0] = pred_params
    
    # 5. Run DE
    print(f"⚡ Starting Vectorized DE on {device}...")
    print(f"Population: {total_pop} candidates per generation.")
    
    start_time = time.time()
    
    result = differential_evolution(
        objective_function_vectorized,
        bounds,
        strategy='best1bin',
        maxiter=args.max_iter,
        popsize=args.pop_size, # Multiplier
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        vectorized=True,       # <--- GPU MAGIC
        workers=1,             # Single CPU worker driving GPU
        polish=True,           # L-BFGS-B at end (will run on CPU/Scalar usually? check scipy)
        init=init_pop,         # Custom initialization
        disp=True
    )
    
    end_time = time.time()
    
    print("\n" + "="*40)
    print(f"✅ Optimization Complete in {end_time - start_time:.2f}s")
    print(f"Best Parameters: {result.x}")
    print(f"Final Cost: {result.fun}")
    
    # 6. Visualize
    refined_params = torch.tensor(result.x, device=device).unsqueeze(0) # (1, 5)
    
    # Reconstruct Map
    # We can reuse batch_forward_model but we need the MAP not the cost.
    # Let's just inline reconstruction for plot
    xc, yc, S, wl, f = result.x
    
    linspace = torch.linspace(-0.5, 0.5, GLOBAL_N, device=device)
    Y_base, X_base = torch.meshgrid(linspace, linspace, indexing='ij')
    X_phys = X_base * S + xc
    Y_phys = Y_base * S + yc
    R2 = X_phys**2 + Y_phys**2
    phase = (2.0 * np.pi / wl) * (torch.sqrt(R2 + f**2) - f)
    
    refined_map = torch.atan2(torch.sin(phase), torch.cos(phase)).cpu().numpy()
    
    # GT Map
    gt_cos = input_tensor[0].numpy()
    gt_sin = input_tensor[1].numpy()
    gt_map = np.arctan2(gt_sin, gt_cos)
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(gt_map, cmap='hsv')
    ax[0].set_title("Ground Truth")
    
    ax[1].imshow(refined_map, cmap='hsv')
    ax[1].set_title(f"Refined (Cost: {result.fun:.4f})")
    
    diff = wrap_phase(refined_map - gt_map)
    im = ax[2].imshow(diff, cmap='RdBu', vmin=-np.pi, vmax=np.pi)
    ax[2].set_title("Residual")
    plt.colorbar(im, ax=ax[2])
    
    out_path = os.path.join(args.output_dir, f"refine_gpu_sample_{args.sample_id}.png")
    plt.savefig(out_path)
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    main()
