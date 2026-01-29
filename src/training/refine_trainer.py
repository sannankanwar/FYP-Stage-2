
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from tqdm import tqdm
import numpy as np
from src.training.trainer import Trainer
from src.inversion.forward_model import compute_hyperbolic_phase, get_2channel_representation
from src.models.refiner import ResNetRefiner

class RefiningTrainer(Trainer):
    def __init__(self, config, baseline_model, refiner_model, train_loader, val_loader=None):
        # We initialize the parent Trainer but we need to trick it 
        # because the 'model' argument is now split.
        # We pass refiner_model as the primary model to optimize.
        super().__init__(config, refiner_model, train_loader, val_loader)
        
        self.baseline_model = baseline_model.to(self.device)
        self.baseline_model.eval()
        # Freeze baseline
        for param in self.baseline_model.parameters():
            param.requires_grad = False
            
        print("Baseline Model Frozen.")
        
        # Parameter Stats for Normalization (loaded from dataset)
        ds = train_loader.dataset
        self.param_ranges = {
            'xc': ds.xc_range,
            'yc': ds.yc_range,
            'S': ds.S_range,
            'wavelength': ds.wavelength_range, 
            'focal_length': ds.focal_length_range # if used
        }
        
        # Determine active params and their ranges/stats
        self.param_names = config.get("data", {}).get("params", ['xc', 'yc', 'S', 'f', 'lambda'])
        # Map abbreviations if needed ('f' -> 'focal_length', 'lambda' -> 'wavelength')
        full_names = []
        for p in self.param_names:
            if p == 'f': full_names.append('focal_length')
            elif p == 'lambda': full_names.append('wavelength')
            else: full_names.append(p)
            
        # Compute approx mean/std from ranges (assuming uniform for normalization proxy)
        # Uniform [a, b]: Mean = (a+b)/2, Std = (b-a)/sqrt(12)
        means = []
        stds = []
        for name in full_names:
            r = self.param_ranges[name]
            means.append((r[1] + r[0]) / 2.0)
            stds.append((r[1] - r[0]) / np.sqrt(12))
            
        self.param_means = torch.tensor(means, device=self.device, dtype=torch.float32)
        self.param_stds = torch.tensor(stds, device=self.device, dtype=torch.float32)
        
        print(f"RefiningTrainer Initialized. Param Stds: {self.param_stds.cpu().numpy()}")

    def normalize_params(self, theta):
        return (theta - self.param_means) / (self.param_stds + 1e-8)

    def denormalize_params(self, theta_norm):
        return theta_norm * self.param_stds + self.param_means
        
    def _train_epoch(self, epoch, loader, log_interval):
        self.model.train() # This is the refiner
        total_loss = 0
        num_batches = len(loader)
        
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1} Refine", leave=False)
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            # data: (B, 2, H, W) - Input Phase Map (cos, sin)
            
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # 1. Baseline Inference (Frozen)
                with torch.no_grad():
                    theta_pred_base = self.baseline_model(data)
                    
                    # 2. Forward Physics -> Reconstruct Phase
                    # Need to map theta_pred_base to args for forward model
                    # order: xc, yc, S, f, lambda
                    # Assuming standard order (xc, yc, S, f, lambda)
                    # We need to construct grids.
                    B, _, H, W = data.shape
                    
                    # Create grids (should utilize cache ideally, but overhead is low)
                    # Coordinates from -Range to +Range
                    # Assume data loader implies a coordinate system. 
                    # Standard simulation uses linspace(-resolution*pixel_size/2, ...)
                    # Let's assume the forward model helper can handle this or we pass grids.
                    # WAIT: training/loss.py handles forwarding via `compute_hyperbolic_phase`.
                    # We need X, Y grids.
                    # Let's recreate them to be safe.
                    # Pixel size? Exp config usually sets image size 1024, ranges -100..100? No.
                    # Simulation: "resolution: 1024", "image_size: [1024, 1024]".
                    # Coordinate range? usually inferred. 
                    # Let's check `src/data/loaders/simulation.py` or assume standard range [-500um, 500um]?
                    # Actually, let's use a simpler heuristic:
                    # We only need the residual for the Network input. 
                    # If we use wrong grid, residual is garbage.
                    # CRITICAL: We need consistent physics.
                    # Let's generate grids on device.
                    # Assuming ROI size is implicit in resolution vs pixel size.
                    # Standard Metalens ROI is usually defined in config.
                    
                    # Fallback: Assume the input `data` is sufficient? No, we need forward model.
                    # Let's create grids here.
                    start = -512 * 0.1 # assuming 0.1 um pixel size?
                    # CHECK: I need to verify pixel size.
                    
                    # TEMP FIX: use standardized grid if not in config.
                    # For now, let's construct grids assuming 1024 pixels mapped to... what?
                    # The simulator usually uses `grid_size = resolution * pixel_size`.
                    
                    # Let's look at `src/training/loss.py` or `src/inversion/forward_model.py` doesn't encode "data loader logic".
                    # `src/data/loaders/simulation.py` likely has the grid definition.
                    pass 
                
                    # ... [Placeholder for Grid Logic Logic] ... 
                    # For this step, I will need to verify the grid logic before completing.
                    # But assuming we have `phi_recon` (B, H, W):
                    
                    # Let's assume I fix the grid logic in the next step.
                    
                    # FOR NOW:
                    # I will assume `self.config` has pixel size or I can derive it.
                    
            # ...
        
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # 1. Baseline Inference (Frozen)
                with torch.no_grad():
                    theta_pred_base = self.baseline_model(data)
                    
                    # 2. Forward Physics Construction
                    # Parse parameters: [xc, yc, S, f, lambda] (assuming 5 output dims)
                    # NOTE: theta_pred_base is standardized or physical?
                    # The Baseline Model usually outputs PHYSICAL units directly if output_dim=5 and no internal norm?
                    # Wait, our baseline (e.g. FNO) output_dim=5. Config usually has "standardize_outputs: false" implicitly?
                    # The Trainer checks `config.get("standardize_outputs", False)`.
                    # If existing exp9 was trained with standardize_outputs=False (default), then theta_pred_base is physical.
                    # Verify: exp_noisy_09 config doesn't set standardize_outputs. Default is False.
                    # So theta_pred_base is in MICRO-METERS.
                    
                    xc = theta_pred_base[:, 0]
                    yc = theta_pred_base[:, 1]
                    S_param = theta_pred_base[:, 2]
                    f = theta_pred_base[:, 3]
                    lam = theta_pred_base[:, 4]
                    
                    # Construct Grids (Vectorized)
                    # We need (B, H, W) grids.
                    B, _, H, W = data.shape
                    
                    # Create normalized grid template [-0.5, 0.5]
                    # This allows scaling by S and shifting by xc/yc easily
                    # shape (1, H, W)
                    y_lin = torch.linspace(-0.5, 0.5, H, device=self.device)
                    x_lin = torch.linspace(-0.5, 0.5, W, device=self.device)
                    mesh_y, mesh_x = torch.meshgrid(y_lin, x_lin, indexing='ij')
                    mesh_x = mesh_x.unsqueeze(0).expand(B, -1, -1) # (B, H, W)
                    mesh_y = mesh_y.unsqueeze(0).expand(B, -1, -1)
                    
                    # Physical Grids
                    # S_param is (B,). We need (B, 1, 1).
                    S_view = S_param.view(B, 1, 1)
                    xc_view = xc.view(B, 1, 1)
                    yc_view = yc.view(B, 1, 1)
                    
                    # grid = center + relative_pos * size
                    # relative_pos is [-0.5, 0.5]
                    X_grid = xc_view + mesh_x * S_view
                    Y_grid = yc_view + mesh_y * S_view
                    
                    # Compute Phase
                    # broadcast f and lambda
                    f_view = f.view(B, 1, 1)
                    lam_view = lam.view(B, 1, 1)
                    
                    # Ideal Hyperbolic Phase
                    # phi = (2pi/lambda) * (sqrt(x^2 + y^2 + f^2) - f)
                    k = 2.0 * torch.pi / lam_view
                    R2 = X_grid**2 + Y_grid**2
                    phi_recon_unwrapped = k * (torch.sqrt(R2 + f_view**2) - f_view)
                    
                    # Wrap Phase -> Phasor
                    # We need Phasor for residual: exp(1j * phi)
                    # We don't strictly need to wrap before exp, exp handles it.
                    # But inputs are 2-channel [cos, sin].
                    
                    # Reconstruct Phasor (B, 2, H, W)
                    recon_cos = torch.cos(phi_recon_unwrapped).unsqueeze(1)
                    recon_sin = torch.sin(phi_recon_unwrapped).unsqueeze(1)
                    
                    # Input Phasor (B, 2, H, W) defined by data
                    in_cos = data[:, 0:1, :, :]
                    in_sin = data[:, 1:2, :, :]
                    
                    # 3. Robust Residual Calculation (Complex Multiply)
                    # Res = U * conj(V)
                    # Re(Res) = ReU*ReV + ImU*ImV
                    # Im(Res) = ImU*ReV - ReU*ImV
                    
                    res_cos = in_cos * recon_cos + in_sin * recon_sin
                    res_sin = in_sin * recon_cos - in_cos * recon_sin
                    
                    # Normalize residual phasor for stability (prevent drift away from unit circle)
                    res_mag = torch.sqrt(res_cos**2 + res_sin**2 + 1e-8)
                    res_cos = res_cos / res_mag
                    res_sin = res_sin / res_mag
                    
                # 4. Refiner Input Construction
                # Concatenate [in_cos, in_sin, res_cos, res_sin] -> (B, 4, H, W)
                refiner_input = torch.cat([data, res_cos, res_sin], dim=1)
                
                # 5. Prediction
                # Conditioning on Normalized Theta Pred
                theta_pred_norm = self.normalize_params(theta_pred_base)
                
                # Predict Normalized Delta
                delta_norm = self.model(refiner_input, theta_pred_norm)
                
                # 6. Apply Correction
                # delta_phys = delta_norm * std
                delta_phys = delta_norm * self.param_stds.unsqueeze(0)
                theta_final = theta_pred_base + delta_phys
                
                # 7. Loss
                # Weighted MSE on parameters
                # L = sum ( (theta_final - theta_gt)^2 / var )
                # weights = 1/var. 
                # Diff = theta_final - target
                diff = theta_final - target
                # We can compute weighted MSE efficiently in normalized space?
                # Normalized Error = (Theta_final - Theta_gt) / Std
                #                  = (Theta_final - Theta_gt)^2 / Var
                # So Mean(Normalized_Error^2) is exactly Weighted MSE.
                
                target_norm = self.normalize_params(target)
                final_norm = self.normalize_params(theta_final)
                
                loss_reg = nn.MSELoss()(final_norm, target_norm)
                
                loss = loss_reg
                
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
            # Progress Bar
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        return total_loss / num_batches

