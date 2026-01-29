
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

    def _compute_step(self, data, target):
        """
        Shared logic for Training and Validation.
        Computes forward model, residual, refiner prediction, and loss.
        """
        # 1. Baseline Inference (Frozen)
        with torch.no_grad():
            theta_pred_base = self.baseline_model(data)
            
            # 2. Forward Physics Construction
            xc = theta_pred_base[:, 0]
            yc = theta_pred_base[:, 1]
            S_param = theta_pred_base[:, 2]
            f = theta_pred_base[:, 3]
            lam = theta_pred_base[:, 4]
            
            # Construct Grids (Vectorized)
            B, _, H, W = data.shape
            
            y_lin = torch.linspace(-0.5, 0.5, H, device=self.device)
            x_lin = torch.linspace(-0.5, 0.5, W, device=self.device)
            mesh_y, mesh_x = torch.meshgrid(y_lin, x_lin, indexing='ij')
            mesh_x = mesh_x.unsqueeze(0).expand(B, -1, -1)
            mesh_y = mesh_y.unsqueeze(0).expand(B, -1, -1)
            
            S_view = S_param.view(B, 1, 1)
            xc_view = xc.view(B, 1, 1)
            yc_view = yc.view(B, 1, 1)
            
            X_grid = xc_view + mesh_x * S_view
            Y_grid = yc_view + mesh_y * S_view
            
            f_view = f.view(B, 1, 1)
            lam_view = lam.view(B, 1, 1)
            
            k = 2.0 * torch.pi / lam_view
            R2 = X_grid**2 + Y_grid**2
            phi_recon_unwrapped = k * (torch.sqrt(R2 + f_view**2) - f_view)
            
            recon_cos = torch.cos(phi_recon_unwrapped).unsqueeze(1)
            recon_sin = torch.sin(phi_recon_unwrapped).unsqueeze(1)
            
            in_cos = data[:, 0:1, :, :]
            in_sin = data[:, 1:2, :, :]
            
            # 3. Robust Residual Calculation
            res_cos = in_cos * recon_cos + in_sin * recon_sin
            res_sin = in_sin * recon_cos - in_cos * recon_sin
            
            res_mag = torch.sqrt(res_cos**2 + res_sin**2 + 1e-8)
            res_cos = res_cos / res_mag
            res_sin = res_sin / res_mag
            
        # 4. Refiner Input Construction
        refiner_input = torch.cat([data, res_cos, res_sin], dim=1)
        
        # 5. Prediction
        theta_pred_norm = self.normalize_params(theta_pred_base)
        
        # Predict Normalized Delta
        delta_norm = self.model(refiner_input, theta_pred_norm)
        
        # 6. Apply Correction
        delta_phys = delta_norm * self.param_stds.unsqueeze(0)
        theta_final = theta_pred_base + delta_phys
        
        # 7. Loss (Normalized MSE)
        target_norm = self.normalize_params(target)
        final_norm = self.normalize_params(theta_final)
        
        loss = nn.MSELoss()(final_norm, target_norm)
        return loss
        
    def _train_epoch(self, epoch, loader, log_interval):
        self.model.train()
        total_loss = 0
        num_batches = len(loader)
        
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1} Refine", leave=False)
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            
            try:
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    loss = self._compute_step(data, target)
            except AttributeError: # Fallback for older torch
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    loss = self._compute_step(data, target)
                
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        return total_loss / num_batches

    def _validate_epoch(self, epoch):
        self.model.eval()
        total_loss = 0
        loader = self.val_loader
        
        if not loader:
            return 0.0

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                
                try:
                    with torch.amp.autocast('cuda', enabled=self.use_amp):
                        loss = self._compute_step(data, target)
                except AttributeError:
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        loss = self._compute_step(data, target)
                        
                total_loss += loss.item()
                
        return total_loss / len(loader)
