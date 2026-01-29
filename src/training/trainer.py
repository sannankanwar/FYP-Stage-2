import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from src.training.loss import build_loss, CompositeLoss
from src.utils.normalization import ParameterNormalizer
from scripts.evaluate import plot_scatter

class Trainer:
    def __init__(self, config, model, train_loader, val_loader=None):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Setup Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = self.model.to(self.device)
        
        # Setup Normalizer
        self.normalizer = None
        if config.get("standardize_outputs", False):
            print("Using Output Standardization")
            ds = train_loader.dataset
            ranges = {
                'xc': ds.xc_range,
                'yc': ds.yc_range,
                'S': ds.S_range,
                'wavelength': ds.wavelength_range,
                'focal_length': ds.focal_length_range
            }
            self.normalizer = ParameterNormalizer(ranges)

        # Setup Loss Function via Factory
        # This supports the standardized config schema (loss: { ... })
        loss_cfg = config.get("loss", {})
        data_cfg = config.get("data", {})
        
        # If legacy flattened config is passed, try to adapt (optional, but good for safety)
        if not loss_cfg and "loss_function" in config:
            print("WARNING: Detecting legacy config format. Attempting to adapt...")
            loss_cfg = {"mode": "unit_standardized"} # Fallback
            
        print("Building Loss Module...")
        self.criterion = build_loss(loss_cfg, data_cfg)
        self.criterion = self.criterion.to(self.device)

        # Setup Optimizer (AFTER LOSS so we can include loss params)
        lr = float(config.get("learning_rate", 1e-3))
        opt_name = config.get("optimizer", "adam").lower()
        
        # Collect parameters (Model + Loss)
        params_to_optimize = list(model.parameters())
        if isinstance(self.criterion, nn.Module) and len(list(self.criterion.parameters())) > 0:
            print(f"Adding {len(list(self.criterion.parameters()))} parameters from Loss function to Optimizer.")
            params_to_optimize += list(self.criterion.parameters())
        
        if opt_name == "muon":
            from src.training.optimizers import Muon
            print("Using Muon Optimizer")
            self.optimizer = Muon(params_to_optimize, lr=lr) 
        else:
            print("Using Adam Optimizer")
            self.optimizer = optim.Adam(params_to_optimize, lr=lr)
            
        # Setup Scheduler
        self.scheduler = None
        scheduler_name = config.get("scheduler", None)
        if scheduler_name == "plateau":
            print("Using ReduceLROnPlateau Scheduler")
            patience = int(config.get("scheduler_patience", 10))
            factor = float(config.get("scheduler_factor", 0.1))
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=patience, factor=factor
            )
        
        # NOTE: Removed fixed anchor grid methodology
        # Training now uses only random on-the-fly data for all epochs

        # Training State
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        
        # Paths
        experiment_name = config.get("experiment_name", "default_experiment")
        output_root = config.get("output_dir", "outputs_2")
        self.experiment_dir = os.path.join(output_root, experiment_name)
        self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        self.log_dir = os.path.join(self.experiment_dir, "logs")
        self.snapshot_dir = os.path.join(self.experiment_dir, "snapshots") # New
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.snapshot_dir, exist_ok=True)
        
        # Save experiment info
        description = config.get("description", "No description provided.")
        with open(os.path.join(self.experiment_dir, "experiment_info.md"), "w") as f:
            f.write(f"# Experiment: {experiment_name}\n\n")
            f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## Description\n{description}\n")

        # AMP Scaler
        self.use_amp = self.device.type == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        if self.use_amp:
            print("AMP (Automatic Mixed Precision) Enabled")

    def train(self):
        epochs = int(self.config.get("epochs", 10))
        log_interval = int(self.config.get("log_interval", 10))
        
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(self.start_epoch, epochs):
            start_time = time.time()
            
            # All epochs use random on-the-fly data
            print(f"--- Epoch {epoch+1} (Random Data) ---")
            active_loader = self.train_loader
            
            train_loss = self._train_epoch(epoch, active_loader, log_interval)
            
            val_loss = 0.0
            if self.val_loader:
                val_loss = self._validate_epoch(epoch)
                
            end_time = time.time()
            epoch_time = end_time - start_time
            
            print(f"Epoch {epoch+1}/{epochs} | Time: {epoch_time:.2f}s | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            # Save Checkpoint
            if self.val_loader and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self._save_checkpoint(epoch, val_loss, name="best_model.pth")
                
                # Save Best Metrics JSON (Atomic)
                import json
                import tempfile
                import shutil
                
                metrics = {
                    "run_id": self.config.get("experiment_name", "unknown"),
                    "best_epoch": epoch,
                    "best_val_mse": val_loss,
                    "best_val_r2": 0.0, # Placeholder until implemented
                    "schema_version": 1
                }
                
                metrics_path = os.path.join(self.experiment_dir, "best_metrics.json")
                tmp_path = metrics_path + ".tmp"
                
                with open(tmp_path, "w") as f:
                    json.dump(metrics, f, indent=2)
                os.rename(tmp_path, metrics_path)
            
            self._save_checkpoint(epoch, val_loss, name="latest_checkpoint.pth")
            
            # Save Epoch Metrics CSV
            history_path = os.path.join(self.experiment_dir, "metrics", "epoch_metrics.csv")
            os.makedirs(os.path.dirname(history_path), exist_ok=True)
            
            with open(history_path, "a") as f:
                if epoch == 0:
                     f.write("epoch,train_loss,val_loss,time\n")
                f.write(f"{epoch+1},{train_loss},{val_loss},{epoch_time}\n")
            
            # Snapshot Logic (Every 5 Epochs)
            if (epoch + 1) % 5 == 0 and self.val_loader:
                self._save_snapshot(epoch)
            
            # Save History to CSV
            history_path = os.path.join(self.experiment_dir, "history.csv")
            with open(history_path, "a") as f:
                # Header if new
                if epoch == 0:
                    f.write("epoch,train_loss,val_loss,time\n")
                f.write(f"{epoch+1},{train_loss},{val_loss},{epoch_time}\n")
            
    def _train_epoch(self, epoch, loader, log_interval):
        self.model.train()
        total_loss = 0
        num_batches = len(loader)
        
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1} Train", leave=False)
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                output = self.model(data)
                
                # Using CompositeLoss interface which handles everything
                # Passing epoch for physics scheduling
                # Passing data (images) for physics loss
                loss, metrics = self.criterion(
                    pred_params=output, 
                    true_params=target, 
                    input_images=data,
                    epoch=epoch
                )
                
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
            # Update progress bar with key metrics
            postfix = {'loss': f"{loss.item():.4f}"}
            if "loss_physics" in metrics:
                postfix["phys"] = f"{metrics['loss_physics']:.4f}"
            progress_bar.set_postfix(postfix)
            
        return total_loss / num_batches

    def _validate_epoch(self, epoch):
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                loss, metrics = self.criterion(
                    pred_params=output, 
                    true_params=target, 
                    input_images=data,
                    epoch=epoch
                )
                
                total_loss += loss.item()
                
        avg_val_loss = total_loss / num_batches
        
        if self.scheduler:
            self.scheduler.step(avg_val_loss)
            
        return avg_val_loss

    def _save_snapshot(self, epoch):
        """
        Run inference on validation set and generate scatter plots.
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data = data.to(self.device)
                output = self.model(data)
                
                # Denormalize based on Model Contract
                output_space = getattr(self.model, 'output_space', 'unknown')
                if output_space == 'physical':
                     pass # Do nothing, output is correct
                elif self.normalizer:
                   output = self.normalizer.denormalize_tensor(output)
                
                all_preds.append(output.cpu().numpy())
                all_targets.append(target.numpy())
                
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        save_path = os.path.join(self.snapshot_dir) # plot_scatter appends filename
        # plot_scatter saves as "chat_plots.png" or similar. We need to customize filename or rename.
        # Let's modify plot_scatter usage or just rely on it overwriting/making new file.
        # Ideally, we pass a filename. The existing plot_scatter takes output_dir.
        
        # We will wrap plot_scatter logic here to customize filename
        # Or better, update plot_scatter in evaluate.py to take filename arg?
        # For now, let's just make a subdir per epoch to avoid mess
        epoch_snap_dir = os.path.join(self.snapshot_dir, f"epoch_{epoch+1}")
        os.makedirs(epoch_snap_dir, exist_ok=True)
        
        print(f"Saving Snapshot for Epoch {epoch+1}...")
        try:
            plot_scatter(all_targets, all_preds, epoch_snap_dir, title=f"Epoch {epoch+1} Validation")
        except Exception as e:
            print(f"Failed to generate scatter plot: {e}")
        
        # Generate residual phase map for one sample
        try:
            self._plot_residual_phase(epoch_snap_dir, epoch)
        except Exception as e:
            print(f"Failed to generate residual plot: {e}")

    def _save_checkpoint(self, epoch, val_loss, name):
        path = os.path.join(self.checkpoint_dir, name)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        torch.save(checkpoint, path)

    def _plot_residual_phase(self, save_dir, epoch):
        """
        Generate residual phase maps for 5 specific test cases.
        """
        import matplotlib
        matplotlib.use('Agg')
        from data.loaders.simulation import generate_single_sample
        
        resolution = self.config.get("resolution", 256)
        
        # Define 5 Test Cases with S (window size) parameter
        # Format: [xc, yc, S, wavelength, focal_length]
        test_cases = [
            {'name': 'Case1_Base',     'p': [100.0, 100.0, 20.0, 0.6, 50.0]},
            {'name': 'Case2_Shift',    'p': [200.0, 200.0, 20.0, 0.6, 50.0]},
            {'name': 'Case3_Far',      'p': [300.0, 300.0, 30.0, 0.6, 70.0]},
            {'name': 'Case4_SmallS',   'p': [300.0, 300.0, 10.0, 0.5, 40.0]},
            {'name': 'Case5_LargeS',   'p': [300.0, 300.0, 40.0, 0.4, 80.0]},
        ]
        
        for case in test_cases:
            true_params = np.array(case['p'])
            
            # Generate input using S as window size
            inp, _ = generate_single_sample(
                N=resolution, 
                xc=true_params[0], yc=true_params[1], S=true_params[2],
                wavelength=true_params[3], focal_length=true_params[4]
            )
            
            # Predict
            inp_tensor = torch.from_numpy(inp.transpose(2,0,1)).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                pred_raw = self.model(inp_tensor)
                
                # Denormalize based on Model Contract
                output_space = getattr(self.model, 'output_space', 'unknown')
                if output_space == 'physical':
                    pred_params = pred_raw.cpu().numpy()[0]
                elif self.normalizer and self.config.get("standardize_outputs", False):
                    pred_denorm = self.normalizer.denormalize_tensor(pred_raw)
                    pred_params = pred_denorm.cpu().numpy()[0]
                else:
                    pred_params = pred_raw.cpu().numpy()[0]
            
            # Generate True/Pred Phase Maps for comparison
            true_inp, _ = generate_single_sample(
                N=resolution, 
                xc=true_params[0], yc=true_params[1], S=true_params[2],
                wavelength=true_params[3], focal_length=true_params[4]
            )
            pred_inp, _ = generate_single_sample(
                N=resolution, 
                xc=pred_params[0], yc=pred_params[1], S=pred_params[2],
                wavelength=pred_params[3], focal_length=pred_params[4]
            )
            
            true_phase = np.arctan2(true_inp[:,:,1], true_inp[:,:,0])
            pred_phase = np.arctan2(pred_inp[:,:,1], pred_inp[:,:,0])
            residual = np.angle(np.exp(1j * (true_phase - pred_phase)))
            
            # Plot
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            im0 = axes[0].imshow(true_phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
            axes[0].set_title('True Phase')
            axes[0].axis('off')
            plt.colorbar(im0, ax=axes[0], fraction=0.046)
            
            im1 = axes[1].imshow(pred_phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
            axes[1].set_title('Predicted Phase')
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046)
            
            im2 = axes[2].imshow(residual, cmap='RdBu', vmin=-np.pi, vmax=np.pi)
            axes[2].set_title('Residual (True - Pred)')
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], fraction=0.046)
            
            # Param comparison
            param_names = ['xc', 'yc', 'S', 'λ', 'f']
            x = np.arange(5)
            width = 0.35
            axes[3].bar(x - width/2, true_params, width, label='True', alpha=0.7)
            axes[3].bar(x + width/2, pred_params, width, label='Pred', alpha=0.7)
            axes[3].set_xticks(x)
            axes[3].set_xticklabels(param_names)
            axes[3].legend()
            axes[3].set_title(f'Params: {case["name"]}')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'residual_{case["name"]}.png'), dpi=100)
            plt.close()

