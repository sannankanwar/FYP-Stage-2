import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.training.loss import Naive5ParamMSELoss, WeightedStandardizedLoss, WeightedPhysicsLoss
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
        
        # Setup Optimizer
        lr = float(config.get("learning_rate", 1e-3))
        opt_name = config.get("optimizer", "adam").lower()
        
        if opt_name == "muon":
            from src.training.optimizers import Muon
            print("Using Muon Optimizer")
            self.optimizer = Muon(model.parameters(), lr=lr) 
        else:
            print("Using Adam Optimizer")
            self.optimizer = optim.Adam(model.parameters(), lr=lr)
            
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
        
        # Setup Normalizer
        self.normalizer = None
        if config.get("standardize_outputs", False):
            print("Using Output Standardization")
            ds = train_loader.dataset
            ranges = {
                'xc': ds.xc_range,
                'yc': ds.yc_range,
                'fov': ds.fov_range,
                'wavelength': ds.wavelength_range,
                'focal_length': ds.focal_length_range
            }
            self.normalizer = ParameterNormalizer(ranges)

        # Setup Loss Function
        loss_name = config.get("loss_function", "mse")
        if loss_name == "naive_5param":
            print("Using Naive5ParamMSELoss")
            self.criterion = Naive5ParamMSELoss(normalizer=self.normalizer)
        elif loss_name == "weighted_standardized":
            print("Using WeightedStandardizedLoss")
            weights = config.get("loss_weights", [1.0, 1.0, 1.0, 10.0, 10.0])
            self.criterion = WeightedStandardizedLoss(weights=weights, normalizer=self.normalizer)
        elif loss_name == "weighted_physics":
            print("Using WeightedPhysicsLoss")
            weights = config.get("loss_weights", [1.0, 1.0, 1.0, 10.0, 10.0])
            # These lambdas should be in config
            l_param = float(config.get("lambda_param", 1.0))
            l_physics = float(config.get("lambda_physics", 0.1))
            self.criterion = WeightedPhysicsLoss(
                lambda_param=l_param,
                lambda_physics=l_physics,
                param_weights=weights,
                window_size=config.get("window_size", 100.0),
                normalizer=self.normalizer
            )
        else:
            print(f"Using Standard MSELoss (Fallback for '{loss_name}')")
            self.criterion = nn.MSELoss()
        
        # NOTE: Removed fixed anchor grid methodology
        # Training now uses only random on-the-fly data for all epochs

        # Training State
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        
        # Paths
        experiment_name = config.get("experiment_name", "default_experiment")
        self.experiment_dir = os.path.join("outputs_2", experiment_name)
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

    def train(self):
        epochs = int(self.config.get("epochs", 10))
        log_interval = int(self.config.get("log_interval", 10))
        
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(self.start_epoch, epochs):
            start_time = time.time()
            
            # All epochs use random on-the-fly data
            print(f\"--- Epoch {epoch+1} (Random Data) ---\")
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
                self._save_checkpoint(epoch, val_loss, name="best_model.pth")
            
            self._save_checkpoint(epoch, val_loss, name="latest_checkpoint.pth")
            
            # Snapshot Logic (Every 5 Epochs)
            if (epoch + 1) % 5 == 0 and self.val_loader:
                self._save_snapshot(epoch)
            
    def _train_epoch(self, epoch, loader, log_interval):
        self.model.train()
        total_loss = 0
        num_batches = len(loader)
        
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1} Train", leave=False)
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # Advanced Losses handle standardization internally via self.normalizer
            if isinstance(self.criterion, (Naive5ParamMSELoss, WeightedStandardizedLoss, WeightedPhysicsLoss)):
                 # These accept (pred, target, input_images)
                 loss, details = self.criterion(output, target, data)
            else:
                # Fallback for standard MSE, need manual normalization if not handled
                current_target = target
                if self.normalizer:
                    current_target = self.normalizer.normalize_tensor(target)
                loss = self.criterion(output, current_target)
                
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        return total_loss / num_batches

    def _validate_epoch(self, epoch):
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                if isinstance(self.criterion, (Naive5ParamMSELoss, WeightedStandardizedLoss, WeightedPhysicsLoss)):
                     loss, _ = self.criterion(output, target, data)
                else:
                    current_target = target
                    if self.normalizer:
                        current_target = self.normalizer.normalize_tensor(target)
                    loss = self.criterion(output, current_target)
                
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
                
                # Denormalize output for plotting real units
                if self.normalizer:
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
            print(f"Failed to generate snapshot: {e}")

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
