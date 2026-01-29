
from src.training.trainer import Trainer
from tqdm import tqdm
import torch
import torch.nn as nn

class ActiveTrainer(Trainer):
    def __init__(self, config, model, train_loader, val_loader=None, sampler=None):
        super().__init__(config, model, train_loader, val_loader)
        self.sampler = sampler
        print("ActiveTrainer Initialized.")

    def _train_epoch(self, epoch, loader, log_interval):
        self.model.train()
        total_loss = 0
        num_batches = len(loader)
        
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1} Active", leave=False)
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # Forward
                output = self.model(data)
                
                # Standard Loss (Mean) for Optimization
                loss, metrics = self.criterion(
                    pred_params=output, 
                    true_params=target, 
                    input_images=data,
                    epoch=epoch
                )
                
            # Backward
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
            # --- Active Learning Update ---
            if self.sampler is not None:
                with torch.no_grad():
                    # Calculate per-sample loss for difficulty update
                    # Assuming regression output matches target columns
                    # Simple MSE on parameters
                    # (B, 5) - (B, 5) -> (B,)
                    
                    # NOTE: We should use the SAME logic as the difficulty definition.
                    # Usually MSE.
                    # Output might be normalized? Check config. 
                    # Usually "output_dim=5" means physical units in this project unless standardized.
                    # Assuming physical units.
                    
                    sq_err = (output - target) ** 2
                    # Mean over params -> (B,)
                    per_sample_loss = sq_err.mean(dim=1)
                    
                    # Update Sampler
                    # target contains the parameters [xc, yc, S, f, lambda]
                    self.sampler.update(target, per_sample_loss)
            
            # Progress
            postfix = {'loss': f"{loss.item():.4f}"}
            progress_bar.set_postfix(postfix)
            
        return total_loss / num_batches
