import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from tqdm import tqdm

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
        
        # Setup Optimizer and Loss
        lr = config.get("learning_rate", 1e-3)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        # Training State
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        
        # Paths
        self.checkpoint_dir = config.get("checkpoint_dir", "outputs/checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def train(self):
        epochs = self.config.get("epochs", 10)
        log_interval = self.config.get("log_interval", 10)
        
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(self.start_epoch, epochs):
            start_time = time.time()
            train_loss = self._train_epoch(epoch, log_interval)
            
            val_loss = 0.0
            if self.val_loader:
                val_loss = self._validate_epoch(epoch)
                
            end_time = time.time()
            epoch_time = end_time - start_time
            
            print(f"Epoch {epoch+1}/{epochs} | Time: {epoch_time:.2f}s | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            # Save Checkpoint
            is_best = False
            if self.val_loader and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                is_best = True
                self._save_checkpoint(epoch, val_loss, name="best_model.pth")
            
            # Always save latest
            self._save_checkpoint(epoch, val_loss, name="latest_checkpoint.pth")
            
    def _train_epoch(self, epoch, log_interval):
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} Train", leave=False)
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
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
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
        return total_loss / num_batches

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
