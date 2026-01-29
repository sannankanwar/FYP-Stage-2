
import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader

# Add root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import load_config
from src.models.factory import get_model
from src.training.active_trainer import ActiveTrainer
from src.data.sampling.difficulty_sampler import DifficultySampler
from data.loaders.simulation import OnTheFlyDataset

def main():
    parser = argparse.ArgumentParser(description="Train Difficulty-Aware Active Learning")
    parser.add_argument("--config", type=str, required=True, help="Experiment config")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # 1. Load Config
    config = load_config(args.config)
    
    # 2. Init Sampler
    print("Initializing Difficulty Sampler...")
    sampler = DifficultySampler(config.get("data", {}))
    
    # 3. Setup Data
    print("Initializing Data...")
    # Critical: num_workers=0 to allow sampler state updates in main process
    train_dataset = OnTheFlyDataset(config, length=config.get("train_samples", 2000), sampler=sampler)
    train_loader = DataLoader(train_dataset, batch_size=config.get("batch_size", 64), shuffle=True, num_workers=0)
    
    val_dataset = OnTheFlyDataset(config, length=config.get("val_samples", 200)) # Validation is random uniform? Yes.
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # 4. Init Model
    print("Initializing Model...")
    model = get_model(config)
    
    # 5. Training
    print("Starting Active Training...")
    trainer = ActiveTrainer(config, model, train_loader, val_loader, sampler=sampler)
    trainer.train()

if __name__ == "__main__":
    main()
