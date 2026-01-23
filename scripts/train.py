import argparse
import yaml
import sys
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Ensure src is importable
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import load_config
from src.training.trainer import Trainer
from src.models.factory import get_model
from data.loaders.simulation import generate_single_sample, OnTheFlyDataset

def main():
    parser = argparse.ArgumentParser(description="Train the function inverse model.")
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="Path to training config")
    parser.add_argument("--model-config", type=str, default="configs/model.yaml", help="Path to model config")
    parser.add_argument("--data-config", type=str, default="configs/data.yaml", help="Path to data config")
    args = parser.parse_args()

    print("Loading configurations...")
    train_config = load_config(args.config)
    model_config = load_config(args.model_config)
    data_config = load_config(args.data_config)
    
    # Merge configs for convenience or keep separate?
    # Trainer expects one config dict mostly for hyperparameters.
    # Let's pass the merged training config + relevant bits.
    
    # Flatten config if nested (experiments format)
    full_config = data_config.copy() # Start with data defaults
    full_config.update(model_config) # Add model info
    full_config.update(train_config) # Add/Override with training specifics
    
    if 'training' in train_config:
        print("Detected nested config structure. Flattening 'training' section...")
        full_config.update(train_config['training'])
    if 'data' in train_config:
        full_config.update(train_config['data'])
    if 'model' in train_config:
        full_config.update(train_config['model'])
        
    print(f"Training for {full_config.get('epochs', 10)} epochs (Configured).")

    print("Initializing Data Loaders...")
    # Train Dataset
    train_samples = full_config.get("train_samples", data_config.get("train_samples", 1000))
    train_dataset = OnTheFlyDataset(full_config, length=train_samples)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=full_config.get("batch_size", 32), 
        shuffle=True, 
        num_workers=full_config.get("num_workers", 0)
    )
    
    # Val Dataset
    val_samples = full_config.get("val_samples", data_config.get("val_samples", 200))
    val_dataset = OnTheFlyDataset(full_config, length=val_samples)
    val_loader = DataLoader(
        val_dataset,
        batch_size=full_config.get("batch_size", 32), 
        shuffle=False,
        num_workers=full_config.get("num_workers", 0)
    )
    
    print(f"Initializing Model: {full_config.get('name')}...")
    model = get_model(full_config)
    
    print("Initializing Trainer...")
    trainer = Trainer(full_config, model, train_loader, val_loader)

    print("Starting training...")
    trainer.train()

if __name__ == "__main__":
    main()
