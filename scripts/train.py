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
    # The default training.yaml is flat, but experiment yamls have 'training', 'data', 'model' sections
    full_config = train_config.copy()
    if 'training' in train_config:
        print("Detected nested config structure. Flattening 'training' section...")
        full_config.update(train_config['training'])
        
    # Also merge data config if it overrides anything? 
    # Usually data config is separate, but experiments configs often have data params too.
    if 'data' in train_config:
        full_config.update(train_config['data'])
        
    print(f"Training for {full_config.get('epochs', 10)} epochs (Configured).")
    
    # Update loaded configs with overrides from experiment config if present
    # This is tricky because model_config might be separate.
    # For now, let's just ensure Trainer gets the right flatten dict for hyperparameters.

    print("Initializing Data Loaders...")
    # Train Dataset
    train_dataset = OnTheFlyDataset(data_config, length=data_config.get("train_samples", 1000))
    train_loader = DataLoader(
        train_dataset, 
        batch_size=full_config.get("batch_size", 32), 
        shuffle=True, 
        num_workers=data_config.get("num_workers", 0)
    )
    
    # Val Dataset (Validation is also random generation for now, but could be fixed seed)
    val_dataset = OnTheFlyDataset(data_config, length=data_config.get("val_samples", 200))
    val_loader = DataLoader(
        val_dataset,
        batch_size=full_config.get("batch_size", 32), 
        shuffle=False,
        num_workers=data_config.get("num_workers", 0)
    )
    
    print(f"Initializing Model: {model_config.get('name')}...")
    model = get_model(model_config)
    
    print("Initializing Trainer...")
    trainer = Trainer(full_config, model, train_loader, val_loader)

    print("Starting training...")
    trainer.train()

if __name__ == "__main__":
    main()
