
import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader

# Add root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import load_config
from src.models.factory import get_model
from src.models.refiner import ResNetRefiner
from src.training.refine_trainer import RefiningTrainer
from data.loaders.simulation import OnTheFlyDataset

def main():
    parser = argparse.ArgumentParser(description="Train Learned Optimizer (Refiner)")
    parser.add_argument("--config", type=str, required=True, help="Refinement experiment config")
    parser.add_argument("--baseline-config", type=str, required=True, help="Config of the baseline model (Exp9)")
    parser.add_argument("--baseline-checkpoint", type=str, required=True, help="Path to frozen baseline weights")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # 1. Load Configs
    refine_config = load_config(args.config)
    baseline_config_raw = load_config(args.baseline_config)
    
    # Merge for Data Config (Refiner uses same data distribution as baseline usually)
    # But Refiner might want different training Parameters (lr, epochs)
    full_config = baseline_config_raw.copy()
    full_config.update(refine_config) # Refiner config overrides
    
    # 2. Setup Data
    print("Initializing Data...")
    train_dataset = OnTheFlyDataset(full_config, length=full_config.get("train_samples", 2000))
    train_loader = DataLoader(train_dataset, batch_size=full_config.get("batch_size", 64), shuffle=True, num_workers=0)
    
    val_dataset = OnTheFlyDataset(full_config, length=full_config.get("val_samples", 200)) # Validation requires different seed logic if fixed?
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # 3. Load Baseline Model
    print("Loading Baseline Model...")
    baseline_model = get_model(baseline_config_raw)
    
    # Load Weights
    if os.path.exists(args.baseline_checkpoint):
        ckpt = torch.load(args.baseline_checkpoint, map_location='cpu')
        state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        # handle strictness (sometimes module. prefix exists)
        try:
            baseline_model.load_state_dict(state)
        except Exception as e:
            print(f"Standard load failed ({e}), trying strict=False...")
            baseline_model.load_state_dict(state, strict=False)
        print(f"Loaded baseline from {args.baseline_checkpoint}")
    else:
        print(f"WARNING: Baseline checkpoint not found at {args.baseline_checkpoint}. Using random weights (Garbage In, Garbage Out).")
    
    # 4. Init Refiner
    print("Initializing Refiner...")
    # Input channels = 4 (cos, sin, res_cos, res_sin)
    # Condition dim = 5 (xc, yc, S, f, lambda)
    # Output dim = 5 (deltas)
    refiner = ResNetRefiner(input_channels=4, condition_dim=5, output_dim=5)
    
    # 5. Training
    print("Starting Refiner Training...")
    trainer = RefiningTrainer(full_config, baseline_model, refiner, train_loader, val_loader)
    trainer.train()

if __name__ == "__main__":
    main()
