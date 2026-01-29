
import sys
import os
import torch
import shutil

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import load_config
from src.training.trainer import Trainer
from src.models.factory import get_model
from data.loaders.simulation import OnTheFlyDataset, GridDataset

def test_refinements():
    print("=== Testing 5-Parameter Refinements (Loop, Loss, Snapshot) ===")
    
    # 1. Setup minimal configs
    config = {
        'experiment_name': 'test_refinements_artifacts',
        'epochs': 6, # Enough to trigger snapshot (epoch 5) and alternate
        'batch_size': 4,
        'standardize_outputs': True,
        'loss_function': 'weighted_standardized', # Test one of the new ones
        'grid_strategy': 'mean',
        'fov_range': [1.0, 20.0],
        'output_dim': 5, # Model config
        'name': 'spectral_resnet',
        'modes': 2, # Very low for stability
        'resolution': 64 # Low for speed
    }
    
    # Clean up previous test
    if os.path.exists("outputs/test_refinements_artifacts"):
        shutil.rmtree("outputs/test_refinements_artifacts")
    
    # 2. Setup Data Loaders
    print("Initializing Loaders...")
    train_ds = OnTheFlyDataset(config, length=10)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=4)
    
    # Val Loader (Grid with offset)
    val_ds = GridDataset(config, steps=3, offset=True) # 3x3=9 samples
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=4)
    
    # 3. Setup Model
    print("Initializing Model...")
    model = get_model(config)
    
    # 4. Initialize Trainer
    print("Initializing Trainer...")
    trainer = Trainer(config, model, train_loader, val_loader)
    
    # Verify Fixed Anchor Loader initialized
    assert hasattr(trainer, 'fixed_train_loader'), "Trainer missing fixed_train_loader"
    print("Fixed Anchor Loader present.")
    
    # 5. Run Short Training Loop
    print("Running Training Loop for 6 epochs...")
    trainer.train()
    
    # 6. Verify Artifacts
    print("\nVerifying Artifacts...")
    snap_dir = "outputs/test_refinements_artifacts/snapshots/epoch_5"
    if os.path.exists(snap_dir) and os.listdir(snap_dir):
        print(f"Snapshot found at {snap_dir}: {os.listdir(snap_dir)}")
    else:
        print(f"ERROR: Snapshot not found at {snap_dir}")
        raise FileNotFoundError("Snapshot missing")
        
    print("\n=== Refinement Verification Passed ===")

if __name__ == "__main__":
    test_refinements()
