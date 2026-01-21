
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.models.factory import get_model
from src.training.trainer import Trainer

def test_activation_replacement():
    print("\n--- Testing Activation Replacement ---")
    config = {
        "name": "resnet18",
        "activation": "silu",
        "input_channels": 2,
        "output_dim": 3
    }
    model = get_model(config)
    
    # Check if any ReLU exists
    relu_count = 0
    silu_count = 0
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            relu_count += 1
        if isinstance(m, nn.SiLU):
            silu_count += 1
            
    print(f"ReLUs found: {relu_count} (Should be 0)")
    print(f"SiLUs found: {silu_count} (Should be > 0)")
    
    if relu_count == 0 and silu_count > 0:
        print("SUCCESS: Activation replacement worked.")
    else:
        print("FAILURE: Activation replacement failed.")
        exit(1)

def test_muon_and_scheduler():
    print("\n--- Testing Muon Optimizer & Scheduler ---")
    config = {
        "name": "spectral_resnet",
        "optimizer": "muon",
        "learning_rate": 0.01,
        "scheduler": "plateau",
        "scheduler_patience": 1,
        "loss_function": "mse",
        "epochs": 1,
        "modes": 2, # Reduce modes to fit small dummy input (64->4x4 feature map, rfft width=3)
        "log_interval": 1,
        "checkpoint_dir": "tests/temp_outputs",
        "log_dir": "tests/temp_outputs"
    }
    
    model = get_model(config)
    
    # Dummy Data
    x = torch.randn(4, 2, 64, 64)
    y = torch.randn(4, 3)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=2)
    
    trainer = Trainer(config, model, loader, loader)
    
    # Check Optimizer type
    print(f"Optimizer: {type(trainer.optimizer).__name__}")
    if type(trainer.optimizer).__name__ != "Muon":
        print("FAILURE: Optimizer is not Muon.")
        exit(1)
        
    # Check Scheduler existence
    print(f"Scheduler: {type(trainer.scheduler).__name__}")
    if not isinstance(trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
         print("FAILURE: Scheduler is not ReduceLROnPlateau.")
         exit(1)
         
    # Run 1 epoch
    print("Running 1 epoch training...")
    try:
        trainer.train()
        print("SUCCESS: Training loop with Muon/Scheduler completed without error.")
    except Exception as e:
        print(f"FAILURE: Training crashed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    test_activation_replacement()
    test_muon_and_scheduler()
