import os
os.environ['MPLBACKEND'] = 'Agg'
import sys
import torch
import torch.nn as nn
import shutil

# Ensure we can import src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.training.refine_trainer import RefiningTrainer
from src.models.refiner import ResNetRefiner

class MockDataset:
    def __init__(self):
        self.xc_range = (0, 100)
        self.yc_range = (0, 100)
        self.S_range = (10, 50)
        self.wavelength_range = (0.4, 0.7)
        self.focal_length_range = (10, 100)

class MockLoader(list):
    def __init__(self, dataset):
        self.dataset = dataset
        # 2 Batches
        # Data: (B, 2, 64, 64), Target: (B, 5)
        for _ in range(2):
            self.append((torch.randn(2, 2, 64, 64), torch.randn(2, 5)))

class MockBaseline(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        B = x.shape[0]
        # Return dummy params (B, 5)
        out = torch.tensor([50., 50., 20., 0.5, 50.], device=x.device).repeat(B, 1)
        out += torch.randn_like(out) * 0.1
        return out

def main():
    print(">>> Setting up Refinement Verification...")
    
    config = {
        "experiment_name": "debug_refiner",
        "output_dir": "outputs/debug_refiner",
        "epochs": 1,
        "batch_size": 2,
        "optimizer": "adam",
        "data": {
            "params": ['xc', 'yc', 'S', 'f', 'lambda'],
            # Ranges Required by Loss Module
            "xc_range": [0, 100],
            "yc_range": [0, 100],
            "S_range": [10, 50],
            "wavelength_range": [0.4, 0.7],
            "focal_length_range": [10, 100],
        }
    }
    
    # Cleanup output
    if os.path.exists(config['output_dir']):
        try:
            shutil.rmtree(config['output_dir'])
        except Exception:
            pass
    os.makedirs(config['output_dir'], exist_ok=True)
    
    ds = MockDataset()
    loader = MockLoader(ds)
    baseline = MockBaseline()
    refiner = ResNetRefiner(input_channels=4, condition_dim=5, output_dim=5)
    
    print(">>> Initializing Trainer...")
    trainer = RefiningTrainer(config, baseline, refiner, loader, loader)
    
    # Also disable AMP for test to ensure stability on CPU/MPS
    trainer.use_amp = False
    
    print(">>> Triggering Snapshot (expect crash if buggy)...")
    try:
        # We manually call the method that caused the crash
        trainer._save_snapshot(0)
        print(">>> FAIL: Snapshot DID NOT CRASH (Unexpected).")
    except TypeError as e:
        print(f">>> CAUGHT EXPECTED CRASH: {e}")
        # We successfully reproduced the bug. Exit clean or with specific code?
        # Let's exit 0 to indicate "Reproduction Successful"
        print(">>> Reproduction Successful.")
    except Exception as e:
        print(f">>> CRASHED WITH DIFFERENT ERROR: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()
