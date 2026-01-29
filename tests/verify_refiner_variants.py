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

# Re-use Mocks
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
        for _ in range(2):
            self.append((torch.randn(2, 2, 64, 64), torch.randn(2, 5)))

class MockBaseline(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        B = x.shape[0]
        out = torch.tensor([50., 50., 20., 0.5, 50.], device=x.device).repeat(B, 1)
        out += torch.randn_like(out) * 0.1
        return out

def test_variant(mode_name, config_override):
    print(f"\n>>> Testing Variant: {mode_name}")
    
    config = {
        "experiment_name": f"test_{mode_name}",
        "output_dir": f"outputs/test_{mode_name}",
        "epochs": 1,
        "batch_size": 2,
        "optimizer": "adam",
        "loss": config_override,
        "data": {
            "params": ['xc', 'yc', 'S', 'f', 'lambda'],
            "xc_range": [0, 100], "yc_range": [0, 100],
            "S_range": [10, 50], "wavelength_range": [0.4, 0.7],
            "focal_length_range": [10, 100],
        }
    }
    
    if os.path.exists(config['output_dir']):
        try: shutil.rmtree(config['output_dir'])
        except: pass
    os.makedirs(config['output_dir'], exist_ok=True)
    
    ds = MockDataset()
    loader = MockLoader(ds)
    baseline = MockBaseline()
    refiner = ResNetRefiner(4, 5, 5)
    
    trainer = RefiningTrainer(config, baseline, refiner, loader, loader)
    trainer.use_amp = False # Stability
    
    # Run 1 epoch
    print("  > Running Train Epoch...")
    loss = trainer._train_epoch(0, loader, 10)
    print(f"  > Train Loss: {loss:.4f}")
    
    # Run Snapshot
    print("  > Running Snapshot...")
    trainer._save_snapshot(0)
    print(f"  > {mode_name} Verified.")

def main():
    # 1. GradFlow
    test_variant("gradflow", {"mode": "gradient_flow", "physics_enabled": False})
    
    # 2. PINN
    test_variant("pinn", {"mode": "unit_standardized", "physics_enabled": True, "physics_weight": 0.5})

if __name__ == "__main__":
    try:
        main()
        print("\n>>> ALL VARIANTS PASSED.")
    except Exception as e:
        print(f"\n>>> FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
