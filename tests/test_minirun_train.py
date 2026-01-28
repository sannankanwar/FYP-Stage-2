import torch
import pytest
import torch.nn as nn
import torch.optim as optim
from src.training.loss import UnitStandardizedParamLoss, CompositeLoss

# Skip if CUDA unavailable (optional, but intended for GPU validation)
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

class SimpleModel(nn.Module):
    def __init__(self, out_dim=5):
        super().__init__()
        self.conv = nn.Conv2d(2, 8, 3, padding=1)
        self.flat = nn.Flatten()
        # 64*64 input -> 8*64*64
        self.fc = nn.Linear(8 * 32 * 32, out_dim) # Assuming 32x32 input

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.flat(x)
        return self.fc(x)

def test_mini_training_loop():
    """Run a tiny training loop to ensure backprop works without crashing."""
    device = torch.device("cuda")
    B, H, W = 4, 32, 32
    
    # 1. Setup Model
    model = SimpleModel(out_dim=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 2. Setup Loss
    ranges = {"xc":(-1,1),"yc":(-1,1),"S":(-1,1),"f":(-1,1),"lambda":(-1,1)}
    reg_loss = UnitStandardizedParamLoss(["xc","yc","S","f","lambda"], ranges)
    loss_fn = CompositeLoss(reg_loss)
    
    # 3. Loop
    for i in range(5):
        optimizer.zero_grad()
        
        # Synthetic data
        inp = torch.randn(B, 2, H, W).to(device)
        target = torch.randn(B, 5).to(device)
        
        # Forward
        preds = model(inp)
        loss, _ = loss_fn(preds, target)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        assert not torch.isnan(loss), "Loss became NaN"
        assert loss.item() > 0
    
    print("Mini training loop passed.")
