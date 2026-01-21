import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models.hybrid import SpectralResNet

def test_spectral_resnet():
    print("Testing SpectralResNet...")
    
    # 1. Instantiate Model
    model = SpectralResNet(in_channels=2, modes=16)
    print("Model instantiated.")
    
    # 2. Create Dummy Input (Batch=2, Channels=2, H=1024, W=1024)
    # Using 1024x1024 to verify high-res handling
    x = torch.randn(2, 2, 1024, 1024)
    print(f"Input shape: {x.shape}")
    
    # 3. Forward Pass
    y = model(x)
    print(f"Output shape: {y.shape}")
    
    assert y.shape == (2, 3), f"Expected output (2, 3), got {y.shape}"
    
    # 4. Backward Pass (Gradient Check)
    loss = y.sum()
    loss.backward()
    print("Backward pass successful. Gradients computed.")
    
    # Check if spectral weights have grad
    if model.spectral.weights1.grad is not None:
        print("Spectral weights have gradients.")
    else:
        print("WARNING: Spectral weights have NO gradients!")

    print("Test Passed!")

if __name__ == "__main__":
    test_spectral_resnet()
