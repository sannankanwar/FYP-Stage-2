
import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.fno.fno_resnet18 import FNOResNet18

def verify_architecture():
    print("=== FNO-ResNet18 Architecture Verification ===")
    
    # Instantiate Model for 1024x1024
    model = FNOResNet18(
        in_channels=2, 
        output_dim=5, 
        input_resolution=1024,
        S_range=(1, 40)
    )
    
    # 1. Check Downsampler
    ds = model.downsampler
    print(f"\nDownsampler: {ds}")
    
    # Check layer 1 (Conv 2->8)
    # The first layer in sequential is index 0
    conv1_ds = ds[0]
    assert conv1_ds.in_channels == 2, f"Downsampler input channels must be 2, got {conv1_ds.in_channels}"
    assert conv1_ds.out_channels == 8, f"Downsampler layer 1 output must be 8, got {conv1_ds.out_channels}"
    
    # Check layer 4 (Conv 8->16) (0=conv, 1=bn, 2=relu, 3=conv...)
    # Index 3 is the second conv
    conv2_ds = ds[3]
    assert conv2_ds.in_channels == 8, f"Downsampler layer 2 input must be 8, got {conv2_ds.in_channels}"
    assert conv2_ds.out_channels == 16, f"Downsampler layer 2 output must be 16, got {conv2_ds.out_channels}"
    
    print("✅ Downsampler Channel Progression: 2 -> 8 -> 16 (Verified)")

    # 2. Check ResNet Stem Input
    stem_conv = model.conv1
    print(f"\nResNet Stem Conv: {stem_conv}")
    assert stem_conv.in_channels == 16, f"ResNet stem MUST accept 16 channels, got {stem_conv.in_channels}"
    print("✅ ResNet Stem accepts 16 channels (Verified)")
    
    # 3. Forward Pass Check
    print("\nRunning Forward Pass with (2, 2, 1024, 1024)...")
    x = torch.randn(2, 2, 1024, 1024)
    
    # Hook to check intermediate shape
    def hook_fn(module, input, output):
        print(f"Downsampler Output Shape: {output.shape}")
        assert output.shape[1] == 16, f"Expected 16 channels after downsampling, got {output.shape[1]}"
        assert output.shape[2] == 256, f"Expected 256x256 spatial, got {output.shape[2]}x{output.shape[3]}"

    handle = model.downsampler.register_forward_hook(hook_fn)
    
    out = model(x)
    print(f"Model Output Shape: {out.shape}")
    assert out.shape == (2, 5), f"Expected output (2, 5), got {out.shape}"
    
    print("✅ Forward pass successful")
    handle.remove()
    
    # 4. Backward Pass Check
    print("\nRunning Backward Pass...")
    target = torch.randn(2, 5)
    loss = nn.MSELoss()(out, target)
    loss.backward()
    
    print("Checking Gradients:")
    ds_grad_norm = model.downsampler[0].weight.grad.norm().item()
    stem_grad_norm = model.conv1.weight.grad.norm().item()
    fno_grad_norm = model.fno.spectral.weights1.grad.norm().item()
    
    print(f"  Downsampler Grad: {ds_grad_norm:.6f}")
    print(f"  ResNet Stem Grad: {stem_grad_norm:.6f}")
    print(f"  FNO Block Grad:   {fno_grad_norm:.6f}")
    
    assert ds_grad_norm > 0, "Downsampler has zero gradient!"
    assert stem_grad_norm > 0, "Stem has zero gradient!"
    assert fno_grad_norm > 0, "FNO has zero gradient!"
    print("✅ Gradients flowing successfully")

    print("\n=== All Architecture Checks Passed ===")

if __name__ == "__main__":
    verify_architecture()
