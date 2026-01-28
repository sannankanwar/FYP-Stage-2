import torch
import pytest

# Skip this entire file if CUDA is not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

def test_cuda_tensors_basic():
    """Ensure we can allocate and move tensors to GPU."""
    x = torch.randn(10).cuda()
    y = torch.randn(10).cuda()
    z = x + y
    assert z.device.type == 'cuda'
    assert z.shape == (10,)

def test_cuda_model_forward():
    """Simple smoke test for a convolution operation on CUDA."""
    B, C, H, W = 2, 2, 64, 64
    x = torch.randn(B, C, H, W).cuda()
    conv = torch.nn.Conv2d(2, 4, 3, padding=1).cuda()
    out = conv(x)
    assert out.shape == (B, 4, H, W)
    assert out.device.type == 'cuda'
