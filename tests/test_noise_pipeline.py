import torch
import pytest
from src.noise.noise_pipeline import NoisePipeline, get_default_noise_config

def test_noise_determinism():
    cfg = get_default_noise_config()
    cfg['seed'] = 12345
    cfg['coordinate_warp']['enabled'] = True
    
    pipeline = NoisePipeline(cfg)
    
    B, H, W = 1, 32, 32
    phi = torch.zeros(B, H, W)
    
    # Run 1
    out1, _, _ = pipeline.apply(phi)
    
    # Run 2 - New Pipeline same seed
    pipeline2 = NoisePipeline(cfg)
    out2, _, _ = pipeline2.apply(phi)
    
    assert torch.allclose(out1, out2), "Noise pipeline must be deterministic with same seed"

def test_noise_randomness():
    cfg1 = get_default_noise_config()
    cfg1['seed'] = 12345
    cfg1['coordinate_warp']['enabled'] = True
    
    cfg2 = get_default_noise_config()
    cfg2['seed'] = 67890 # Different seed
    cfg2['coordinate_warp']['enabled'] = True
    
    p1 = NoisePipeline(cfg1)
    p2 = NoisePipeline(cfg2)
    
    phi = torch.zeros(1, 32, 32)
    out1, _, _ = p1.apply(phi)
    out2, _, _ = p2.apply(phi)
    
    assert not torch.allclose(out1, out2), "Different seeds must produce different noise"

def test_noise_shapes():
    cfg = get_default_noise_config()
    pipeline = NoisePipeline(cfg)
    B, H, W = 2, 32, 32
    phi = torch.randn(B, H, W)
    
    phi_out, img2_out, _ = pipeline.apply(phi)
    
    assert phi_out.shape == (B, H, W)
    assert img2_out.shape == (B, 2, H, W)
