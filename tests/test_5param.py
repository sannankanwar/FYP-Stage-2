
import sys
import os
import torch
import torch.nn as nn
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.factory import get_model
from src.training.loss import PhysicsInformedLoss
from src.utils.normalization import ParameterNormalizer
from data.loaders.simulation import OnTheFlyDataset
from src.utils.config import load_config

def test_5param_pipeline():
    print("=== Testing 5-Parameter Inversion Pipeline ===")
    
    # 1. Load Configs
    print("Loading Configs...")
    data_config = load_config("configs/data.yaml")
    model_config = load_config("configs/model.yaml")
    
    print(f"Data Wavelength Range: {data_config['wavelength_range']}")
    print(f"Model Output Dim: {model_config['output_dim']}")
    assert model_config['output_dim'] == 5, "Model output dim should be 5"
    
    # 2. Test Dataset
    print("\nTesting Dataset...")
    test_config = data_config.copy()
    test_config['resolution'] = 256
    
    dataset = OnTheFlyDataset(
        config=test_config,
        length=10
    )
    
    inp, tgt = dataset[0]
    print(f"Input Shape: {inp.shape}")
    print(f"Target Shape: {tgt.shape} -> {tgt}")
    assert tgt.shape == (5,), f"Target shape should be (5,), got {tgt.shape}"
    
    # Check if wavelength and focal length are within range
    wl = tgt[3]
    fl = tgt[4]
    print(f"Sampled Wavelength: {wl}, Focal Length: {fl}")
    assert data_config['wavelength_range'][0] <= wl <= data_config['wavelength_range'][1]
    assert data_config['focal_length_range'][0] <= fl <= data_config['focal_length_range'][1]
    
    # 3. Test Normalizer
    print("\nTesting Normalizer...")
    ranges = {
        'xc': dataset.xc_range,
        'yc': dataset.yc_range,
        'fov': dataset.fov_range,
        'wavelength': dataset.wavelength_range,
        'focal_length': dataset.focal_length_range
    }
    normalizer = ParameterNormalizer(ranges)
    
    # Test batch normalization
    tgt_batch = torch.stack([tgt, tgt]) # (2, 5)
    tgt_norm = normalizer.normalize_tensor(tgt_batch)
    print(f"Normalized Target: {tgt_norm[0]}")
    assert tgt_norm.shape == (2, 5)
    assert torch.all(tgt_norm >= -1.1) and torch.all(tgt_norm <= 1.1), "Normalized values should be approx [-1, 1]"
    
    tgt_denorm = normalizer.denormalize_tensor(tgt_norm)
    print(f"Denormalized Target: {tgt_denorm[0]}")
    assert torch.allclose(tgt_denorm, tgt_batch, atol=1e-5), "Denormalization failed"
    
    # 4. Test Model
    print("\nTesting Model...")
    model_config['resolution'] = 256 # For spectral model
    model_config['modes'] = 8 # Reduce modes to fit 256 resolution test (16x16 feature map -> 9 freq bins)
    model = get_model(model_config)
    
    # Dummy forward pass
    inputs = inp.unsqueeze(0).float() # (1, 2, 256, 256)
    preds = model(inputs)
    print(f"Prediction Shape: {preds.shape} -> {preds}")
    assert preds.shape == (1, 5), "Prediction shape mismatch"
    
    # 5. Test Loss Function
    print("\nTesting PhysicsInformedLoss...")
    loss_fn = PhysicsInformedLoss(
        lambda_param=1.0, 
        lambda_physics=0.1, 
        normalizer=normalizer
    )
    
    # Prepare batch
    data_batch = inputs
    target_batch = tgt.unsqueeze(0) # (1, 5) real units
    
    # Preds are considered "normalized" output from model if we trained with standardization
    # So let's feed random normalized values
    pred_batch_norm = (torch.rand(1, 5) * 2 - 1).requires_grad_(True) # [-1, 1]
    
    total_loss, details = loss_fn(pred_batch_norm, target_batch, data_batch)
    print(f"Loss Details: {details}")
    
    # Check gradients
    total_loss.backward()
    print("Backward pass successful.")
    
    print("\n=== All Tests Passed ===")

if __name__ == "__main__":
    test_5param_pipeline()
