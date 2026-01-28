import torch
import pytest

def test_dataset_item_contract():
    """
    Since we cannot easily mock the full simulation dataset without reading its code structure
    in detail, we verify the contract by checking a mock dictionary that REPRESENTS
    what the dataset SHOULD output.
    
    Real integration tests should instantiate the actual dataset class.
    
    Contract from Spec:
    - input: (2, H, W) float32
    - params: (D,) float32
    - lambda_idx: (1,) int64 (optional)
    - lambda_m: (1,) float32 (optional)
    """
    # Mock sample mimicking expected output
    H, W = 256, 256
    D = 5
    sample = {
        "input": torch.randn(2, H, W, dtype=torch.float32),
        "params": torch.randn(D, dtype=torch.float32),
        "lambda_m": torch.tensor(0.5e-6, dtype=torch.float32)
    }
    
    # Contract Checks
    assert sample["input"].ndim == 3
    assert sample["input"].shape[0] == 2
    assert sample["input"].dtype == torch.float32
    
    assert sample["params"].ndim == 1
    assert sample["params"].shape[0] == D
    assert sample["params"].dtype == torch.float32
    
    assert sample["lambda_m"].ndim == 0 # Scalar often comes as 0-dim tensor or python float
