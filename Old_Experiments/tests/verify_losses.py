
import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.training.loss import (
    RawPhysicsLoss, 
    AdaptivePhysicsLoss, 
    WeightedPhysicsLoss, 
    AuxiliaryPhysicsLoss
)
from src.utils.normalization import ParameterNormalizer

def verify_losses():
    print("=== Verifying S-Parameter Loss Functions ===")
    
    # Setup Dummy Data
    B, C, H, W = 2, 2, 256, 256
    input_images = torch.randn(B, C, H, W)
    
    # True Params (Physical Units)
    # [xc, yc, S, wl, fl]
    true_params = torch.tensor([
        [100.0, 100.0, 20.0, 0.5, 50.0],
        [-50.0, -50.0, 10.0, 0.6, 80.0]
    ])
    
    # Pred Params (Physical for Raw/Adaptive, can be Normalized for Weighted/Aux)
    # Let's assume we test consistent usage.
    
    pred_params_phys = (true_params + 0.1).clone().detach().requires_grad_(True)
    
    # Normalizer for Weighted/Aux losses
    ranges = {
        'xc': [-500, 500],
        'yc': [-500, 500],
        'S': [1, 40],
        'wavelength': [0.4, 0.7],
        'focal_length': [10, 100]
    }
    normalizer = ParameterNormalizer(ranges)
    pred_params_norm = normalizer.normalize_tensor(pred_params_phys)
    
    print("\n--- 1. RawPhysicsLoss (expS01) ---")
    loss_fn = RawPhysicsLoss()
    loss, details = loss_fn(pred_params_phys, true_params, input_images) # Expects purely physical
    print(f"Loss: {loss.item():.6f}")
    print(f"Details: {details}")
    loss.backward()
    print("✅ Backward pass successful")
    
    print("\n--- 2. AdaptivePhysicsLoss (expS02) ---")
    loss_fn = AdaptivePhysicsLoss()
    # Expects physical params? Code check:
    # forward(pred_params, true_params...) -> diff = (pred - true)**2
    # Logic uses reconstruction physical formula on pred_params.
    # So yes, expects physical.
    loss, details = loss_fn(pred_params_phys, true_params, input_images)
    print(f"Loss: {loss.item():.6f}")
    print(f"Details: {details}")
    loss.backward()
    print("✅ Backward pass successful")
    
    print("\n--- 3. WeightedPhysicsLoss (expS03) ---")
    loss_fn = WeightedPhysicsLoss(normalizer=normalizer)
    # Expects Normalized Preds (if normalizer provided)?
    # Code check: param_loss(pred_params, true_params)
    # param_loss is WeightedStandardizedLoss.
    # WeightedStandardizedLoss (FIXED) normalizes pred_params if normalizer exists.
    # Wait, if `pred_params` passed IN are physical (which HybridScaledOutput produces),
    # then `WeightedStandardizedLoss` will normalize them.
    # If `pred_params` passed IN are normalized, then we shouldn't pass normalizer to WeightedStandardizedLoss?
    # BUT `WeightedPhysicsLoss` ALSO does reconstruction.
    # Reconstruction logic:
    # if self.normalizer: pred = denormalize(pred)
    # else: pred = pred
    #
    # So `WeightedPhysicsLoss` expects NORMALIZED inputs if `normalizer` is provided.
    # IF `pred_params` are physical (from HybridScaledOutput), then we should NOT pass `normalizer` to `WeightedPhysicsLoss`?
    # NO. If outputs are physical, we want reconstruction to use them AS IS.
    # But `param_loss` (WeightedStandardizedLoss) wants to compare standardized values.
    #
    # CASE A: Standardize Outputs = True (expS03)
    # Trainer passes `normalizer` to `WeightedPhysicsLoss`.
    # Model outputs PHYSICAL values.
    # `WeightedPhysicsLoss` receives PHYSICAL values.
    # `param_loss` (WeightedStandardizedLoss) gets PHYSICAL values.
    #   -> Internally normalizes them (my fix). Compares with normalized True. Correct.
    # `reconstruction` part checks `if self.normalizer: pred = denormalize(pred)`.
    #   -> If it denormalizes PHYSICAL values, it gets NONSENSE.
    #   -> PHYSICAL values denormalized? No.
    #   -> If input is ALREADY physical, we should NOT denormalize.
    #
    # BUG IDENTIFIED: `WeightedPhysicsLoss` assumes if normalizer is present, input is NORMALIZED.
    # But `HybridScaledOutput` returns PHYSICAL values.
    # Trainer instantiates `WeightedPhysicsLoss(normalizer=norm)` if `standardize_outputs=True`.
    # This creates a conflict.
    #
    # If `standardize_outputs=True`, we want the PARAM LOSS to be on standardized scale.
    # But the MODEL output is physical.
    # Reconstruction needs physical.
    #
    # Fix needed in `WeightedPhysicsLoss`: 
    # If input is physical, do NOT denormalize for reconstruction.
    # But how does it know?
    # We should assume `WeightedPhysicsLoss` works with whatever format the model outputs.
    # 
    # Current code:
    # 1. param_loss: uses weights.
    # 2. reconstruction: denormalizes if normalizer exists.
    #
    # If we pass physical values:
    # 1. param_loss normalizes then compares. (Good, since weights make sense on standardized scale).
    # 2. reconstruction denormalizes?? BAD.
    #
    # I MUST FIX `WeightedPhysicsLoss` (and `AuxiliaryPhysicsLoss`) to NOT denormalize for reconstruction
    # if the input is already physical.
    # But wait, does standard implementation use `HybridScaledOutput`? Yes `fno_resnet18` uses it.
    # So `pred_params` ARE physical.
    # So `WeightedPhysicsLoss` should NEVER denormalize `pred_params` for reconstruction if they are physical.
    # But `denormalize` is only needed if `pred_params` WERE normalized.
    #
    # Conclusion: `WeightedPhysicsLoss` should ASSUME input is compatible with reconstruction (i.e. physical)
    # OR we must handle normalized inputs.
    #
    # Since we moved to `HybridScaledOutput` (Physical Output), `WeightedPhysicsLoss` logic "if normalizer: denormalize" is WRONG.
    # It should be "if normalizer: use it for param_loss standardization, but for reconstruction use raw (since it's physical)".
    #
    # Let's verify this hypothesis with the script.
    
    print("...testing WeightedPhysicsLoss logic...")
    try:
        # Pass PHYSICAL params
        loss, details = loss_fn(pred_params_phys, true_params, input_images)
        print(f"Loss: {loss.item():.6f}")
        print(f"Details: {details}")
        loss.backward()
    except Exception as e:
        print(f"FAILED: {e}") 
        
    print("\n--- 4. AuxiliaryPhysicsLoss (expS04) ---")
    loss_fn = AuxiliaryPhysicsLoss(normalizer=normalizer)
    # Same logic applies. Expected Physical Inputs.
    try:
        loss, details = loss_fn(pred_params_phys, true_params, input_images)
        print(f"Loss: {loss.item():.6f}")
        print(f"Details: {details}")
        loss.backward()
    except Exception as e:
        print(f"FAILED: {e}") 

if __name__ == "__main__":
    verify_losses()
