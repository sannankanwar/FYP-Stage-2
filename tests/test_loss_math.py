import torch
import pytest
from src.training.loss import (
    UnitStandardizedParamLoss, 
    PhaseGradientFlowLoss, 
    CompositeLoss, 
    PhysicsPhaseMapMSELoss
)

@pytest.fixture
def dummy_ranges():
    return {
        "xc": (-10.0, 10.0), # range = 20
        "yc": (-10.0, 10.0),
        "S": (0.0, 1.0),
        "f": (0.0, 100.0),
        "lambda": (0.4, 0.7)
    }

def test_unit_std_loss_math(dummy_ranges):
    loss_fn = UnitStandardizedParamLoss(["xc"], dummy_ranges)
    
    # Range of xc is 20.
    # Pred = 5, True = 0. Error = 5.
    # Loss = (5^2) / (20^2) = 25 / 400 = 0.0625
    
    pred = torch.tensor([[5.0]])
    target = torch.tensor([[0.0]])
    
    loss, metrics = loss_fn(pred, target)
    assert torch.isclose(loss, torch.tensor(0.0625))

def test_gradflow_loss_math(dummy_ranges):
    # Old gradflow definition but using new 1/range weighting
    loss_fn = PhaseGradientFlowLoss(["xc"], dummy_ranges)
    
    # Range of xc is 20.
    # Pred = 5, True = 0. Error Sq = 25.
    # Loss = 25 * (1/20) = 1.25
    
    pred = torch.tensor([[5.0]])
    target = torch.tensor([[0.0]])
    
    loss, metrics = loss_fn(pred, target)
    assert torch.isclose(loss, torch.tensor(1.25))

def test_physics_scheduling_gate(dummy_ranges):
    reg_loss = UnitStandardizedParamLoss(["xc"], dummy_ranges)
    
    # Physics weight 1.0, start epoch 10
    comp_loss = CompositeLoss(
        regression_loss=reg_loss,
        physics_enabled=True,
        physics_weight=1.0,
        physics_start_epoch=10,
        predicted_params=["xc"]
    )
    
    # Create inputs
    B = 1
    pred_params = torch.zeros(B, 1) # xc=0 (physically valid in range)
    true_params = torch.zeros(B, 1)
    # Mock input images with mismatch to ensure physics loss would be non-zero
    input_images = torch.ones(B, 2, 32, 32) 
    
    # Before Gate
    loss_early, m_early = comp_loss(pred_params, true_params, input_images, {}, epoch=5)
    assert m_early["physics_active"] is False
    assert m_early.get("loss_physics") is None
    
    # After Gate
    loss_late, m_late = comp_loss(pred_params, true_params, input_images, 
                                  batch={"lambda_m": torch.tensor([0.5e-6])}, # Needed for recon
                                  epoch=11)
    assert m_late["physics_active"] is True
    assert "loss_physics" in m_late
    assert m_late["loss_physics"] > 0
