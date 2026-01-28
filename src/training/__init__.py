"""Training module for metalens phase-map inversion."""

from src.training.loss import (
    # Factory function (primary interface)
    build_loss,
    # Core loss classes
    CompositeLoss,
    UnitStandardizedParamLoss,
    PhaseGradientFlowLoss,
    KendallUnitStandardizedLoss,
    CustomRegressionLoss,
    PhysicsPhaseMapMSELoss,
    # Helper functions
    extract_param_ranges,
    build_full_params_for_reconstruction,
    reconstruct_phase_map,
    # Constants
    PARAM_ORDER,
    ALLOWED_WAVELENGTHS_M,
)

__all__ = [
    # Factory
    "build_loss",
    # Composite
    "CompositeLoss",
    # Regression losses
    "UnitStandardizedParamLoss",
    "PhaseGradientFlowLoss",
    "KendallUnitStandardizedLoss",
    "CustomRegressionLoss",
    # Physics loss
    "PhysicsPhaseMapMSELoss",
    # Helpers
    "extract_param_ranges",
    "build_full_params_for_reconstruction",
    "reconstruct_phase_map",
    # Constants
    "PARAM_ORDER",
    "ALLOWED_WAVELENGTHS_M",
]
