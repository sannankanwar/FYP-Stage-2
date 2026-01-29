# Incident Report: Exploding Predictions in Visualization

**Date:** 2026-01-26
**Severity:** High (Misleading Debug Information) / Low (Functional/Training Impact)
**Status:** Resolved

## 1. Executive Summary
During the testing of the new FNO-ResNet18 architecture with standardized training (S09-S14), the validation plots (snapshots) displayed predicted parameters that were orders of magnitude larger than the ground truth (e.g., predicted `xc` ~60,000 μm vs true `100` μm).

**Conclusion:** The model was training correctly, but the **visualization code in `Trainer.py` contained a logic error** that applied "denormalization" to values that were already physical (unnormalized), resulting in a "Double Denormalization" effect.

---

## 2. Technical Root Cause

### A. The Conflict
The system had two competing design patterns active simultaneously:

1.  **Model Output (New Pattern):**
    The `FNOResNet18` architecture was updated to use `HybridScaledOutput`. This layer enforces physical scaling *inside the model*.
    *   **Result:** `model(x)` returns **Physical Units** (e.g., 100.0).

2.  **Trainer Optimization (Legacy Pattern):**
    The experiments (S09-S14) utilized `standardize_outputs: true` to stabilize gradients.
    *   **Trainer Assumption:** "If standardization is on, the model must be predicting normalized Z-scores (e.g., 0.5)."
    *   **Trainer Action:** "I must denormalize the output before plotting."

### B. The Arithmetic Error
The Trainer took the physical output (100.0) and treated it as a normalized score.

**Formula Applied:** `Prediction_Plot = (Model_Output * Sigma) + Mean`

**Example (`xc` parameter):**
*   **True Value:** 100 μm
*   **Model Prediction:** 100 μm (Correct)
*   **Standard Deviation ($\sigma$):** ~500 μm (Range [-500, 500])
*   **Visualization Calculation:**
    $$100.0 \times 500.0 + 0 = 50,000.0$$

The visualization code displayed `50,000` instead of `100`, falsely indicating the model had "exploded".

---

## 3. Impact Assessment

### Training (Unaffected) ✅
The training loop uses robust loss functions (e.g., `WeightedStandardizedLoss`).
*   **Mechanism:** The loss function receives the physical output from the model. It then **internally normalizes** this output before comparing it to the standardized targets.
*   **Result:** The gradients were calculated correctly. The model learned the correct physical values.

### Visualization (Broken) ❌
*   **Snapshots (`_save_snapshot`):** Scatter plots showed massive scaling errors.
*   **Residuals (`_plot_residual_phase`):** Computed phase maps using the exploded parameters (Example: wavelength=20,000 μm instead of 0.5 μm), causing phase map generation to fail or look like static noise.

---

## 4. Resolution

### Code Fix
I modified `src/training/trainer.py` to remove the conditional denormalization logic in the visualization methods.

**Validation Loop (`_validate_epoch`):**
*   **Before:** `if standardizing: denormalize(output)`
*   **After:** `output` is used directly (as it is known to be physical).

### Verification
I verified that `FNOResNet18` uses the `HybridScaledOutput` layer, guaranteeing physical outputs. The fix aligns the Trainer with this architecture guarantee.

**Status:** Code is patched. Re-running visualization will now show correct values (Overlaying True vs Pred bars will line up).
