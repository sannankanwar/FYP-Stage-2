# Experiment Suite 5: FNO-ResNet18 Loss Variants

**Status:** IMPLEMENTED & VERIFIED
**Base Model:** `FNOResNet18` (Physical Outputs)
**Goal:** Compare regression performance and physical consistency across 4 distinct loss formulations.

---

## 1. Overview

This suite tests whether incorporating physical constraints (Gradient Flow, PINN Residuals) or uncertainty weighting (Kendall) improves parameter estimation accuracy over standard weighted regression.

### Constants (Controlled Variables)
*   **Model Architecture:** `FNOResNet18` (with `HybridScaledOutput`).
*   **Dataset:** `OnTheFlyDataset` (Random sampling during training).
*   **Splits:** 80/10/10 (defined in config).
*   **Optimizer:** Adam (`lr=0.001`).
*   **Scheduler:** ReduceLROnPlateau.
*   **Seed:** Default system seed.

### Varied Factors
*   **Loss Function Logic:** (See Matrix below).
*   **Loss Hyperparameters:** (Weights, Learnable Variances).

---

## 2. Experiment Matrix

| Exp ID | Config Name | Loss Type | Formula (Simplified) | Extra Trainable Params? |
| :--- | :--- | :--- | :--- | :--- |
| **5.1** | `exp5_1_loss_unitstd.yaml` | **Baseline** (Weighted Std) | $L = \sum w_i (Z_{pred} - Z_{true})^2$ | No |
| **5.2** | `exp5_2_loss_gradflow.yaml` | **Gradient Consistency** | $L_{base} + \lambda \|\nabla \Phi_{recon} - \nabla \Phi_{input}\|^2$ | No |
| **5.3** | `exp5_3_loss_kendall.yaml` | **Kendall Uncertainty** | $\sum \frac{1}{2\sigma_i^2} L_i + \frac{1}{2}\log(\sigma_i^2)$ | **Yes** (5 LogVariances) |
| **5.4** | `exp5_4_loss_pinn.yaml` | **Composite PINN** | $L_{base} + 0.1 \cdot \|\text{Recon} - \text{Input}\|^2$ | No |

*   **$Z$**: Standardized (Z-score) value.
*   **$\Phi$**: Phase map.
*   **Recon**: Physics-based reconstruction from predicted parameters.

---

## 3. Base Model Contract: Physical Outputs

The experiment relies on a strict contract: **The Model outputs Physical Units (Microns).**

### 3.1 Source of Truth
In `src/models/fno/fno_resnet18.py`, the model explicitly defines its output space.

```python
# src/models/fno/fno_resnet18.py

class FNOResNet18(nn.Module):
    def __init__(self, ...):
        # ...
        self.scaled_output = HybridScaledOutput(...) 
        
        # EXPLICIT CONTRACT: Model outputs physical units (microns)
        # Downstream tools verify this before attempting denormalization.
        self.output_space = "physical"
```

### 3.2 Safety enforce mechanism
In `src/utils/model_utils.py`, downstream tools (Evaluation, Visualization) verify this contract to prevent "Double Denormalization" (exploding values).

```python
# src/utils/model_utils.py

def process_predictions(model, predictions, normalizer, config):
    output_space = getattr(model, 'output_space', 'unknown')
    
    if output_space == 'physical':
        # SAFE: Do NOT denormalize, even if config says 'standardize_outputs: true'
        pass 
    elif normalizer and config.get("standardize_outputs", False):
        predictions = normalizer.denormalize_tensor(predictions)
        
    return predictions
```

---

## 4. Loss Functions (Detailed)

All losses are implemented in `src/training/loss.py`.

### 4.1 Loss 1: Weighted Standardized Loss (Baseline)

*   **Purpose:** Standard regression using Z-scores to balance feature magnitudes.
*   **Interaction:** Takes **Physical** inputs, normalizes them internally using `ParameterNormalizer`, then computes MSE.

```python
class WeightedStandardizedLoss(nn.Module):
    def forward(self, pred_params, true_params, input_images=None):
        # Normalize INTERNAL to the loss
        pred_norm = self.normalizer.normalize_tensor(pred_params)
        true_norm = self.normalizer.normalize_tensor(true_params)
        
        # Weighted MSE
        diff = (pred_norm - true_norm)
        weighted_sq_diff = self.weights * (diff ** 2)
        return torch.mean(weighted_sq_diff), ...
```

*   **Failure Modes:**
    1.  `normalizer` is None: Falls back to raw MSE (scale imbalance).
    2.  `weights` mismatch: If weight tensor shape != (5,), broadcasing fails.

### 4.2 Loss 2: Gradient Consistency Loss

*   **Purpose:** Forces the model to predict parameters that generate the correct **fringe spacing** (spatial frequency).
*   **Logic:** Computes Sobel gradients of the Reconstructed Phase and matches them to Input Gradients.

```python
class GradientConsistencyLoss(nn.Module):
    def __init__(self, normalizer=None, gradient_weight=1.0):
        # ... Sobel filters registered as buffers ...

    def forward(self, pred_params, true_params, input_images):
        # 1. Base Loss
        loss_param = ... (Weighted Std)
        
        # 2. Reconstruct
        # Uses src.inversion.forward_model logic
        recon_img = reconstruct_phase(pred_params) 
        
        # 3. Gradient Loss
        target_grad = self._compute_gradient(input_images)
        pred_grad = self._compute_gradient(recon_img)
        loss_grad = self.mse(pred_grad, target_grad)
        
        return loss_param + self.gradient_weight * loss_grad, ...
```

*   **Failure Modes:**
    1.  **Phase Wrapping Discontinuities:** Determining gradient of wrapped phase is numerically unstable. We compute gradients of `[Cos, Sin]` representation (2 channels) to avoid this.
    2.  **Detach:** If `reconstruct_phase` breaks the graph (e.g. uses numpy), gradients die. (Verified: uses `torch`).

### 4.3 Loss 3: Kendall Uncertainty Loss

*   **Purpose:** Automatically tune the weights between the 5 parameters (xc, yc, S, wl, fl) based on prediction difficulty.
*   **Logic:** Learned `log_vars` ($s$) act as attenuation.

```python
class KendallUncertaintyLoss(nn.Module):
    def __init__(self, normalizer=None, init_var=0.0):
        # 5 learnable parameters
        self.log_vars = nn.Parameter(torch.full((5,), init_var))

    def forward(self, pred_params, true_params, ...):
        # ... Normalize ...
        squared_diff = (pred - true) ** 2
        precision = torch.exp(-self.log_vars)
        
        # Kendall Objective
        loss = 0.5 * precision * squared_diff + 0.5 * self.log_vars
        return loss.sum(dim=1).mean(), ...
```

*   **Config Keys:**
    ```yaml
    loss_function: "kendall"
    init_log_var: 0.0
    ```
*   **Critical Implementation Detail:**
    *   `Trainer` **must** add `criterion.parameters()` to the optimizer.
    *   **Verified:** `src/training/trainer.py` collects these parameters before `optim.Adam` initialization.

### 4.4 Loss 4: Composite PINN Loss

*   **Purpose:** Soft physics constraint.
*   **Formula:** `Total = Base + 0.1 * ReconstructionMSE`.

```python
class CompositePINNLoss(nn.Module):
    def __init__(self, normalizer=None, pinn_weight=0.1):
        self.param_loss = WeightedStandardizedLoss(normalizer=normalizer)
        self.pinn_weight = pinn_weight

    def forward(self, pred, true, input_images):
        loss_param, _ = self.param_loss(pred, true)
        
        # Physics Reconstruction
        recon_img = reconstruct_phase(pred) # Differentiable
        loss_physics = self.mse(recon_img, input_images)
        
        return loss_param + self.pinn_weight * loss_physics, ...
```

*   **Config Keys:**
    ```yaml
    loss_function: "pinn"
    pinn_weight: 0.1
    ```

---

## 5. Experiment Configs

Location: `configs/experiments_5_loss_study/`

### 5.1 Baseline
```yaml
# configs/experiments_5_loss_study/exp5_1_unitstd.yaml
name: "fno_resnet18"
loss_function: "weighted_standardized"
standardize_outputs: true
loss_weights: [1.0, 1.0, 1.0, 10.0, 10.0]
```

### 5.2 Gradient
```yaml
# configs/experiments_5_loss_study/exp5_2_gradflow.yaml
loss_function: "gradient_consistency"
gradient_weight: 1.0
```

### 5.3 Kendall
```yaml
# configs/experiments_5_loss_study/exp5_3_kendall.yaml
loss_function: "kendall"
init_log_var: 0.0
```

### 5.4 PINN
```yaml
# configs/experiments_5_loss_study/exp5_4_pinn.yaml
loss_function: "pinn"
pinn_weight: 0.1
```

**Running:**
```bash
./run_experiments_5.sh
```

---

## 6. Evaluation & Logging

### Metrics
*   **Total Loss**: `total_loss` (varies by function).
*   **Parameter MSE**: `loss_param` (comparable across all).
*   **Components**: `loss_grad`, `loss_physics`, `sigma_xc` (logged in `details` dict).

### Safety
Evaluation uses `scripts/evaluate.py`.
*   It imports `process_predictions`.
*   It detects `model.output_space == 'physical'`.
*   It skips denormalization.
*   **Result:** Plots show correct physical units (Microns), not 50,000 range.

---

## 7. Validation (Smoke Tests)

File: `tests/test_loss_variants.py`

**Test Logic for Each Loss:**
1.  Instantiate `FNOResNet18`.
2.  Create dummy batch `(B, 2, 64, 64)`.
3.  Forward pass -> `pred` (Physical).
4.  Compute Loss.
5.  `loss.backward()`.
6.  **Assert:** `model.parameters()` have non-None gradients.
7.  **Assert (Kendall):** `loss_func.log_vars.grad` is not None.

**Result:** PASS (Verified in shell).

---

## 8. Final Checklist

- [x] **Only loss differs across experiments** (Configs extend base params).
- [x] **Output-space contract enforced** (Explicit 'physical' attribute).
- [x] **No denormalize on physical outputs** (Verified via `test_integrity.py`).
- [x] **Kendall params included in optimizer** (Trainer updated to add `criterion.parameters()`).
- [x] **PINN weight exactly 0.1 in experiment 4** (Hardcoded/Configured default).
- [x] **Smoke tests pass** (`tests/test_loss_variants.py` Passed).

**Verdict:** The experiment suite is structurally sound, safe from scaling bugs, and ready for execution.
