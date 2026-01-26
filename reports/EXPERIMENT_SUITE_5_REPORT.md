# Experiment Suite 5 Report: Hardening & Loss Benchmark

**Status:** Ready for Execution
**Suite:** FNOResNet18 5-Parameter Regression
**Guardrails:** Strict (Crash-on-Unknown, Physical-Protection, Drift-Check)

---

## 1. Hardening: What Changed?

To ensure this suite is trustworthy by construction, the following guardrails were implemented:

### 1.1 Legacy Trap (Kill Switch)
*   **Location:** `src/training/trainer.py`
*   **Action:** If `loss_function == "naive_5param"` is requested, the system raises `RuntimeError`.
*   **Message:** *"Naive5ParamMSELoss is deprecated and unsafe... permanently disabled."*

### 1.2 Strict Tensor Space Contract
*   **Location:** `src/utils/model_utils.py` -> `process_predictions`
*   **Action:**
    1.  Checks `hasattr(model, 'output_space')`. If missing -> `RuntimeError`.
    2.  Checks `output_space != 'unknown'`. If unknown -> `RuntimeError`.
    3.  If `output_space == 'physical'`, prevents denormalization even if config requests it.

### 1.3 Config Invariant Check
*   **Tool:** `tests/verify_exp5_invariants.py`
*   **Action:** Validates that `exp5_1`...`exp5_4` config files are **identical** for all keys except `loss_function`, `loss_weights`, and specific loss params.
*   **Effect:** Prevents "secret" hyperparameter drift (learning rate, batch size) from invalidating the loss comparison.

### 1.4 PINN Renaming
*   **Change:** `CompositePINNLoss` -> `PhysicsConsistencyLoss`.
*   **Reason:** "PINN" implies solving a PDE. We are strictly enforcing consistency with the forward optical model.

---

## 2. Experiment Design

**Base Model:** `FNOResNet18` (Physical Outputs)
**Dataset:** `OnTheFlyDataset` (Infinite random samples)
**Epochs:** 100 per run

### Experiment Matrix

| Run ID | Name | Loss Logic | Hypothesis |
| :--- | :--- | :--- | :--- |
| **5.1** | `unitstd` | Weighted MSE (Z-scores) | Baseline regression performance. |
| **5.2** | `gradflow` | Baseline + Gradient MSE | Improves fringe definition/sharpness. |
| **5.3** | `kendall` | Uncertainty-Weighted MSE | Balances multi-scale parameters (Wavelength vs Focal Length) automatically. |
| **5.4** | `pinn` | Baseline + 0.1 * Physics | Enforces consistency between predicted parameters and input phase image. |

### Coverage Metric
Ranked by:
1.  **Lowest Validation MSE** (Aggregation of all 5 parameters)
2.  (Tie) Highest Validation R²
3.  (Tie) Best Median Parameter Error

---

## 3. Diagrams

### 3.1 Tensor Space Flow
```mermaid
graph TD
    A[Input Phase Image] --> B[FNOResNet18]
    B -->|Physical Units (microns)| C{Output Space Check}
    C -- "Physical" --> D[Loss Function]
    C -- "Physical" --> E[Evaluator]
    D -->|Internal Normalization| F[Parameter Update]
    E -->|No Denorm (Safe)| G[Metrics/Plots]
    
    style C fill:#ccffcc,stroke:#006600
    style G fill:#ccffcc,stroke:#006600
```

### 3.2 Experiment Pipelines
```mermaid
graph LR
    subgraph Benchmark
        E1[Exp 5.1 UnitStd]
        E2[Exp 5.2 GradFlow]
        E3[Exp 5.3 Kendall]
        E4[Exp 5.4 Physics]
    end
    
    E1 --> S{Selector}
    E2 --> S
    E3 --> S
    E4 --> S
    
    S -->|Best MSE| W[Winner]
    W -->|Resume Checkpoint| EXT[Extension Run]
    EXT -->|Train +250 Epochs| FIN[Final Model (350 Epochs)]
```

---

## 4. How to Reproduce

### 4.1 Run the Benchmark (100 Epochs)
```bash
# Verify Configs (Safe by Construction)
python tests/verify_exp5_invariants.py

# Run Experiments
nohup python -m src.main --config configs/experiments_5_loss_study/exp5_1_unitstd.yaml --run_dir outputs_exp5/exp5_1 > outputs_exp5/exp5_1.log 2>&1 &
nohup python -m src.main --config configs/experiments_5_loss_study/exp5_2_gradflow.yaml --run_dir outputs_exp5/exp5_2 > outputs_exp5/exp5_2.log 2>&1 &
nohup python -m src.main --config configs/experiments_5_loss_study/exp5_3_kendall.yaml --run_dir outputs_exp5/exp5_3 > outputs_exp5/exp5_3.log 2>&1 &
nohup python -m src.main --config configs/experiments_5_loss_study/exp5_4_pinn.yaml --run_dir outputs_exp5/exp5_4 > outputs_exp5/exp5_4.log 2>&1 &
```

### 4.2 Select & Extend
```bash
# Analyze Results
python scripts/select_best_run.py --suite_dir outputs_exp5

# Extend (Example: if exp5_3 was best)
# Note: Manually point to the winner found by select_best_run.py
nohup python -m src.main --config configs/experiments_5_loss_study/exp5_3_kendall.yaml --run_dir outputs_exp5/exp5_3_extended --resume_checkpoint outputs_exp5/exp5_3/checkpoints/best_model.pth --epochs 350 > outputs_exp5/exp5_3_ext.log 2>&1 &
```

---

## 5. Creation Task: Why This Is Trustworthy

1.  **Automated Safety:** You cannot accidentally run a "legacy" loss or "unknown" model. The code refuses to run.
2.  **Structural Comparability:** We verify that experimental variables are isolated. `verify_exp5_invariants.py` proves that `lr`, `batch_size`, and `model` are identical across runs.
3.  **Methodological Rigor:** We do not guess the best model. `select_best_run.py` parses the training history and picks the winner based on validation MSE, removing human bias.
4.  **Extension Logic:** We extend the *best* model to 350 epochs to see asymptotic convergence, but only *after* proving it wins the sprint. This saves compute.

**Invalidation Conditions:**
*   If `verify_exp5_invariants.py` fails, the results are invalid (hidden variables).
*   If `output_space` logic is bypassed (e.g. by direct Tensor manipulation), results are invalid.
