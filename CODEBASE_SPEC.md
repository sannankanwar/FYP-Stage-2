# Repository Specification and Implementation Guide

## 1. Executive Summary
This repository implements a **Deep Learning frameworks for Metalens Inverse Design and Parameter Inference**. The system is designed to predict geometric parameters (e.g., radius, height, gap) of metalens unit cells from desired optical phase maps, or conversely, to simulate phase maps from parameters.

It employs a wide variety of neural architectures (ResNets, UNets, Fourier Neural Operators (FNO), Swin Transformers, and GANs) and explores "Physics-Informed" training strategies (PINNs) where the loss function incorporates the underlying physical wave equations. The project is structured into progressive experimental phases (Baseline -> Physics-Informed -> Robust Losses -> Advanced Architectures).

## 2. Repo Map at a Glance
```text
.
├── configs/                 # Configuration Source of Truth (Hydra/OmegaConf style)
│   ├── experiments/         # Overrides for specific runs (Phase 1)
│   ├── experiments_2/       # Phase 2: Architecture Search & Hyperparams
│   ├── experiments_3/       # Phase 3: Field-of-View & Loss Terms
│   ├── experiments_4/       # Phase 4: FNO & Swin optimizations
│   ├── experiments_5/       # Phase 5: Sensitivity & Robustness
│   └── *.yaml               # Base configs (data, model, training)
├── data/                    # Data Handling
│   └── loaders/             # PyTorch Datasets/Dataloaders (Simulation vs Real)
├── real_data/               # Measured/Ground Truth CSVs (Metalens001...)
├── notebooks/               # Interactive analysis & prototyping
├── scripts/                 # Operational Entrypoints (train, eval, plot)
├── src/                     # Core Library Code
│   ├── inversion/           # Forward Physics Models (differentiable)
│   ├── models/              # Neural Network Definitions (Factories & Pytorch Mods)
│   ├── training/            # Loop logic, Loss functions, Optimizers
│   └── utils/               # Helpers (Config, Logging, Normalization)
└── tests/                   # Verification Suite (Unit & Integration tests)
```

## 3. System Goals and Problem Statement
### Goal
To build a robust, invertible mapping between:
1.  **Input**: Desired Phase Map $\phi(x, y)$ (Image/Tensor)
2.  **Output**: Geometric Parameters $\theta$ (Vector, e.g., $P_1, P_2, P_3, P_4, P_5$)

### Problem Type
-   **Regression**: Predicting continuous scalar values (params) from high-dimensional inputs.
-   **Inversion**: Solving the inverse scattering problem.
-   **Physics-Informed**: Constraining the solution space using known optical equations in `src/inversion/forward_model.py`.

## 4. Data Model and Domain Concepts
### key Entities
-   **Phase Map**: A 2D grid (likely grayscale image) representing the phase shift induced by the metalens.
    -   *Inferred Shape*: `(B, 1, H, W)` or `(B, 2, H, W)` (if using cos/sin embedding).
-   **Parameters (The "Label")**: The physical dimensions of the microscopic pillars.
    -   *Likely Params*: Height, Radius, Period, Duty Cycle, etc.
    -   *Inferred Shape*: `(B, K)` where $K \in \{3, 5\}$ (based on `test_5param.py`).
-   **Forward Model**: A function $F(\theta) \to \phi$ that simulates the physics.
-   **Real Data**: CSV files (`MetalensXXX.csv`) containing measured pairs of (Parameters, Phase Response).

### Flow
1.  **Online Simulation**: `data/loaders/simulation.py` generates synthetic $(\theta, \phi)$ pairs on the fly for infinite training data.
2.  **Offline Real Data**: `real_data` provides ground-truth validation points.

## 5. Configuration System
The system clearly uses a composed configuration system (likely Hydra or a custom merge function in `src/utils/config.py`).

### Canonical Schema (Inferred)
#### `training.yaml`
```yaml
batch_size: 32           # (Inferred)
epochs: 100              # (Inferred)
lr: 0.001                # (Inferred)
optimizer: "adam"        # (Inferred) [adam, sgd, adamw]
scheduler: "plateau"     # (Inferred) [plateau, cosine, step]
device: "cuda"           # (Inferred)
save_dir: "./checkpoints" # (Inferred)
```

#### `model.yaml`
```yaml
name: "resnet18"         # (Inferred) [resnet18, fno, unet, swin]
in_channels: 1           # (Inferred)
out_dim: 5               # (Inferred) typically 3 or 5 parameters
hidden_dim: 64           # (Inferred)
activation: "relu"       # (Inferred) [relu, gelu, silu, tanh]
physics_enabled: false   # (Inferred)
```

#### `data.yaml`
```yaml
dataset_type: "simulation" # (Inferred) [simulation, real]
image_size: 64             # (Inferred) [64, 128, 256]
params_count: 5            # (Inferred)
param_ranges:              # (Inferred) Min/Max for normalization
  p1: [0, 100]
  p2: [-1, 1]
```

### Hierarchy
1.  **Base Configs**: `training.yaml`, `model.yaml`, `data.yaml` defines defaults.
2.  **Experiment Overrides**: `configs/experiments/exp_*.yaml` overrides specific keys (e.g., `model.name`, `loss_function`) to define a unique run.

## 6. Training System
Located in `src/training/`.

### `src/training/trainer.py`
-   **Responsibility**: The main loop. Handles iterating over dataloader, zero_grad, backward, optimizer step.
-   **Public API (Inferred)**:
    -   `class Trainer(config, model, optimizer, scheduler, loss_fn)`
    -   `fit(train_loader, val_loader)`: Runs epochs.
-   **(Verify by)**: Checking if it handles mixed precision (AMP) or distributed training (DDP).

### `src/training/loss.py`
-   **Responsibility**: Defines the objective functions. Crucial for "Physics-Informed" vs "Data-Driven".
-   **Likely Definitions**:
    -   `MSELoss`: Standard regression loss $|\theta_{pred} - \theta_{gt}|^2$.
    -   `PhysicsLoss`: $|F(\theta_{pred}) - \phi_{input}|^2$. Re-simulates the phase from prediction and checks consistency.
    -   `SpectralLoss`: Fourier-domain distance.
    -   `RobustLoss`: Huber, LogCosh (seen in experiments `expS10_robust_huber.yaml`).

## 7. Models and Architectures
Located in `src/models/`.

### Factory (`factory.py`)
-   **Responsibility**: Strings $\to$ Class Instances.
-   `create_model(config_dict) -> nn.Module`

### Implementations
-   **ResNet (`inversion/resnet18.py`)**: Standard CNN backbone for regression. Strong baseline.
-   **FNO (`fno/`)**: Fourier Neural Operator. Good for resolution-invariant physics mapping.
    -   `fno_resnet18.py`, `fno_unet.py`: Hybrids? Or FNO layers injected into standard architectures.
-   **Swin (`transformer/swin.py`)**: Vision Transformer. hierarchical attention.
-   **GAN (`gan/inverter.py`)**: Likely an adversarial approach where a Discriminator ensures predicted parameters generate realistic phase maps.
-   **Hybrid (`hybrid.py`)**: Combinations (e.g., CNN + Spectral).

## 8. Inversion and Forward Model
### `src/inversion/forward_model.py`
-   **Critical Component**: This is the differentiable physics engine.
-   **Input**: Parameters $\theta$.
-   **Output**: Phase Map $\phi$.
-   **Usage**: Used inside `PhysicsLoss` to close the loop: $Input \to Model \to \theta_{pred} \to ForwardModel \to \phi_{reconstructed} \approx Input$.
-   **Failure Modes**: Gradient explosion if physics equations contain singularities.

## 9. Evaluation and Metrics
Located in `src/evaluation/`.

### Metrics
1.  **MSE/MAE (Parameter Space)**: How close are the dimensions?
2.  **Reconstruction Error (Phase Space)**: $|Forward(\theta_{pred}) - \phi_{gt}|$.
3.  **Inference Speed**: (Time per batch).

### `evaluate.py`
-   Runs validation set inference.
-   Generates residuals plots (`plot_residuals.py`).
-   Likely dumps results to a JSON or CSV.

## 10. Scripts and Operational Workflows
Crucial entry points for the "Agent".

### Primary Workflows
1.  **Train a Model**:
    ```bash
    python src/main.py --config configs/experiments/exp01_resnet_baseline.yaml
    # OR
    python scripts/train.py --config configs/experiments/exp01_resnet_baseline.yaml
    ```
2.  **Run Experiment Queue**:
    ```bash
    bash scripts/run_experiments_nohup.sh
    # or
    bash scripts/queue_experiments.sh
    ```
    *(These scripts likely iterate over config files and launch jobs).*
3.  **Evaluate Best Run**:
    ```bash
    python scripts/select_best_run.py --dir outputs/
    python scripts/evaluate.py --checkpoint best_model.pth
    ```
4.  **Visualize**:
    ```bash
    python scripts/visualize_reconstruction.py --model_path ...
    ```

## 11. Experiments and Reports
The repo is organized into strict "Generations" of experiments.

-   **`experiments` (Phase 1)**: Baselines. Resnet vs Spectral vs Physics. Activation functions (SiLU, Tanh, GELU).
-   **`experiments_2` (Phase 2)**: Systematic Architecture Search (A-Series), Optimizer/Scheduler tuning (B-Series), Physics Weight tuning (C-Series), Activations (D-Series).
-   **`experiments_3` (Phase 3)**: Refining "Field of View" (FOV) parameters and Loss types (Gradient vs Fringe specific losses).
-   **`experiments_5` (Phase 5)**: Robustness. Sensitivity analysis (`exp5_B03_physics_sensitivity.yaml`). Entropy minimization.

## 12. Tests, Verification, and Invariants
Located in `tests/`.

-   **`test_integrity.py`**: Likely checks if all imports work and file structure is valid.
-   **`verify_arch.py`**: Checks if models produce correct output shapes (e.g., $[B, 5]$).
-   **`test_5param.py`**: **Critical**. Verifies the expansion from 3-param to 5-param inversion logic works.
-   **`verify_losses.py`**: Checks if custom losses (Physics, Spectral) are differentiable and non-negative.
-   **`smoke_test_v2.py`**: A fast end-to-end run (1 epoch, tiny data) to ensure no runtime crashes.

## 13. Outputs, Artifacts, and Checkpoints
*(Inferred Locations)*
-   **Checkpoints**: `checkpoints/` or `experiments/<EXP_NAME>/weights/`.
-   **Logs**: `runs/` (Tensorboard/WandB).
-   **Plots**: `outputs/figures/`.

## 14. Dependency Graph and Call Flows
**(Inferred simplified flow)**
1.  **Entry**: `scripts/train.py`
2.  **Config**: `src/utils/config.py` loads YAMLs.
3.  **Data**: `data/loaders` instanties `SimulationDataset`.
4.  **Model**: `src/models/factory.py` builds `ResNet18`.
5.  **Train**: `src/training/trainer.py` loops.
    -   Calls `model(input)`
    -   Calls `src/training/loss.py` (which might call `src/inversion/forward_model.py`)
    -   Calls `optimizer.step()`
6.  **Save**: Dumps `.pth` to disk.

## 15. Known Gaps, Assumptions, and How to Verify
-   **Assumption**: The framework uses PyTorch Lightning?
    -   *Evidence*: Existence of `Trainer` class suggests either custom loop or Lightning. `uv.lock` would confirm.
    -   *(Verify by)*: Checking `src/training/trainer.py` or imports.
-   **Assumption**: Data is generated dynamically.
    -   *Evidence*: `loaders/simulation.py`.
    -   *(Verify by)*: Checking if `data.yaml` has a "path" key or just simulation params.
-   **Gap**: How are "Real" CSVs used?
    -   Are they used for training or strict validation?
    -   *(Verify by)*: Grepping for code that loads `real_data`.

## 16. Glossary
-   **FNO**: Fourier Neural Operator. Learn resolution-independent operators.
-   **PINN**: Physics-Informed Neural Network. Loss includes residuals of PDEs.
-   **Metalens**: Geometric optical surface structured at sub-wavelength scale.
-   **Phase Map**: The distribution of phase delays $\phi(x,y)$ created by the lens.
-   **Forward Model**: The analytic/physical equation $P(Params) \to Phase$.
-   **Inversion**: The ML task $M(Phase) \to Params$.
-   **SWIN**: Hierarchical Vision Transformer using Shifted Windows.
