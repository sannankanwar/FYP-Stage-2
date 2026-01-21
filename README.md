# Function Inverse Research

Research project for approximating function inverses using Deep Learning, specifically focusing on metalens parameter inversion.

## Table of Contents

- [Function Inverse Research](#function-inverse-research)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Directory Structure](#directory-structure)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Training](#training)
    - [Evaluation](#evaluation)
    - [Inversion](#inversion)
  - [Configuration](#configuration)
  - [Contributing](#contributing)

## Project Overview

This project implements deep learning models to solve the inverse problem for metalens design. Given a desired phase response or spectral characteristic, the model predicts the geometric parameters (e.g., Width, Length) required to achieve it.

Key features:
*   **Hybrid Models**: Combining ResNet stems with Spectral Gating (FFT) for global context.
*   **Simulation**: Forward model simulation for generating training data.
*   **Evaluation**: Multi-tiered evaluation (Fast, Slow, Final) to assess model accuracy across the parameter space.

## Directory Structure

The project is organized as follows:

```text
.
├── configs/                 # Configuration files (Hydra/OmegaConf style)
│   ├── data.yaml            # Data generation and loading configs
│   ├── model.yaml           # Model architecture configs
│   └── training.yaml        # Training hyperparameters
├── data/                    # Data handling
│   ├── loaders/             # PyTorch DataLoaders (e.g., simulation.py)
│   └── transforms/          # Data transformations and augmentations
├── notebooks/               # Jupyter notebooks for EDA and prototyping
├── outputs/                 # Experiment logs, checkpoints, and evaluation results
├── scripts/                 # Executable scripts
│   ├── train.py             # Main training script
│   ├── evaluate.py          # Evaluation pipeline script
│   └── invert.py            # Inference/Inversion script
├── src/                     # Source code package
│   ├── anomaly/             # Anomaly detection components
│   ├── evaluation/          # evaluation logic (e.g. evaluator.py)
│   ├── inversion/           # Inversion-specific logic (e.g., forward_model.py)
│   ├── layers/              # Custom neural network layers
│   ├── models/              # Model architectures (ResNet, Hybrid, etc.)
│   └── training/            # Training components (e.g., loss.py)
├── tests/                   # Unit and integration tests
├── .vscode/                 # VS Code configuration (extensions, settings)
├── pyproject.toml           # Project dependencies and build configuration
└── uv.lock                  # Dependency lock file
```

## Installation

This project uses `uv` for dependency management.

1.  **Install uv** (if not already installed):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Sync Dependencies**:
    ```bash
    uv sync
    ```

3.  **Activate Virtual Environment**:
    ```bash
    source .venv/bin/activate
    ```

## Usage

### Training

To train the model using the default configuration:

```bash
python scripts/train.py
```

You can override configuration parameters via command line (assuming Hydra/OmegaConf usage):

```bash
python scripts/train.py training.batch_size=32 model.name=resnet50
```

### Evaluation

To evaluate a trained model:

```bash
python scripts/evaluate.py --checkpoint paths/to/checkpoint.pth
```

### Inversion

To run inversion on specific inputs:

```bash
python scripts/invert.py
```

## Configuration

Configurations are located in the `configs/` directory.

*   `data.yaml`: Controls dataset parameters (wavelengths, focal lengths, grid sizes).
*   `model.yaml`: Defines model architecture, hidden dimensions, and specific layers.
*   `training.yaml`: Sets learning rate, epochs, optimizer settings, and loss functions.

## Contributing

**Policy**: If you add new files or change the directory structure, you **MUST** update this `README.md` to reflect the changes. Keep the directory tree and description up to date.
