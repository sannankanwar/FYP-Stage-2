# Pure Regression Experiments (S10-S14) Documentation

## Experiment Overview

| ID | Loss Function | Standardization | Use Case | Epochs |
|---|---|---|---|---|
| **S10** | Huber | **YES** | Robust to outliers (general purpose) | 100 |
| **S11** | Log-Cosh | **YES** | Smooth robust loss (differentiable) | 100 |
| **S12** | MSLE | **NO** (Raw) | Multiplicative noise / Wide ranges | 100 |
| **S13** | Wing | **YES** | High precision coordinates | 100 |
| **S14** | Biweight | **YES** | Ignore extreme outliers | 100 |

## Standardization Logic
For experiments with **Standardization = YES**, the model outputs values in the range `[-1, 1]`. You must denormalize them to get physical units (μm).

**Formula:**
$$x_{physical} = x_{norm} \cdot \sigma + \mu$$

Where:
- $\mu = (max + min) / 2$
- $\sigma = (max - min) / 2$

**Parameter Ranges (defaults):**
- `xc, yc`: `[-0.5, 0.5]` (normalized relative to image size) OR `[-500, 500]` μm depending on dataset. Code uses physical `[-500, 500]`.
- `S`: `[1.0, 40.0]` μm
- `wavelength`: `[0.4, 0.7]` μm
- `focal_length`: `[10.0, 100.0]` μm

*(Note: Check `data.yaml` or train logs for specific range values used if changed).*

---

## Prompt for Inference Notebook

**Copy and paste the following prompt into a chat with an AI (Cursor/ChatGPT) to generate your inference notebook:**

> "I have a PyTorch model (FNO-ResNet18) trained for a physics inverse problem. I need a Jupyter Notebook to run inference on checkpoints from 5 different experiments (S10-S14).
>
> **Code Context:**
> - Model Class: `src.models.fno.fno_resnet18.FNOResNet18`
> - Normalizer Class: `src.utils.normalization.ParameterNormalizer`
> - Config Loading: Use `yaml` to load config files from `configs/experiments/`.
> - Checkpoints: Located in `outputs_2/expSXX.../checkpoints/best_model.pth`.
>
> **Requirements:**
> 1.  **Load Config & Model**: Write a function to load the model and config for a given experiment ID (e.g., 'expS10_robust_huber').
> 2.  **Handle Normalization**:
>     - Check `config['standardize_outputs']`.
>     - If True: Instantiate `ParameterNormalizer(ranges)` using the ranges defined in the config (S_range, wavelength_range, etc.).
>     - If True: `pred_physical = normalizer.denormalize_tensor(model_output)`.
>     - If False: `pred_physical = model_output`.
> 3.  **Inference Loop**: Load a few test samples (create dummy tensor `(B, 2, 1024, 1024)` or load real data) and run the model.
> 4.  **Visualization**: Display the predicted parameters vs true parameters (if available) or just print the physical predictions.
>
> **Important**: The model outputs 5 values: `[xc, yc, S, wavelength, focal_length]`. Ensure correct denormalization for each channel."
