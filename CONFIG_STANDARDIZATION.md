# Configuration System Standardization

This document defines the canonical configuration schema for all experiment YAMLs in this project. It covers data, model, training, loss, and noise sections.

---

## Table of Contents

1. [Current Configuration Patterns (As-Is)](#1-current-configuration-patterns-as-is)
2. [Implicit Conventions and Inferred Behavior](#2-implicit-conventions-and-inferred-behavior)
3. [Pain Points and Risks](#3-pain-points-and-risks)
4. [Proposed Standard Configuration Schema](#4-proposed-standard-configuration-schema)
5. [Experiment Configuration Rules](#5-experiment-configuration-rules)
6. [Noise Pipeline Configuration](#6-noise-pipeline-configuration)
7. [Complete Example Configs](#7-complete-example-configs)
8. [Migration Strategy](#8-migration-strategy)
9. [Validation and Checklists](#9-validation-and-checklists)

---

## 1. Current Configuration Patterns (As-Is)

Currently, the configuration system is **hierarchical but fragmented**, relying heavily on **filename conventions** to convey experiment intent rather than structural clarity.

*   **Global Defaults**: The root `configs/` directory contains base configurations:
    *   `data.yaml`: Defines the default simulation parameters (ranges, generic image size).
    *   `model.yaml`: A generic model definition (likely a placeholder or simple default).
    *   `training.yaml`: Default hyperparameters (lr=0.001, adam, etc.).
    *   **Variants**: There are explicit variant files like `model_resnet18.yaml` and `training_resnet18.yaml`, suggesting that users often copy-paste or mix-and-match specific base files manually rather than overriding keys.
*   **Experiment Overrides**: The `configs/experiments*` folders contain the bulk of the "business logic".
    *   **Phase-Based Grouping**: Experiments are grouped by "Phases" (`experiments`, `experiments_2`, etc.) which likely correspond to chronological project stages or specific research questions.
    *   **YAML Inheritance**: Experiment YAMLs are composed *on top* of the base YAMLs using a library like Hydra or a custom recursive merge.

---

## 2. Implicit Conventions and Inferred Behavior

The repository heavily uses **implicit mapping** between filenames and configuration intent.

*   **Filename Encoding**:
    *   `exp2_D03_resnet18_silu.yaml`: Encodes **Phase** (2), **Group** (D/Activations), **ID** (03), **Architecture** (ResNet18), and **Hyperparameter** (SiLU).
    *   `exp3_C04_fov10_wlfl20.yaml`: Encodes precise physics parameters.
*   **Physics Flags**:
    *   Files like `exp2_C01_resnet18_no_physics.yaml` vs `...physics_01.yaml` suggest a boolean flag or a weight.
*   **Optimizer/Scheduler**:
    *   `exp2_B05_resnet18_adamw_plateau.yaml` implies the config has `optimizer: adamw` and `scheduler: plateau`.
*   **Drift Risk**: There is no guarantee that `exp..._adamw.yaml` actually contains `optimizer: adamw`. The filename is a label, the content is the truth.

---

## 3. Pain Points and Risks

*   **Semantic Coupling**: The filename carries configuration data. If you rename the file but forget to update the YAML (or vice versa), the experiment is misleading.
*   **Config Duplication**: Multiple base files share 90% of their content.
*   **Ambiguous Overrides**: Without a clear `defaults` list, it's impossible to know the baseline just by looking at the file.
*   **Deeply Nested Variants**: Micro-managing numeric values in filenames scales poorly.
*   **Validation Difficulty**: No schema validation. Typos cause silent failures.

---

## 4. Proposed Standard Configuration Schema

We propose a **StrictMode** schema. Every config MUST define these top-level sections:

```yaml
meta:       # Experiment metadata
data:       # Input pipeline
model:      # Architecture
training:   # Optimization loop
loss:       # Loss function configuration
noise:      # Noise pipeline configuration
```

### 4.1 `meta` Section

```yaml
meta:
  name: experiment_name
  description: "Brief description of experiment purpose"
  version: "1.0"
  tags: ["ablation", "noise", "physics"]
  base_config: "configs/defaults.yaml"  # Optional inheritance trace
```

### 4.2 `data` Section

```yaml
data:
  mode: simulation  # or 'real'
  loader:
    batch_size: 32
    num_workers: 4
  simulation:                 # Only if mode == simulation
    image_size: [1024, 1024]
    apply_noise: true         # Links to noise.enabled
    params: [xc, yc, S, focal_length, wavelength]
    ranges:
      xc: [-100, 100]
      yc: [-100, 100]
      S: [5, 50]
      focal_length: [10, 100]
      wavelength: [0.4, 0.8]
  real:                       # Only if mode == real
    csv_path: "real_data/Metalens001.csv"
```

### 4.3 `model` Section

```yaml
model:
  architecture: unet          # [unet, resnet, fno, swin]
  variant: unet_standard
  in_channels: 2              # cos(φ), sin(φ)
  out_channels: 5             # Regression targets
  encoder:
    activation: silu          # [relu, gelu, silu, tanh]
  head:
    hidden_dim: 128
  physics:
    forward_model_enabled: false
```

### 4.4 `training` Section

```yaml
training:
  epochs: 100
  seed: 42
  optimizer:
    name: adamw               # [adam, adamw, sgd]
    lr: 1.0e-4
    weight_decay: 1.0e-5
  scheduler:
    name: plateau             # [plateau, cosine, step]
    patience: 10
    factor: 0.5
```

### 4.5 `loss` Section

```yaml
loss:
  type: mse                   # [mse, l1, huber, coordinate]
  weights:
    regression: 1.0
    physics: 0.0              # Set >0 to enable PINN mode
    spectral: 0.0             # Frequency domain loss
    robustness: 0.0
```

---

## 5. Experiment Configuration Rules

1.  **Unique Experiment ID**: Every experiment file must have a unique ID in the filename (e.g., `exp_001_...`).
2.  **No "Base" Variants**: Use a SINGLE `defaults.yaml`. Experiments must explicitly override values.
3.  **Explicit Intent Keys**: Add a `meta` section to every experiment config.
4.  **Filename Decoupling**: The filename is for humans; the YAML is for the machine.
    *   *Bad*: relying on filename `..._lr001.yaml` to set learning rate.
    *   *Good*: `..._lr_sweep.yaml` where the content explicitly has `training.optimizer.lr: 0.001`.

---

## 6. Noise Pipeline Configuration

The `noise` section controls the synthetic noise augmentation pipeline. It mirrors the implementation in `src/noise/noise_pipeline.py`.

### 6.1 Canonical Structure

```yaml
noise:
  enabled: <bool>           # Master switch for the entire pipeline
  mode: <str>               # 'training' | 'inference' | 'demo'
  seed: <int | null>        # Master seed (null = stochastic)
  pipeline_order: <list>    # Ordered list of component names
  <component_name>:         # One block per component
    enabled: <bool>
    <param>: <value>
```

### 6.2 Key Semantics

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `noise.enabled` | bool | **Yes** | Master switch. If `false`, entire pipeline is skipped. |
| `noise.mode` | str | **Yes** | Context hint: `training`, `inference`, or `demo`. |
| `noise.seed` | int \| null | **Yes** | Set for determinism, `null` for stochastic augmentation. |
| `noise.pipeline_order` | list[str] | **Yes** | Execution order. Must end with `wrap_phase`. |
| `noise.<component>.enabled` | bool | **Yes** | Per-component toggle. |

### 6.3 Component Parameters

All parameters must be **explicitly stated**. No implicit fallbacks.

#### `coordinate_warp`

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `enabled` | bool | false | |
| `seed` | int \| null | null | Inherits from `noise.seed` if null |
| `displacement_std_px` | float | 1.0 | Magnitude of spatial distortion |
| `correlation_length_px` | float | 5.0 | Smoothness of distortion field |
| `interpolation` | str | "bilinear" | Options: nearest, bilinear, bicubic |
| `padding_mode` | str | "reflection" | Options: zeros, reflection, border |
| `anisotropic` | bool | false | |
| `anisotropy_ratio` | float | 1.0 | Only used if anisotropic=true |

#### `fabrication_grf`

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `enabled` | bool | false | |
| `seed` | int \| null | null | |
| `amplitude_rad` | float | 0.15 | Phase noise amplitude in radians |
| `correlation_length_px` | float | 15.0 | Spatial smoothness |
| `mean` | float | 0.0 | Bias offset |
| `clip_std` | float | 3.0 | Clipping threshold in std units |

#### `structured_sinusoid`

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `enabled` | bool | false | |
| `seed` | int \| null | null | |
| `amplitude_rad` | float | 0.1 | |
| `spatial_frequency_px` | float | 50.0 | Period in pixels |
| `orientation_deg` | float | 45.0 | Angle of wave propagation |
| `phase_offset_rad` | str \| float | "random" | Use "random" or a fixed float |
| `fixed_phase_offset_rad` | float | 0.0 | Used only if phase_offset_rad is not "random" |

#### `sensor_grain`

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `enabled` | bool | false | |
| `seed` | int \| null | null | |
| `noise_type` | str | "gaussian" | Options: gaussian, poisson |
| `std_rad` | float | 0.2 | Standard deviation in radians |
| `spatially_correlated` | bool | false | |
| `correlation_length_px` | float | 2.0 | Only if spatially_correlated=true |

#### `dead_pixels`

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `enabled` | bool | false | |
| `seed` | int \| null | null | |
| `density` | float | 4e-4 | Fraction of total pixels |
| `region_type` | str | "pixels" | Options: pixels, blobs |
| `blob_radius_px` | list[float] | [1.0, 3.0] | [min, max] radius for blobs |
| `phase_value_mode` | str | "random" | Options: random, fixed |
| `fixed_phase_rad` | float | 0.0 | Used only if phase_value_mode="fixed" |
| `distribution` | str | "uniform" | Options: uniform, clustered |

#### `wrap_phase`

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `enabled` | bool | **true** | Must always be enabled for wrapped-phase data |

### 6.4 Integration with Data Section

Add one boolean flag in `data.simulation` to control noise at data-loading level:

```yaml
data:
  simulation:
    apply_noise: true  # Training code checks: data.simulation.apply_noise AND noise.enabled
```

---

## 7. Complete Example Configs

### 7.1 Baseline (No Noise)

```yaml
meta:
  name: baseline_no_noise
  description: "Clean simulation data, no augmentation"
  version: "1.0"

data:
  mode: simulation
  simulation:
    image_size: [1024, 1024]
    apply_noise: false

model:
  architecture: unet
  in_channels: 2
  out_channels: 5

training:
  epochs: 100
  batch_size: 8
  optimizer:
    name: adamw
    lr: 1e-4

loss:
  type: mse
  weights:
    regression: 1.0

noise:
  enabled: false
  mode: training
  seed: null
  pipeline_order:
    - coordinate_warp
    - fabrication_grf
    - structured_sinusoid
    - sensor_grain
    - dead_pixels
    - wrap_phase

  coordinate_warp:
    enabled: false
    seed: null
    displacement_std_px: 1.0
    correlation_length_px: 5.0
    interpolation: bilinear
    padding_mode: reflection
    anisotropic: false
    anisotropy_ratio: 1.0

  fabrication_grf:
    enabled: false
    seed: null
    amplitude_rad: 0.15
    correlation_length_px: 15.0
    mean: 0.0
    clip_std: 3.0

  structured_sinusoid:
    enabled: false
    seed: null
    amplitude_rad: 0.1
    spatial_frequency_px: 50.0
    orientation_deg: 45.0
    phase_offset_rad: random
    fixed_phase_offset_rad: 0.0

  sensor_grain:
    enabled: false
    seed: null
    noise_type: gaussian
    std_rad: 0.2
    spatially_correlated: false
    correlation_length_px: 2.0

  dead_pixels:
    enabled: false
    seed: null
    density: 4e-4
    region_type: pixels
    blob_radius_px: [1.0, 3.0]
    phase_value_mode: random
    fixed_phase_rad: 0.0
    distribution: uniform

  wrap_phase:
    enabled: true
```

### 7.2 Single-Noise Ablation (Sensor Grain Only)

```yaml
meta:
  name: ablation_sensor_grain
  description: "Isolate sensor grain noise effect"
  version: "1.0"

data:
  mode: simulation
  simulation:
    image_size: [1024, 1024]
    apply_noise: true

model:
  architecture: unet
  in_channels: 2
  out_channels: 5

training:
  epochs: 100
  batch_size: 8
  optimizer:
    name: adamw
    lr: 1e-4

loss:
  type: mse
  weights:
    regression: 1.0

noise:
  enabled: true
  mode: training
  seed: null
  pipeline_order:
    - coordinate_warp
    - fabrication_grf
    - structured_sinusoid
    - sensor_grain
    - dead_pixels
    - wrap_phase

  coordinate_warp:
    enabled: false
    seed: null
    displacement_std_px: 1.0
    correlation_length_px: 5.0
    interpolation: bilinear
    padding_mode: reflection
    anisotropic: false
    anisotropy_ratio: 1.0

  fabrication_grf:
    enabled: false
    seed: null
    amplitude_rad: 0.15
    correlation_length_px: 15.0
    mean: 0.0
    clip_std: 3.0

  structured_sinusoid:
    enabled: false
    seed: null
    amplitude_rad: 0.1
    spatial_frequency_px: 50.0
    orientation_deg: 45.0
    phase_offset_rad: random
    fixed_phase_offset_rad: 0.0

  sensor_grain:
    enabled: true  # <-- ONLY THIS IS ENABLED
    seed: null
    noise_type: gaussian
    std_rad: 0.2
    spatially_correlated: false
    correlation_length_px: 2.0

  dead_pixels:
    enabled: false
    seed: null
    density: 4e-4
    region_type: pixels
    blob_radius_px: [1.0, 3.0]
    phase_value_mode: random
    fixed_phase_rad: 0.0
    distribution: uniform

  wrap_phase:
    enabled: true
```

### 7.3 Composite Noise (Full Realism)

```yaml
meta:
  name: composite_full_noise
  description: "All noise components enabled for realistic augmentation"
  version: "1.0"

data:
  mode: simulation
  simulation:
    image_size: [1024, 1024]
    apply_noise: true

model:
  architecture: unet
  in_channels: 2
  out_channels: 5

training:
  epochs: 100
  batch_size: 8
  optimizer:
    name: adamw
    lr: 1e-4

loss:
  type: mse
  weights:
    regression: 1.0

noise:
  enabled: true
  mode: training
  seed: null
  pipeline_order:
    - coordinate_warp
    - fabrication_grf
    - structured_sinusoid
    - sensor_grain
    - dead_pixels
    - wrap_phase

  coordinate_warp:
    enabled: true
    seed: null
    displacement_std_px: 1.0
    correlation_length_px: 5.0
    interpolation: bilinear
    padding_mode: reflection
    anisotropic: false
    anisotropy_ratio: 1.0

  fabrication_grf:
    enabled: true
    seed: null
    amplitude_rad: 0.15
    correlation_length_px: 15.0
    mean: 0.0
    clip_std: 3.0

  structured_sinusoid:
    enabled: true
    seed: null
    amplitude_rad: 0.1
    spatial_frequency_px: 50.0
    orientation_deg: 45.0
    phase_offset_rad: random
    fixed_phase_offset_rad: 0.0

  sensor_grain:
    enabled: true
    seed: null
    noise_type: gaussian
    std_rad: 0.2
    spatially_correlated: false
    correlation_length_px: 2.0

  dead_pixels:
    enabled: true
    seed: null
    density: 4e-4
    region_type: pixels
    blob_radius_px: [1.0, 3.0]
    phase_value_mode: random
    fixed_phase_rad: 0.0
    distribution: uniform

  wrap_phase:
    enabled: true
```

---

## 8. Migration Strategy

1.  **Inventory**: Run a script to parse all `configs/experiments/**/*.yaml` and flatten their effective configuration.
2.  **Consolidate**: Identify the "True Defaults" (the most common value for every parameter). Write this to `configs/defaults_v2.yaml`.
3.  **Refactor**: For each Phase folder:
    *   Read the old file.
    *   Map keys to the New Schema.
    *   Write new file to `configs_v2/experiments/...`.
4.  **Lint**: Create a `scripts/lint_configs.py` that validates schema compliance.

---

## 9. Validation and Checklists

### StrictMode Compliance Checklist

Before committing any experiment YAML:

- [ ] Top-level keys present: `meta`, `data`, `model`, `training`, `loss`, `noise`
- [ ] `noise.enabled` is explicitly `true` or `false`
- [ ] `noise.pipeline_order` is a complete list ending with `wrap_phase`
- [ ] Every noise component has `enabled` explicitly set
- [ ] All numeric parameters are explicit (no missing keys)
- [ ] `data.simulation.apply_noise` matches `noise.enabled` intent
- [ ] Seed is set if reproducibility is required

### Noise Scenario Summary

| Scenario | `data.simulation.apply_noise` | `noise.enabled` | Components Enabled |
|----------|-------------------------------|-----------------|-------------------|
| Baseline | `false` | `false` | None |
| Ablation | `true` | `true` | 1 specific |
| Composite | `true` | `true` | All 5 + wrap |

### Bad Config Example (Do NOT do this)

```yaml
# VIOLATION: Implicit inheritance, no meta, mixing keys
lr: 0.001                 # Loose key, should be under training.optimizer
use_physics: true         # Ambiguous. model physics? loss physics?
resnet_depth: 18          # Schema violation. Should be model.variant
noise_level: 0.5          # Undefined key. Use noise.sensor_grain.std_rad
```

---

## Summary

This standard ensures:

1. **Reproducibility**: Explicit seeds and parameters everywhere.
2. **Modularity**: Toggle any component without code changes.
3. **Consistency**: All experiments follow the same schema.
4. **Compatibility**: Works with training pipelines and demo scripts.
5. **Clarity**: The YAML is the source of truth, not the filename.
