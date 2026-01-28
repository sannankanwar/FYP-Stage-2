
import os
import yaml

OUTPUT_DIR = "configs/experiments/noise_matrix"
os.makedirs(OUTPUT_DIR, exist_ok=True)

base_config = {
    "data": {
        "xc_range": [-100.0, 100.0],
        "yc_range": [-100.0, 100.0],
        "S_range": [5.0, 50.0],
        "wavelength_range": [0.4, 0.7],
        "focal_length_range": [10.0, 100.0],
        "resolution": 1024,
        "train_samples": 4000,
        "val_samples": 400,
        "simulation": {
            "image_size": [1024, 1024],
            "params": ["xc", "yc", "S", "f", "lambda"]
        }
    },
    "model": {
        "name": "fno_resnet18",
        "input_channels": 2,
        "output_dim": 5,
        "fno_norm": "instance",
        "fno_activation": "gelu"
    },
    "training": {
        "epochs": 50,
        "batch_size": 16,
        "optimizer": "adamw",
        "learning_rate": 1.0e-4,
        "scheduler": "plateau",
        "log_interval": 10,
        "output_dir": "outputs/noise_matrix"
    },
    "loss": {
        "predicted_params": ["xc", "yc", "S", "f", "lambda"],
        "physics_weight": 0.1,
        "physics_start_epoch": 5
    },
    "noise": {
        "seed": 42,
        "pipeline_order": ["sensor_grain", "dead_pixels", "wrap_phase"],
        "coordinate_warp": {"enabled": False},
        "fabrication_grf": {"enabled": False},
        "structured_sinusoid": {"enabled": False},
        "sensor_grain": {
            "enabled": True,
            "noise_type": "gaussian",
            "std_rad": 0.2,
            "spatially_correlated": False
        },
        "dead_pixels": {
            "enabled": True,
            "density": 4e-4,
            "region_type": "pixels",
            "phase_value_mode": "random"
        },
        "wrap_phase": {"enabled": True}
    }
}

experiments = [
    # Part 1: No Noise
    {
        "id": "exp_noisy_01_baseline_unitstd",
        "loss_mode": "unit_standardized",
        "physics": False,
        "noise_enabled": False,
        "desc": "Baseline: Unit Standardized Loss, No Noise"
    },
    {
        "id": "exp_noisy_02_baseline_gradflow",
        "loss_mode": "gradient_flow",
        "physics": False,
        "noise_enabled": False,
        "desc": "Baseline: Gradient Flow Loss (Param-Space), No Noise"
    },
    {
        "id": "exp_noisy_03_baseline_kendall",
        "loss_mode": "kendall",
        "physics": False,
        "noise_enabled": False,
        "desc": "Baseline: Kendall Uncertainty Loss, No Noise"
    },
    {
        "id": "exp_noisy_04_baseline_pinn",
        "loss_mode": "unit_standardized",
        "physics": True,
        "noise_enabled": False,
        "desc": "Baseline: Unit Standardized + Physics Loss, No Noise"
    },
    # Part 2: With Noise
    {
        "id": "exp_noisy_05_noise_unitstd",
        "loss_mode": "unit_standardized",
        "physics": False,
        "noise_enabled": True,
        "desc": "Experiment: Unit Standardized Loss, With Noise"
    },
    {
        "id": "exp_noisy_06_noise_gradflow",
        "loss_mode": "gradient_flow",
        "physics": False,
        "noise_enabled": True,
        "desc": "Experiment: Gradient Flow Loss (Param-Space), With Noise"
    },
    {
        "id": "exp_noisy_07_noise_kendall",
        "loss_mode": "kendall",
        "physics": False,
        "noise_enabled": True,
        "desc": "Experiment: Kendall Uncertainty Loss, With Noise"
    },
    {
        "id": "exp_noisy_08_noise_pinn",
        "loss_mode": "unit_standardized",
        "physics": True,
        "noise_enabled": True,
        "desc": "Experiment: Unit Standardized + Physics Loss, With Noise"
    }
]

for exp in experiments:
    cfg = base_config.copy()
    
    # Deep copy needed for nested dicts if we were modifying them, but we replace whole sections mostly
    # Actually safe to just dict(cfg) if we are careful. But lets be safer for 'data' and 'noise'
    import copy
    cfg = copy.deepcopy(base_config)
    
    # Meta
    cfg["meta"] = {
        "name": exp["id"],
        "description": exp["desc"],
        "version": "1.0"
    }
    
    # Training Name
    cfg["training"]["experiment_name"] = exp["id"]
    
    # Loss
    cfg["loss"]["mode"] = exp["loss_mode"]
    cfg["loss"]["physics_enabled"] = exp["physics"]
    if exp["loss_mode"] == "kendall":
        cfg["loss"]["init_log_var"] = 0.0
    
    # Noise
    cfg["noise"]["enabled"] = exp["noise_enabled"]
    cfg["data"]["simulation"]["apply_noise"] = exp["noise_enabled"]
    
    # Save
    filepath = os.path.join(OUTPUT_DIR, f"{exp['id']}.yaml")
    with open(filepath, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    
    print(f"Generated {filepath}")
