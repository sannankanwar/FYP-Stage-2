import yaml
import glob
import sys
import os

def check_invariants():
    config_dir = "configs/experiments_5_loss_study"
    configs = glob.glob(os.path.join(config_dir, "*.yaml"))
    
    if not configs:
        print(f"No configs found in {config_dir}")
        sys.exit(1)
        
    print(f"Checking invariants across {len(configs)} configs...")
    
    loaded_configs = []
    for c in configs:
        with open(c, 'r') as f:
            loaded_configs.append((c, yaml.safe_load(f)))
            
    # Keys that MUST be identical
    invariant_keys = [
        "xc_range", "yc_range", "S_range", "wavelength_range", "focal_length_range",
        "num_samples", "batch_size",
        "name", "input_channels", "output_dim", "fno_norm", "activation", "standardize_outputs",
        "optimizer", "learning_rate", "epochs", "scheduler"
    ]
    
    base_config_path, base_data = loaded_configs[0]
    
    errors = []
    
    for key in invariant_keys:
        expected = base_data.get(key)
        
        for path, data in loaded_configs[1:]:
            current = data.get(key)
            if current != expected:
                errors.append(f"DRIFT DETECTED: Key '{key}' differs.\n  {base_config_path}: {expected}\n  {path}: {current}")
                
    if errors:
        print("\n".join(errors))
        print("\n[FAIL] Experiment Config Invariants Violated.")
        print("Experiments 5.x must differ ONLY in loss settings.")
        sys.exit(1)
    
    print("[PASS] All invariants checks passed.")

if __name__ == "__main__":
    check_invariants()
