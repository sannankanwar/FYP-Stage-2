#!/usr/bin/env python3
"""
Strict Config Validator
-----------------------
Validates experiment configurations against rules defined in the Spec.
"""

import sys
import yaml
import os
import argparse

def load_yaml(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def validate_config(config_path):
    print(f"Validating {config_path}...")
    try:
        cfg = load_yaml(config_path)
    except Exception as e:
        print(f"FAIL: Invalid YAML: {e}")
        return False

    errors = []

    # 1. Top-level sections
    required_sections = ["meta", "data", "model", "training", "loss", "noise"]
    for section in required_sections:
        if section not in cfg:
            errors.append(f"Missing top-level section: '{section}'")

    if errors:
        for e in errors: print(f"  - {e}")
        return False

    # 2. Logic Checks
    
    # Model Output Dim vs Predicted Params
    out_dim = cfg['model'].get('output_dim')
    pred_params = cfg['loss'].get('predicted_params', [])
    if out_dim is not None and len(pred_params) > 0:
        if out_dim != len(pred_params):
            errors.append(f"Mismatch: model.output_dim ({out_dim}) != len(loss.predicted_params) ({len(pred_params)})")

    # Physics Scheduling
    phys_start = cfg['loss'].get('physics_start_epoch', 100)
    epochs = cfg['meta'].get('epochs', 100)
    if phys_start > epochs:
        print(f"  [Warning] physics_start_epoch ({phys_start}) > total epochs ({epochs}). Physics loss will never activate.")

    # Noise Pipeline Order Validity
    pipeline_order = cfg['noise'].get('pipeline_order', [])
    for component in pipeline_order:
        if component not in cfg['noise']:
            errors.append(f"Noise component '{component}' in pipeline_order but not defined in noise section")

    # Parameter Ranges
    data_ranges = [k for k in cfg['data'].keys() if k.endswith("_range")]
    for p in pred_params:
        # Check if corresponding range exists
        # Mapping: xc->xc_range, yc->yc_range, S->S_range, f->focal_length_range, lambda->wavelength_range
        # This mapping logic must match extract_param_ranges in loss.py
        range_key_map = {
            "xc": "xc_range", "yc": "yc_range", "S": "S_range", 
            "f": "focal_length_range", "lambda": "wavelength_range"
        }
        needed_key = range_key_map.get(p)
        if needed_key and needed_key not in cfg['data']:
             errors.append(f"Missing range for predicted param '{p}': expected key '{needed_key}' in data config")

    if errors:
        print("FAIL: Validation Errors:")
        for e in errors:
            print(f"  - {e}")
        return False
    
    print("PASS: Config looks valid.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to experiment config yaml")
    args = parser.parse_args()
    
    success = validate_config(args.config_path)
    sys.exit(0 if success else 1)
