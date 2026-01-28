
# Valid imports provided via PYTHONPATH=.
import pytest
import torch
import yaml
import os
from scripts.validate_config import validate_config

# Create a minimal valid config for testing
@pytest.fixture
def valid_config_path(tmp_path):
    config = {
        "meta": {"epochs": 10},
        "data": {
            "xc_range": [-10, 10], "yc_range": [-10, 10], 
            "S_range": [10, 20], "focal_length_range": [10, 20],
            "wavelength_range": [0.4, 0.7]
        },
        "model": {"output_dim": 5},
        "training": {"batch_size": 8},
        "loss": {
            "predicted_params": ["xc", "yc", "S", "f", "lambda"],
            "physics_start_epoch": 5
        },
        "noise": {
            "pipeline_order": ["wrap_phase"],
            "wrap_phase": {"enabled": True}
        }
    }
    path = tmp_path / "valid_config.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return str(path)

def test_validate_valid_config(valid_config_path):
    assert validate_config(valid_config_path) is True

def test_validate_missing_section(tmp_path):
    path = tmp_path / "bad.yaml"
    with open(path, "w") as f:
        yaml.dump({"meta": {}}, f)
    # Redirect stdout to avoid clutter
    assert validate_config(str(path)) is False

def test_validate_dim_mismatch(tmp_path):
    config = {
        "meta": {}, "data": {}, "model": {"output_dim": 4},
        "training": {}, 
        "loss": {"predicted_params": ["xc", "yc", "S", "f", "lambda"]}, # 5 params
        "noise": {}
    }
    path = tmp_path / "mismatch.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    assert validate_config(str(path)) is False
