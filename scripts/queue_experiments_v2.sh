#!/bin/bash

# Queue Experiments V2: 5-Parameter Inversion Matrix
# Runs experiments 1-16 sequentially.

echo "Starting Experiment Queue V2..."

# Array of experiment config files
configs=(
    "configs/experiments/exp01_resnet_baseline.yaml"
    "configs/experiments/exp02_spectral_baseline.yaml"
    "configs/experiments/exp03_resnet_weighted.yaml"
    "configs/experiments/exp04_spectral_weighted.yaml"
    "configs/experiments/exp05_spectral_physics.yaml"
    "configs/experiments/exp06_resnet_physics.yaml"
    "configs/experiments/exp07_spectral_extreme.yaml"
    "configs/experiments/exp08_spectral_phys_extreme.yaml"
    "configs/experiments/exp09_spectral_silu.yaml"
    "configs/experiments/exp10_spectral_tanh.yaml"
    "configs/experiments/exp11_spectral_gelu.yaml"
    "configs/experiments/exp12_spectral_muon.yaml"
    "configs/experiments/exp13_spectral_adamw.yaml"
    "configs/experiments/exp14_modern_spectral.yaml"
    "configs/experiments/exp15_modern_resnet.yaml"
    "configs/experiments/exp16_random_grid_ctrl.yaml"
)

for config in "${configs[@]}"; do
    echo "----------------------------------------------------------------"
    echo "Running Experiment: $config"
    echo "----------------------------------------------------------------"
    
    uv run scripts/train.py --config "$config"
    
    if [ $? -ne 0 ]; then
        echo "Error running $config. Stopping queue."
        exit 1
    fi
    
    echo "Finished $config"
    sleep 5
done

echo "All experiments completed successfully."
