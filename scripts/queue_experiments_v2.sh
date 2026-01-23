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
    # Extract experiment name from filename (e.g., exp01_resnet_baseline)
    exp_name=$(basename "$config" .yaml)
    exp_dir="outputs_2/$exp_name"
    log_dir="$exp_dir/logs"
    
    mkdir -p "$log_dir"
    
    echo "----------------------------------------------------------------"
    echo "Running Experiment: $exp_name"
    echo "Log: $log_dir/output.log"
    echo "----------------------------------------------------------------"
    
    # 1. Run Training (Redirect output to experiment-specific log)
    uv run scripts/train.py --config "$config" | tee "$log_dir/output.log"
    
    if [ $? -ne 0 ]; then
        echo "Error running $config. Skipping to next or stopping?"
        # exit 1 # Optional: stop on failure
    fi
    
    # 2. Post-Training Visualization (Automated)
    echo "Generating Reports for $exp_name..."
    
    # Loss Curves (Full: 0-100)
    uv run python scripts/plot_loss.py "$log_dir/output.log" \
        --output "$exp_dir/loss_plot_full.png" \
        --title "$exp_name: Full Training Process"
        
    # Loss Curves (Zoomed: 50-100)
    uv run python scripts/plot_loss.py "$log_dir/output.log" \
        --output "$exp_dir/loss_plot_zoomed.png" \
        --min-epoch 50 \
        --title "$exp_name: Convergence Convergence (Last 50 Epochs)"

    # Scatter Plots & Error Analysis
    uv run python scripts/evaluate.py --experiment_dir "$exp_dir"
    
    # Phase Reconstruction Visualization
    uv run python scripts/visualize_reconstruction.py --experiment_dir "$exp_dir"
    
    echo "Finished $exp_name. Artifacts saved to $exp_dir"
    echo ""
    sleep 5
done

echo "All experiments completed successfully."
