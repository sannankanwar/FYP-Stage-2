#!/bin/bash
set -e

# Setup Output Directories
mkdir -p outputs/noise_matrix
mkdir -p outputs/optimizer

# 1. Run Experiment 9 (Noise Matrix) - 150 Epochs
echo "--- Starting Experiment 9 V2 (150 Epochs) ---"
./.venv/bin/python3 src/main.py --config configs/experiments/noise_matrix/exp_noisy_09_v2_150ep.yaml

# Verify Checkpoint Exists
EXP9_CKPT="outputs/noise_matrix/exp_noisy_09_v2_150ep/checkpoints/best_model.pth"
if [ ! -f "$EXP9_CKPT" ]; then
    echo "CRITICAL ERROR: Experiment 9 Checkpoint not found at $EXP9_CKPT"
    exit 1
fi

# 2. Run Note: Refiner Training Scripts use the NEW configs and point to the NEW checkpoint
echo "--- Starting Refiner Opt 2: GradFlow (200 Epochs) ---"
./.venv/bin/python3 scripts/train_refiner.py \
    --config configs/experiments/refinement/exp_opt_02_gradflow_v2.yaml \
    --baseline-config configs/experiments/noise_matrix/exp_noisy_09_v2_150ep.yaml \
    --baseline-checkpoint "$EXP9_CKPT"

echo "--- Starting Refiner Opt 3: PINN (300 Epochs) ---"
./.venv/bin/python3 scripts/train_refiner.py \
    --config configs/experiments/refinement/exp_opt_03_pinn_v2.yaml \
    --baseline-config configs/experiments/noise_matrix/exp_noisy_09_v2_150ep.yaml \
    --baseline-checkpoint "$EXP9_CKPT"

echo "=== All Requested Experiments Completed Successfully ==="
