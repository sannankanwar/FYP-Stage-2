#!/bin/bash
# Run the specific noise matrix experiments (GradFlow & CoordinateLoss)

echo "=== Starting Final Noise Experiments (100 Epochs, 2000 Samples) ==="
date

# Load Environment Variables from .env
if [ -f .env ]; then
    echo "Loading environment variables from .env..."
    set -a
    source .env
    set +a
fi

# 1. GradFlow (The SOTA Baseline)
echo "--- Starting Exp 06: GradFlow ---"
python3 scripts/train.py --config configs/experiments/noise_matrix/exp_noisy_06_noise_gradflow.yaml

# 2. CoordinateLoss (The Challenger)
echo "--- Starting Exp 09: CoordinateLoss ---"
python3 scripts/train.py --config configs/experiments/noise_matrix/exp_noisy_09_coordinate_loss.yaml

echo "=== All Experiments Completed ==="
date
