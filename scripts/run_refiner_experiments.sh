#!/bin/bash
set -e

# Experiment 1: Gradient Flow Loss
echo "Starting Experiment 2: Refiner GradFlow..."
./.venv/bin/python3 scripts/train_refiner.py \
    --config configs/experiments/refinement/exp_opt_02_gradflow.yaml \
    --baseline-config configs/experiments/noise_matrix/exp_noisy_09_coordinate_loss.yaml \
    --baseline-checkpoint outputs/noise_matrix/exp_noisy_09_coordinate_loss/checkpoints/best_model.pth

# Experiment 2: PINN Loss
echo "Starting Experiment 3: Refiner PINN..."
./.venv/bin/python3 scripts/train_refiner.py \
    --config configs/experiments/refinement/exp_opt_03_pinn.yaml \
    --baseline-config configs/experiments/noise_matrix/exp_noisy_09_coordinate_loss.yaml \
    --baseline-checkpoint outputs/noise_matrix/exp_noisy_09_coordinate_loss/checkpoints/best_model.pth

echo "All Refiner Experiments Completed."
