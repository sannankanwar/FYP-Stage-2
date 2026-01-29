#!/bin/bash
# Run Advanced Research Directions (Refiner + Active Learning)

# IMPORTANT: Ensure Exp9 (CoordinateLoss) has finished running first!
# The Refiner (Dir A) depends on Exp9 checkpoint. 
# We assume it is at: outputs/noise_matrix/exp_noisy_09_coordinate_loss/checkpoints/best_model.pth

EXP9_CKPT="outputs/noise_matrix/exp_noisy_09_coordinate_loss/checkpoints/best_model.pth"

# Load Environment Variables from .env
if [ -f .env ]; then
    echo "Loading environment variables from .env..."
    set -a
    source .env
    set +a
else
    echo "WARNING: .env file not found. Ensure TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set."
fi

if [ ! -f "$EXP9_CKPT" ]; then
    echo "CRITICAL WARNING: Exp9 Baseline Checkpoint not found at $EXP9_CKPT"
    echo "Direction A (Refiner) will likely fail or use garbage weights."
    echo "Please ensure Exp9 is trained before running this script."
    # We continue anyway to allow Direction B to run
fi

echo "=== Starting Advanced Research Experiments ==="
date

# 1. Direction A: Learned Optimizer (Refiner)
# Depends on Exp9 Checkpoint
echo "--- Starting Direction A: Refiner Training ---"
python3 scripts/train_refiner.py \
    --config configs/experiments/refinement/exp_opt_01.yaml \
    --baseline-config configs/experiments/noise_matrix/exp_noisy_09_coordinate_loss.yaml \
    --baseline-checkpoint "$EXP9_CKPT"

# 2. Direction B: Active Learning (AdaBoost Sampler)
# Independent of Exp9 (Train from scratch with sampler)
echo "--- Starting Direction B: Active Learning ---"
python3 scripts/train_active.py \
    --config configs/experiments/active/exp_active_01.yaml

# 3. Optional: Heatmap Generation (Diagnostic)
# This runs using the newly trained active model
echo "--- Generating Diagnostic Heatmap for Direction B ---"
ACTIVE_CKPT="outputs/active_learning/exp_active_01_adaboost/checkpoints/best_model.pth"
if [ -f "$ACTIVE_CKPT" ]; then
    python3 scripts/heatmap_diagnostic.py \
        --config configs/experiments/active/exp_active_01.yaml \
        --checkpoint "$ACTIVE_CKPT" \
        --output-dir outputs/active_learning/exp_active_01_adaboost/analysis
fi

echo "=== All Advanced Experiments Completed ==="
date
