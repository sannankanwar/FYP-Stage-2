#!/bin/bash
# Script to continue Refiner training for 100 more epochs
# Assumes previous run finished 100 epochs, so we target 200.

# Resume from Experiment 2 (Refiner GradFlow)
# Correct path based on configs/experiments/refinement/exp_opt_02_gradflow.yaml
# output_dir is "outputs_2"

CHECKPOINT="outputs_2/exp_opt_02_gradflow/checkpoints/latest_checkpoint.pth"
BASELINE_CKPT="outputs/noise_matrix/exp_noisy_09_coordinate_loss/checkpoints/best_model.pth"

# Verify Checkpoint Exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Refiner Checkpoint not found at $CHECKPOINT"
    echo "This means the initial training hasn't completed or started yet."
    exit 1
fi

if [ ! -f "$BASELINE_CKPT" ]; then
    echo "Error: Baseline Checkpoint not found at $BASELINE_CKPT"
    echo "You must run Experiment 9 (Coordinate Loss) first."
    exit 1
fi

echo "Resuming Training from $CHECKPOINT..."
echo "Target Epochs: 200 (+100 from original 100)"

nohup python3 scripts/train_refiner.py \
    --config configs/experiments/refinement/exp_opt_02_gradflow.yaml \
    --baseline-config configs/experiments/noise_matrix/exp_noisy_09_coordinate_loss.yaml \
    --baseline-checkpoint "$BASELINE_CKPT" \
    --resume "$CHECKPOINT" \
    --epochs 200 \
    > outputs_2/refiner_continue.log 2>&1 &

echo "Training running in background. Logs: outputs_2/refiner_continue.log"
