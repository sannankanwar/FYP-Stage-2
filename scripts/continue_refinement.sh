#!/bin/bash
# Script to continue Refiner training for 100 more epochs
# Assumes previous run finished 100 epochs, so we target 200.

# Resume from Experiment 2 (Refiner GradFlow)
# Uses best model or latest? Latest is better for continuity of optimizer state.
# But Best is safer for performance.
# Standard practice is to resume from latest if interrupted, or fine-tune from best.
# Here we want to "continue training", so "latest" makes sense to preserve optimizer momentum.

CHECKPOINT="outputs/experiment_refiner_gradflow/checkpoints/latest_checkpoint.pth"

# Verify Checkpoint Exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    echo "Please update the CHECKPOINT variable in this script."
    exit 1
fi

echo "Resuming Training from $CHECKPOINT..."
echo "Target Epochs: 200 (+100 from original 100)"

nohup python3 scripts/train_refiner.py \
    --config configs/experiments/exp_opt_02_gradflow.yaml \
    --baseline-config configs/experiments/exp_noisy_09_coordinate_loss.yaml \
    --baseline-checkpoint outputs/experiment_noisy_coordinate_loss/checkpoints/best_model.pth \
    --resume "$CHECKPOINT" \
    --epochs 200 \
    > outputs/refiner_continue.log 2>&1 &

echo "Training running in background. Logs: outputs/refiner_continue.log"
