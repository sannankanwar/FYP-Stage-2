#!/bin/bash
# Queue script for S-parameter loss experiments
# Run with: nohup bash scripts/queue_experiments_S.sh > experiments/New_Experiments/queue.log 2>&1 &

set -e
cd "$(dirname "$0")/.."

echo "=== Starting S-Parameter Loss Experiments Queue ==="
echo "Start Time: $(date)"
echo ""

# Ensure output directory exists
mkdir -p experiments/New_Experiments

# Experiment 1: Raw Physics Loss
echo "=== [1/4] Running expS01_raw_physics ==="
echo "Start: $(date)"
uv run python scripts/train.py configs/experiments/expS01_raw_physics.yaml
echo "Done: $(date)"
echo ""

# Experiment 2: Adaptive Physics Loss (Kendall Uncertainty)
echo "=== [2/4] Running expS02_adaptive_physics ==="
echo "Start: $(date)"
uv run python scripts/train.py configs/experiments/expS02_adaptive_physics.yaml
echo "Done: $(date)"
echo ""

# Experiment 3: Weighted Physics Loss
echo "=== [3/4] Running expS03_weighted_physics ==="
echo "Start: $(date)"
uv run python scripts/train.py configs/experiments/expS03_weighted_physics.yaml
echo "Done: $(date)"
echo ""

# Experiment 4: Auxiliary Physics Loss (FFT Fringe)
echo "=== [4/4] Running expS04_auxiliary_physics ==="
echo "Start: $(date)"
uv run python scripts/train.py configs/experiments/expS04_auxiliary_physics.yaml
echo "Done: $(date)"
echo ""

echo "=== All Experiments Complete ==="
echo "End Time: $(date)"
