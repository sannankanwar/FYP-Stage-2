#!/bin/bash
# Queue script for S-Parameter Loss Experiments (Selected Set)
# Run with: nohup bash scripts/queue_experiments_S_v2.sh > experiments/New_Experiments/queue.log 2>&1 &

set -e
cd "$(dirname "$0")/.."

echo "=== Starting Selected S-Parameter Loss Experiments Queue ==="
echo "Start Time: $(date)"
echo ""

# Ensure output directory exists
mkdir -p experiments/New_Experiments

# Experiment S05: Inverse Variance
echo "=== [1/4] Running expS05_inverse_variance ==="
echo "Start: $(date)"
uv run python scripts/train.py --config configs/experiments/expS05_inverse_variance.yaml
echo "Done: $(date)"
echo ""

# Experiment S06: Entropy Weighting
echo "=== [2/4] Running expS06_entropy ==="
echo "Start: $(date)"
uv run python scripts/train.py --config configs/experiments/expS06_entropy.yaml
echo "Done: $(date)"
echo ""

# Experiment S07: Physics Sensitive Weighting
echo "=== [3/4] Running expS07_physics_sensitive ==="
echo "Start: $(date)"
uv run python scripts/train.py --config configs/experiments/expS07_physics_sensitive.yaml
echo "Done: $(date)"
echo ""

# Experiment S08: Kendall Adaptive Loss
echo "=== [4/4] Running expS08_kendall_adaptive ==="
echo "Start: $(date)"
uv run python scripts/train.py --config configs/experiments/expS08_kendall_adaptive.yaml
echo "Done: $(date)"
echo ""

echo "=== All Experiments Complete ==="
echo "End Time: $(date)"
