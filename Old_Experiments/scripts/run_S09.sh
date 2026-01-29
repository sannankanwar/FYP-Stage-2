#!/bin/bash
# Queue script for S09
set -e
cd "$(dirname "$0")/.."

echo "=== Running expS09 (Baseline Restored) ==="
echo "Start: $(date)"
# Note: expS09 uses standardization, which is handled automatically by Trainer/Loss
uv run python scripts/train.py --config configs/experiments/expS09_baseline_restored.yaml
echo "Done: $(date)"
