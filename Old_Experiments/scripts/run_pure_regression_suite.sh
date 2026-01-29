#!/bin/bash
# Run Pure Regression Suite (S10-S14)
set -e
cd "$(dirname "$0")/.."

echo "=== Starting Pure Regression Suite (S10-S14) ==="
echo "Date: $(date)"

# S10: Huber
echo "=== [1/5] Running expS10_robust_huber (Huber) ==="
uv run python scripts/train.py --config configs/experiments/expS10_robust_huber.yaml

# S11: LogCosh
echo "=== [2/5] Running expS11_robust_logcosh (LogCosh) ==="
uv run python scripts/train.py --config configs/experiments/expS11_robust_logcosh.yaml

# S12: MSLE (Raw)
echo "=== [3/5] Running expS12_logspace_msle (MSLE - Raw Inputs) ==="
uv run python scripts/train.py --config configs/experiments/expS12_logspace_msle.yaml

# S13: Wing
echo "=== [4/5] Running expS13_coord_wing (Wing) ==="
uv run python scripts/train.py --config configs/experiments/expS13_coord_wing.yaml

# S14: Biweight
echo "=== [5/5] Running expS14_robust_biweight (Tukey) ==="
uv run python scripts/train.py --config configs/experiments/expS14_robust_biweight.yaml

echo "=== Suite Completed Successfully ==="
echo "End: $(date)"
