#!/bin/bash
# Wrapper to queue RL Refinement training using the remote trainer infra
# Usage: ./scripts/train_refiner_remote.sh <config_path> [steps]

set -u

CONFIG=${1:-"Old_Experiments/configs/phase1/exp9_pinn_silu.yaml"}
STEPS=${2:-100000}

EXP_NAME="RL_Refinement_$(basename "$CONFIG" .yaml)"

echo "Queuing Refiner Training on Server..."
echo "Config: $CONFIG"
echo "Steps: $STEPS"

# Using 8 environments for faster training on server
CMD="python scripts/refine_rl.py --config $CONFIG --steps $STEPS --n_envs 8"

./scripts/train_remote.sh "$EXP_NAME:$CMD"
