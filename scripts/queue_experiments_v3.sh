#!/bin/bash
# Queue Experiments v3 - Run all experiments_2 configs
# Designed for RTX 5000 Pro GPU, ~24 hours total runtime

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_DIR/configs/experiments_2"
OUTPUT_DIR="$PROJECT_DIR/outputs_experiments_2"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log file
LOG_FILE="$OUTPUT_DIR/experiment_queue.log"
echo "=== Experiment Queue Started at $(date) ===" | tee -a "$LOG_FILE"

# Get all experiment configs in order
CONFIGS=($(ls "$CONFIG_DIR"/*.yaml | sort))

echo "Found ${#CONFIGS[@]} experiments to run" | tee -a "$LOG_FILE"

# Run each experiment
for i in "${!CONFIGS[@]}"; do
    CONFIG="${CONFIGS[$i]}"
    EXP_NAME=$(basename "$CONFIG" .yaml)
    
    echo "" | tee -a "$LOG_FILE"
    echo "=== [$((i+1))/${#CONFIGS[@]}] Starting: $EXP_NAME ===" | tee -a "$LOG_FILE"
    echo "Time: $(date)" | tee -a "$LOG_FILE"
    
    # Run training
    uv run python "$PROJECT_DIR/scripts/train.py" \
        --config "$CONFIG" \
        --output_dir "$OUTPUT_DIR/$EXP_NAME" \
        2>&1 | tee -a "$LOG_FILE"
    
    echo "Completed: $EXP_NAME at $(date)" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "=== All Experiments Completed at $(date) ===" | tee -a "$LOG_FILE"
