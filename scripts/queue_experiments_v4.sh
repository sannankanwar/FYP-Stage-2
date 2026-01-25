#!/bin/bash
# Queue Experiments v4 - Run all experiments_3 configs with auxiliary physics loss
# Output to outputs_3/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_DIR/configs/experiments_3"
OUTPUT_DIR="$PROJECT_DIR/outputs_3"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log file
LOG_FILE="$OUTPUT_DIR/experiment_queue.log"
echo "=== Experiment Queue V4 Started at $(date) ===" | tee -a "$LOG_FILE"

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
        2>&1 | tee -a "$LOG_FILE"
    
    echo "Completed: $EXP_NAME at $(date)" | tee -a "$LOG_FILE"
    
    # Clear GPU memory between experiments
    python3 -c "import torch; torch.cuda.empty_cache(); print('GPU memory cleared')" 2>/dev/null || true
    sleep 2
done

echo "" | tee -a "$LOG_FILE"
echo "=== All Experiments Completed at $(date) ===" | tee -a "$LOG_FILE"
