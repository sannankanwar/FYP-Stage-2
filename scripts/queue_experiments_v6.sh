#!/bin/bash
# Queue Experiments v6 - Phase B: Principled Loss Engineering
# Input: configs/experiments_5/
# Output: outputs_5/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_DIR/configs/experiments_5"
OUTPUT_DIR="$PROJECT_DIR/outputs_5"

mkdir -p "$OUTPUT_DIR"

LOG_FILE="$OUTPUT_DIR/experiment_queue.log"
echo "=== Phase B Experiments Started at $(date) ===" | tee -a "$LOG_FILE"

CONFIGS=($(ls "$CONFIG_DIR"/*.yaml | sort))
echo "Found ${#CONFIGS[@]} experiments to run (50 epochs each)" | tee -a "$LOG_FILE"

for i in "${!CONFIGS[@]}"; do
    CONFIG="${CONFIGS[$i]}"
    EXP_NAME=$(basename "$CONFIG" .yaml)
    
    echo "" | tee -a "$LOG_FILE"
    echo "=== [$((i+1))/${#CONFIGS[@]}] Starting: $EXP_NAME ===" | tee -a "$LOG_FILE"
    echo "Time: $(date)" | tee -a "$LOG_FILE"
    
    uv run python "$PROJECT_DIR/scripts/train.py" \
        --config "$CONFIG" \
        2>&1 | tee -a "$LOG_FILE"
    
    echo "Completed: $EXP_NAME at $(date)" | tee -a "$LOG_FILE"
    
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 5
done

echo "" | tee -a "$LOG_FILE"
echo "=== All Experiments Completed at $(date) ===" | tee -a "$LOG_FILE"

# Generate comparison plots (residuals)
echo "Generating residual phase maps..." | tee -a "$LOG_FILE"
uv run python "$PROJECT_DIR/scripts/plot_residuals.py" --output-dir "$OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE"
echo "=== Post-processing Complete ===" | tee -a "$LOG_FILE"
