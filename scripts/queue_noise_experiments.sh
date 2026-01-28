#!/bin/bash

# Run all noise matrix experiments sequentially
# Usage: ./scripts/queue_noise_experiments.sh

CONFIG_DIR="configs/experiments/noise_matrix"
LOG_DIR="logs/noise_experiments"
mkdir -p "$LOG_DIR"

# notification helper
SCRIPT_DIR=$(dirname "$0")

# Load secrets if present
if [ -f "$SCRIPT_DIR/../secrets.sh" ]; then
    source "$SCRIPT_DIR/../secrets.sh"
elif [ -f "secrets.sh" ]; then
    source "secrets.sh"
fi

notify() {
    python "$SCRIPT_DIR/notify.py" "$@"
}

echo "Starting Noise Experiment Matrix..."

# Count experiments
TOTAL_EXPS=$(ls "$CONFIG_DIR"/*.yaml | wc -l)
notify "Noise Matrix Started" "Queued $TOTAL_EXPS experiments.\nConfig Dir: $CONFIG_DIR"

COUNT=0

for config_file in "$CONFIG_DIR"/*.yaml; do
    filename=$(basename -- "$config_file")
    exp_id="${filename%.*}"
    COUNT=$((COUNT + 1))
    
    echo "----------------------------------------------------------------"
    echo "Running Experiment ($COUNT/$TOTAL_EXPS): $exp_id"
    echo "Config: $config_file"
    echo "----------------------------------------------------------------"
    
    LOG_FILE="$LOG_DIR/${exp_id}.log"
    
    notify "Exp Started ($COUNT/$TOTAL_EXPS)" "ID: $exp_id"

    # Run training
    # Python unbuffered for real-time log tailing if needed, though file output handles it
    python -u scripts/train.py --config "$config_file" > "$LOG_FILE" 2>&1
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "SUCCESS: $exp_id"
        notify "Exp Success ($COUNT/$TOTAL_EXPS)" "ID: $exp_id"
    else
        echo "FAILURE: $exp_id. Check logs at $LOG_FILE"
        notify "Exp Failure ($COUNT/$TOTAL_EXPS)" "ID: $exp_id\nExit Code: $EXIT_CODE" --log-file "$LOG_FILE"
    fi
    
done

echo "All experiments completed."
notify "Noise Matrix Completed" "Finished all $TOTAL_EXPS experiments."
