#!/bin/bash

# Run PART 2 of noise matrix experiments (5-8)
# Usage: ./scripts/queue_noise_experiments_part2.sh

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

echo "Starting Noise Experiment Matrix (Part 2: Exps 5-8)..."

# Explicit list of configs for Part 2
CONFIGS=(
    "$CONFIG_DIR/exp_noisy_05_noise_unitstd.yaml"
    "$CONFIG_DIR/exp_noisy_06_noise_gradflow.yaml"
    "$CONFIG_DIR/exp_noisy_07_noise_kendall.yaml"
    "$CONFIG_DIR/exp_noisy_08_noise_pinn.yaml"
)

TOTAL_EXPS=${#CONFIGS[@]}
notify "Noise Part 2 Started" "Queued $TOTAL_EXPS experiments (5-8)."

COUNT=0

for config_file in "${CONFIGS[@]}"; do
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
        
        # Run Visualization
        echo "Generating Visualization for $exp_id..."
        EXP_DIR="outputs/noise_matrix/$exp_id"
        
        if [ -d "$EXP_DIR" ]; then
             python scripts/visualize_and_notify.py --experiment_dir "$EXP_DIR"
        else
             echo "Warning: Exp dir $EXP_DIR not found, skipping visualization."
        fi
        
    else
        echo "FAILURE: $exp_id. Check logs at $LOG_FILE"
        notify "Exp Failure ($COUNT/$TOTAL_EXPS)" "ID: $exp_id\nExit Code: $EXIT_CODE" --log-file "$LOG_FILE"
    fi
    
done

echo "Part 2 experiments completed."
notify "Noise Part 2 Completed" "Finished experiments 5-8."
