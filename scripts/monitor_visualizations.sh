#!/bin/bash
# Monitor outputs directory and run visualization on new experiments
# Usage: nohup ./scripts/monitor_visualizations.sh &

OUTPUT_BASE="outputs/noise_matrix"
SCRIPT_DIR=$(dirname "$0")

# Load secrets
if [ -f "$SCRIPT_DIR/../secrets.sh" ]; then
    source "$SCRIPT_DIR/../secrets.sh"
elif [ -f "secrets.sh" ]; then
    source "secrets.sh"
fi

echo "Monitoring $OUTPUT_BASE for completed experiments..."
echo "Press Ctrl+C to stop."

while true; do
    # Check if output dir exists
    if [ -d "$OUTPUT_BASE" ]; then
        for exp_dir in "$OUTPUT_BASE"/*; do
            if [ -d "$exp_dir" ]; then
                # Verify it has a checkpoint (meaning at least some training happened)
                if [ -f "$exp_dir/checkpoints/best_model.pth" ] || [ -f "$exp_dir/checkpoints/latest_checkpoint.pth" ]; then
                    
                    # Check if already visualized
                    if [ ! -d "$exp_dir/visualizations" ]; then
                        echo "New experiment found: $(basename "$exp_dir"). Visualizing..."
                        python scripts/visualize_and_notify.py --experiment_dir "$exp_dir"
                    fi
                fi
            fi
        done
    else
        echo "Waiting for $OUTPUT_BASE to be created..."
    fi
    
    echo "Sleeping 60s... (Active monitoring)"
    sleep 60
done
