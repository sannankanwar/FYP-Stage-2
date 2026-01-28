#!/bin/bash
# Linux Training Launcher with Telegram Notifications
# Usage: ./scripts/train_remote.sh exp1:"cmd1" exp2:"cmd2" ...

set -u

SCRIPT_DIR=$(dirname "$0")
LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"

notify() {
    python "$SCRIPT_DIR/notify.py" "$@"
}

timestamp() {
    date "+%Y-%m-%d_%H-%M-%S"
}

# 1. Run Pre-flight Checks
echo "Running Pre-flight Tests..."
if ! bash "$SCRIPT_DIR/run_all_tests.sh"; then
    echo "Tests Failed! Aborting."
    notify "Training Aborted" "Pre-flight tests failed. Check console."
    exit 1
fi

notify "Training Started" "Pre-flight tests passed. Queued ${#} experiments."

# 2. Iterate through experiments
count=0
total=${#}

for arg in "$@"; do
    count=$((count + 1))
    
    # Split arg "name:cmd"
    EXP_NAME="${arg%%:*}"
    CMD="${arg#*:}"
    
    if [ "$EXP_NAME" == "$arg" ]; then
        echo "Error: Argument format must be name:command"
        exit 1
    fi

    LOG_FILE="$LOG_DIR/${EXP_NAME}_$(timestamp).log"
    
    notify "Experiment Starting ($count/$total)" "Name: $EXP_NAME\nCommand: $CMD"

    echo "----------------------------------------------------------------"
    echo "Starting $EXP_NAME"
    echo "Cmd: $CMD"
    echo "Log: $LOG_FILE"
    echo "----------------------------------------------------------------"

    # Execute Command
    set +e # Disable exit-on-error to capture failure
    eval "$CMD" > "$LOG_FILE" 2>&1
    EXIT_CODE=$?
    set -e

    if [ $EXIT_CODE -eq 0 ]; then
        notify "Experiment Success ($count/$total)" "$EXP_NAME completed successfully."
    else
        # Check for specific exit codes
        if [ $EXIT_CODE -eq 137 ] || [ $EXIT_CODE -eq 143 ]; then
             notify "CATASTROPHIC FAILURE: OOM/KILL ($count/$total)" "Experiment: $EXP_NAME\nExit Code: $EXIT_CODE (Likely Out of Memory)" --log-file "$LOG_FILE"
        else
             notify "Experiment Failed ($count/$total)" "Experiment: $EXP_NAME\nExit Code: $EXIT_CODE" --log-file "$LOG_FILE"
        fi
        
        # Optional: Decide whether to continue or stop. 
        # For now, we continue to next experiment unless user says otherwise.
        # But commonly in ML, if one fails, we might want to stop. 
        # Given "crash-proof", we will TRY to continue purely independent experiments.
    fi
done

notify "All Experiments Completed" "Finished queue of $total experiments."
