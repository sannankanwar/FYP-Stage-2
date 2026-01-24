#!/bin/bash
# Run experiments in background with nohup
# Usage: ./scripts/run_experiments_nohup.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Create logs directory
mkdir -p "$PROJECT_DIR/logs"

# Get timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$PROJECT_DIR/logs/experiments_${TIMESTAMP}.log"

echo "Starting experiments in background..."
echo "Log file: $LOG_FILE"
echo ""
echo "To monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To check if running:"
echo "  ps aux | grep queue_experiments"
echo ""
echo "To stop:"
echo "  pkill -f queue_experiments_v3.sh"
echo ""

# Run with nohup, disown to fully detach from terminal
nohup bash "$SCRIPT_DIR/queue_experiments_v3.sh" > "$LOG_FILE" 2>&1 &

# Get PID
PID=$!
echo "$PID" > "$PROJECT_DIR/logs/experiment_pid.txt"

echo "Started with PID: $PID"
echo "PID saved to: $PROJECT_DIR/logs/experiment_pid.txt"
