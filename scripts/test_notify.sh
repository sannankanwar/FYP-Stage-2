#!/bin/bash

# Test script for Telegram Notifications + Nohup compatibility
# Test script for Telegram Notifications + Nohup compatibility
# Usage: nohup ./scripts/test_notify.sh &

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

echo "Starting Test Notification Run via Nohup..."
echo "PID: $$"

notify "Test: Start" "Running test_notify.sh via $(ps -p $$ -o comm=)\nPlease verify this message appeared."

echo "Simulating some work..."
sleep 2

echo "Simulating a log file..."
LOG_FILE="test_log_tail.txt"
echo "Line 1" > $LOG_FILE
echo "Line 2" >> $LOG_FILE
echo "Error: Something exploded nicely." >> $LOG_FILE

notify "Test: Log Attachment" "Testing log tail attachment." --log-file "$LOG_FILE"

rm $LOG_FILE

echo "Work done."
notify "Test: Complete" "Test run finished. Nohup compatibility verified if you see this."
