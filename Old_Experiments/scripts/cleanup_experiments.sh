#!/usr/bin/env bash
set -e

# Root of your project (assumes you run this from project root)
ROOT_DIR="$(pwd)"

echo "Project root: $ROOT_DIR"
echo
echo "The following experiment-related paths will be removed:"
echo

TARGETS=(
  experiments
  outputs
  outputs_*
  experiments_*
  outputs_experiments_*
  logs
  exp*.log
  queue*.log
  output.log
)

# Dry run preview
for pattern in "${TARGETS[@]}"; do
  find "$ROOT_DIR" -maxdepth 1 -name "$pattern" -print
done

echo
read -p "Proceed with deletion? Type YES to confirm: " CONFIRM

if [[ "$CONFIRM" != "YES" ]]; then
  echo "Aborted. Nothing deleted."
  exit 0
fi

echo
echo "Deleting..."
for pattern in "${TARGETS[@]}"; do
  find "$ROOT_DIR" -maxdepth 1 -name "$pattern" -exec rm -rf {} +
done

echo "Cleanup complete."
