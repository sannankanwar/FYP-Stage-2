#!/bin/bash
# Test Runner Script
# Runs all tests in the suite and validates configuration.

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Starting Validation & Test Suite ===${NC}"

# 1. Config Validation Check (on defaults or a reference config)
# Find a config to validate. Prioritize experiment config if provided arg, else defaults.
if [ -z "$1" ]; then
    CONFIG_TO_TEST="configs/training.yaml"
    if [ ! -f "$CONFIG_TO_TEST" ]; then
        # Fallback to any yaml in configs
        CONFIG_TO_TEST=$(find configs -name "*.yaml" | head -n 1)
    fi
else
    CONFIG_TO_TEST="$1"
fi

echo "Running Static Config Validation on: $CONFIG_TO_TEST"
python scripts/validate_config.py "$CONFIG_TO_TEST"

# 2. Pytest Suite
echo -e "\n${GREEN}=== Running Unit Tests ===${NC}"
export PYTHONPATH=$PYTHONPATH:.

# Check for CUDA availability
HAS_CUDA=$(python -c "import torch; print(torch.cuda.is_available())")

if [ "$HAS_CUDA" == "True" ]; then
    echo "CUDA Detected. Running FULL test suite including GPU tests."
    pytest tests/ -v
else
    echo "NO CUDA Detected. Running CPU-only tests (skipping @pytest.mark.cuda)."
    # Rely on skipif decorators in the tests
    pytest tests/ -v
fi

echo -e "${GREEN}=== All Tests Passed ===${NC}"
exit 0
