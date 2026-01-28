#!/bin/bash
# Test Runner Script
# Runs all tests in the suite and validates configuration.

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Starting Validation & Test Suite ===${NC}"

# 1. Config Validation Check
# Find a config to validate. Prioritize experiment configs.
if [ -z "${1:-}" ]; then
    # STRATEGY: Prioritize known-good subdirectories like 'noise_matrix'
    CONFIG_TO_TEST=$(find configs/experiments/noise_matrix -name "*.yaml" 2>/dev/null | head -n 1)
    
    # Fallback to general experiments if noise_matrix is empty
    if [ -z "$CONFIG_TO_TEST" ]; then
        CONFIG_TO_TEST=$(find configs/experiments -name "*.yaml" 2>/dev/null | head -n 1)
    fi
    
    # Fallback only to root configs that are NOT partials
    if [ -z "$CONFIG_TO_TEST" ]; then
        CONFIG_TO_TEST=$(find configs -name "*.yaml" | grep -v "training.yaml" | grep -v "model.yaml" | grep -v "data.yaml" | head -n 1)
    fi
    
    # If still nothing, warn and skip (or fail)
    if [ -z "$CONFIG_TO_TEST" ]; then
        echo -e "${RED}[WARN] No valid experiment config found to test. Skipping static validation.${NC}"
        CONFIG_TO_TEST=""
    fi
else
    CONFIG_TO_TEST="$1"
fi

if [ -n "$CONFIG_TO_TEST" ]; then
    echo "Running Static Config Validation on: $CONFIG_TO_TEST"
    python scripts/validate_config.py "$CONFIG_TO_TEST"
fi

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
