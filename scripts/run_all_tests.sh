#!/bin/bash
# Test Runner Script
# Runs all tests in the suite and validates configuration.

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# ==============================================================================
# 0. Environment Check & Auto-Activation
# ==============================================================================

check_torch() {
    python -c "import torch" 2>/dev/null
}

if ! check_torch; then
    echo -e "${RED}[INFO] 'torch' not found in current python environment.${NC}"
    echo "Attempting to activate virtual environment..."
    
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    elif [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    fi
    
    # Re-check
    if ! check_torch; then
        echo -e "${RED}[ERROR] Virtual Environment missing or 'torch' not installed.${NC}"
        echo -e "Please run: ${GREEN}source .venv/bin/activate${NC} (or create venv with uv/pip)"
        echo -e "Current Python: $(which python)"
        exit 1
    else
        echo -e "${GREEN}[OK] Virtual environment activated.${NC}"
    fi
fi


echo -e "${GREEN}=== Starting Validation & Test Suite ===${NC}"

# 1. Config Validation Check
# Find a config to validate.
if [ -z "${1:-}" ]; then
    # STRATEGY: 
    # 1. Look for known valid configs in noise_matrix/
    CONFIG_TO_TEST=$(find configs/experiments/noise_matrix -name "*.yaml" 2>/dev/null | head -n 1)
    
    # 2. If not found, look for any valid experiment file in configs/experiments/
    #    that is NOT a partial config.
    if [ -z "$CONFIG_TO_TEST" ]; then
        CONFIG_TO_TEST=$(find configs/experiments -name "*.yaml" 2>/dev/null | head -n 1)
    fi
    
    # 3. If STILL nothing, skip validation rather than picking a partial config like 'training_resnet18.yaml'
    #    The grep -v approach was failing because you likely have files like training_resnet18.yaml that matched.
    if [ -z "$CONFIG_TO_TEST" ]; then
        echo -e "${RED}[WARN] No valid experiment config found in configs/experiments/ to test.${NC}"
        echo "Skipping static validation step."
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
    pytest tests/ -v
fi

echo -e "${GREEN}=== All Tests Passed ===${NC}"
exit 0
