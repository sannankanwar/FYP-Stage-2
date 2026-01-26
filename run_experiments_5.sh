#!/bin/bash
# run_experiments_5.sh
# Runs the 4 experiments for Loss Variant Study

# Ensure conda/venv is active
# source .venv/bin/activate

EXPERIMENT_DIR="configs/experiments_5_loss_study"

echo "Starting Experiment Suite 5..."

# 1. Unit Standardized (Baseline)
echo "Running Exp 5.1: Unit Standardized..."
python src/main.py --config ${EXPERIMENT_DIR}/exp5_1_unitstd.yaml > ${EXPERIMENT_DIR}/exp5_1.log 2>&1

# 2. Gradient Consistency
echo "Running Exp 5.2: Gradient Consistency..."
python src/main.py --config ${EXPERIMENT_DIR}/exp5_2_gradflow.yaml > ${EXPERIMENT_DIR}/exp5_2.log 2>&1

# 3. Kendall Uncertainty
echo "Running Exp 5.3: Kendall Uncertainty..."
python src/main.py --config ${EXPERIMENT_DIR}/exp5_3_kendall.yaml > ${EXPERIMENT_DIR}/exp5_3.log 2>&1

# 4. PINN Composite
echo "Running Exp 5.4: PINN Composite..."
python src/main.py --config ${EXPERIMENT_DIR}/exp5_4_pinn.yaml > ${EXPERIMENT_DIR}/exp5_4.log 2>&1

echo "All experiments completed."
