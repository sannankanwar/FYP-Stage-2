#!/bin/bash
# run_experiments_5.sh
# Automates Experiment Suite 5:
# 1. Runs 4 experiments in parallel (100 epochs)
# 2. Selects the winner based on coverage metrics
# 3. Extends the winner by 250 epochs

EXPERIMENT_DIR="configs/experiments_5_loss_study"
OUTPUT_DIR="outputs_exp5"

mkdir -p ${OUTPUT_DIR}

echo "========================================================"
echo "🚀 Starting Experiment Suite 5 (4 Parallel Runs)"
echo "========================================================"

# 1. Unit Standardized (Baseline)
echo "Starting Exp 5.1: Unit Standardized..."
nohup python -m src.main --config ${EXPERIMENT_DIR}/exp5_1_unitstd.yaml --epochs 100 --seed 42 --run_dir ${OUTPUT_DIR}/exp5_1 > ${OUTPUT_DIR}/exp5_1.log 2>&1 &
PID1=$!

# 2. Gradient Consistency
echo "Starting Exp 5.2: Gradient Consistency..."
nohup python -m src.main --config ${EXPERIMENT_DIR}/exp5_2_gradflow.yaml --epochs 100 --seed 42 --run_dir ${OUTPUT_DIR}/exp5_2 > ${OUTPUT_DIR}/exp5_2.log 2>&1 &
PID2=$!

# 3. Kendall Uncertainty
echo "Starting Exp 5.3: Kendall Uncertainty..."
nohup python -m src.main --config ${EXPERIMENT_DIR}/exp5_3_kendall.yaml --epochs 100 --seed 42 --run_dir ${OUTPUT_DIR}/exp5_3 > ${OUTPUT_DIR}/exp5_3.log 2>&1 &
PID3=$!

# 4. Physics Consistency (PINN)
echo "Starting Exp 5.4: Physics Consistency..."
nohup python -m src.main --config ${EXPERIMENT_DIR}/exp5_4_pinn.yaml --epochs 100 --seed 42 --run_dir ${OUTPUT_DIR}/exp5_4 > ${OUTPUT_DIR}/exp5_4.log 2>&1 &
PID4=$!

echo "Waiting for experiments to finish (PIDs: $PID1, $PID2, $PID3, $PID4)..."
wait $PID1 $PID2 $PID3 $PID4

echo "========================================================"
echo "✅ Phase 1 Complete. Analyzing Results..."
echo "========================================================"

# Select Winner
python scripts/select_best_run.py --suite_dir ${OUTPUT_DIR}

# Read Selection
if [ -f "${OUTPUT_DIR}/best_run_selection.json" ]; then
    # Python one-liner to parse JSON
    WINNER_DIR=$(python -c "import json; print(json.load(open('${OUTPUT_DIR}/best_run_selection.json'))['exp_dir'])")
    
    echo "========================================================"
    echo "🏆 Winner Identified: ${WINNER_DIR}"
    echo "🚀 Starting Phase 2: Extension (Total 350 Epochs)..."
    echo "========================================================"
    
    # Identify Config used by Winner (Assume naming convention or read from experiment_info.md? 
    # Simpler: The script select_best_run.py returned the directory. We know the mapping map config from directory name?
    # Actually, src.main saves 'config.yaml' or similar in the run_dir? No, Trainer saves 'config' in checkpoint.
    # But to restart, we need the original yaml path OR just pass the run params.
    # The safest way is to map directory name back to config file.
    
    if [[ "${WINNER_DIR}" == *"exp5_1"* ]]; then CONFIG="${EXPERIMENT_DIR}/exp5_1_unitstd.yaml"; fi
    if [[ "${WINNER_DIR}" == *"exp5_2"* ]]; then CONFIG="${EXPERIMENT_DIR}/exp5_2_gradflow.yaml"; fi
    if [[ "${WINNER_DIR}" == *"exp5_3"* ]]; then CONFIG="${EXPERIMENT_DIR}/exp5_3_kendall.yaml"; fi
    if [[ "${WINNER_DIR}" == *"exp5_4"* ]]; then CONFIG="${EXPERIMENT_DIR}/exp5_4_pinn.yaml"; fi

    EXT_DIR="${WINNER_DIR}_extended"
    CHECKPOINT="${WINNER_DIR}/checkpoints/best_model.pth"
    
    echo "Config: ${CONFIG}"
    echo "Resume: ${CHECKPOINT}"
    echo "Output: ${EXT_DIR}"
    
    nohup python -m src.main \
        --config ${CONFIG} \
        --epochs 350 \
        --seed 42 \
        --run_dir ${EXT_DIR} \
        --resume_checkpoint ${CHECKPOINT} \
        > ${OUTPUT_DIR}/winner_extension.log 2>&1 &
        
    echo "Extension run started in background. Log: ${OUTPUT_DIR}/winner_extension.log"
    
else
    echo "❌ Error: Could not determine winner (best_run_selection.json missing)."
fi
