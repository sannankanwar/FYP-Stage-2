#!/bin/bash
# run_exp5_queue.sh
# Robust, sequential runner with failure diagnostics.

set -u

# 0. Python Auto-Detection
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "❌ [Fatal] Check failed: No 'python' or 'python3' found."
    exit 127
fi

echo "Using Python: $PYTHON_CMD ($(which $PYTHON_CMD))"

# Check Environment
echo "Checking Environment..."
$PYTHON_CMD -c "import torch; import numpy; import yaml; print('Environment OK: Torch ' + torch.__version__)" || {
    echo "❌ [Fatal] Environment check failed. Missing dependencies?"
    exit 1
}

# 1. Resolve Root Directory
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
echo "Working Directory: $(pwd)"

# 2. Config
EXPERIMENT_DIR="configs/experiments_5_loss_study"
OUTPUT_DIR="outputs_exp5"
mkdir -p "${OUTPUT_DIR}"

SUMMARY_JSON="${OUTPUT_DIR}/queue_summary.json"
echo "[" > "${SUMMARY_JSON}"

# 3. Experiments
FAILED_RUNS=0
TOTAL_RUNS=4
RUN_COUNT=0

append_summary() {
    local run_id=$1
    local status=$2
    local reason=$3
    if [ "$RUN_COUNT" -gt 0 ]; then echo "," >> "${SUMMARY_JSON}"; fi
    echo "  { \"run_id\": \"$run_id\", \"status\": \"$status\", \"reason\": \"$reason\" }" >> "${SUMMARY_JSON}"
    RUN_COUNT=$((RUN_COUNT + 1))
}

run_experiment() {
    local config_name=$1
    local run_id=$2
    local config_path="${EXPERIMENT_DIR}/${config_name}"
    local run_dir="${OUTPUT_DIR}/${run_id}"
    local log_file="${OUTPUT_DIR}/${run_id}.log"
    
    echo "--------------------------------------------------------"
    echo "[Queue] Starting ${run_id}..."
    echo "        Config: ${config_path}"
    echo "        Log: ${log_file}"
    
    # Check if config exists
    if [ ! -f "${config_path}" ]; then
         echo "❌ [Failure] Config file not found: ${config_path}"
         return 1
    fi
    
    nohup $PYTHON_CMD -m src.main \
        --config "${config_path}" \
        --epochs 100 \
        --seed 42 \
        --run-dir "${run_dir}" \
        > "${log_file}" 2>&1 &
        
    local pid=$!
    wait $pid
    local exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        echo "❌ [Failure] ${run_id} crashed with exit code ${exit_code}."
        echo "👇 --- LAST 50 LINES OF LOG --- 👇"
        tail -n 50 "${log_file}"
        echo "👆 ---------------------------- 👆"
        
        append_summary "${run_id}" "FAILURE" "Crash (Exit ${exit_code})"
        FAILED_RUNS=$((FAILED_RUNS + 1))
        return 1
    fi
    
    if [ ! -f "${run_dir}/best_metrics.json" ]; then
        echo "❌ [Failure] ${run_id} missing best_metrics.json."
        echo "👇 --- LAST 20 LINES OF LOG --- 👇"
        tail -n 20 "${log_file}"
        append_summary "${run_id}" "FAILURE" "Missing best_metrics.json"
        FAILED_RUNS=$((FAILED_RUNS + 1))
        return 1
    else
        echo "✅ [Success] ${run_id} completed."
        append_summary "${run_id}" "SUCCESS" "Complete"
        return 0
    fi
}

run_experiment "exp5_1_unitstd.yaml" "exp5_1"
run_experiment "exp5_2_gradflow.yaml" "exp5_2"
run_experiment "exp5_3_kendall.yaml" "exp5_3"
run_experiment "exp5_4_pinn.yaml" "exp5_4"

echo "]" >> "${SUMMARY_JSON}"

echo "========================================================"
echo "📊 Phase 1 Summary"
echo "========================================================"
if [ $FAILED_RUNS -eq $TOTAL_RUNS ]; then
    echo "❌ All experiments failed. Stopping."
    exit 1
fi

AGG_LOG="${OUTPUT_DIR}/aggregation.log"
$PYTHON_CMD scripts/select_best_run.py --suite_dir "${OUTPUT_DIR}" > "${AGG_LOG}" 2>&1
AGG_EXIT=$?

if [ $AGG_EXIT -ne 0 ]; then
    echo "❌ Aggregation Failed."
    tail -n 20 "${AGG_LOG}"
    exit 1
fi

echo "✅ Aggregation Complete."

# 4. Extension Run
WINNER_ID=$($PYTHON_CMD -c "import json; print(json.load(open('${OUTPUT_DIR}/best_run_selection.json'))['winner_run_id'])")
echo "🏆 Winner ID: ${WINNER_ID}"

if [[ "${WINNER_ID}" == "exp5_1" ]]; then EXT_CONFIG="${EXPERIMENT_DIR}/exp5_1_unitstd.yaml"; fi
if [[ "${WINNER_ID}" == "exp5_2" ]]; then EXT_CONFIG="${EXPERIMENT_DIR}/exp5_2_gradflow.yaml"; fi
if [[ "${WINNER_ID}" == "exp5_3" ]]; then EXT_CONFIG="${EXPERIMENT_DIR}/exp5_3_kendall.yaml"; fi
if [[ "${WINNER_ID}" == "exp5_4" ]]; then EXT_CONFIG="${EXPERIMENT_DIR}/exp5_4_pinn.yaml"; fi

EXT_RUN_DIR="${OUTPUT_DIR}/${WINNER_ID}_extended"
RESUME_CKPT="${OUTPUT_DIR}/${WINNER_ID}/checkpoints/best_model.pth"
EXT_LOG="${OUTPUT_DIR}/${WINNER_ID}_ext.log"

echo "Extending ${WINNER_ID} to 350 epochs..."
nohup $PYTHON_CMD -m src.main \
    --config "${EXT_CONFIG}" \
    --epochs 350 \
    --seed 42 \
    --run-dir "${EXT_RUN_DIR}" \
    --resume-checkpoint "${RESUME_CKPT}" \
    > "${EXT_LOG}" 2>&1 &
    
wait $!
if [ ! -f "${EXT_RUN_DIR}/best_metrics.json" ]; then
    echo "❌ [Failure] Extension run failed."
    tail -n 50 "${EXT_LOG}"
    exit 1
fi

echo "🎉 Benchmark Suite Complete."
