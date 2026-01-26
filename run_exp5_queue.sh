#!/bin/bash
# run_exp5_queue.sh
# Robust, sequential runner for Experiment Suite 5.
# Handles directory resolution, error checking, automated aggregation, and scheduler policy.

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

# 1. Resolve Root Directory (independent of where script is called)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
echo "Working Directory: $(pwd)"

# 2. Config
EXPERIMENT_DIR="configs/experiments_5_loss_study"
OUTPUT_DIR="outputs_exp5"
mkdir -p "${OUTPUT_DIR}"

SUMMARY_JSON="${OUTPUT_DIR}/queue_summary.json"
echo "[" > "${SUMMARY_JSON}" # Init JSON array

# 3. Experiments
echo "========================================================"
echo "🚀 Starting Experiment Suite 5 (Sequential Queue)"
echo "========================================================"

FAILED_RUNS=0
TOTAL_RUNS=4
RUN_COUNT=0

append_summary() {
    local run_id=$1
    local status=$2
    local reason=$3
    
    # Comma if not first
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
    
    echo "[Queue] Starting ${run_id} using ${config_name}..."
    echo "        Log: ${log_file}"
    
    # Use Detected Python
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
        append_summary "${run_id}" "FAILURE" "Crash (Exit ${exit_code})"
        FAILED_RUNS=$((FAILED_RUNS + 1))
        return 1
    fi
    
    # Check if run produced the best_metrics.json (Success Contract)
    if [ ! -f "${run_dir}/best_metrics.json" ]; then
        echo "❌ [Failure] ${run_id} did not produce best_metrics.json."
        echo "   Check log: ${log_file}"
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

echo "]" >> "${SUMMARY_JSON}" # Close array

echo "========================================================"
echo "📊 Running Aggregator (Phase 1 Analysis)"
echo "========================================================"

AGG_LOG="${OUTPUT_DIR}/aggregation.log"

if [ $FAILED_RUNS -eq $TOTAL_RUNS ]; then
    echo "❌ All experiments failed. Aborting aggregation."
    exit 1
fi

$PYTHON_CMD scripts/select_best_run.py --suite_dir "${OUTPUT_DIR}" > "${AGG_LOG}" 2>&1
AGG_EXIT=$?

if [ $AGG_EXIT -ne 0 ]; then
    echo "❌ Aggregation Failed. See ${AGG_LOG}"
    cat "${AGG_LOG}"
    exit 1
fi

echo "✅ Aggregation Complete."
cat "${AGG_LOG}"

# 4. Extension Run
echo "========================================================"
echo "🚀 Phase 2: Extension Run"
echo "========================================================"

# Read Winner from JSON (Robust Python One-Liner)
WINNER_ID=$($PYTHON_CMD -c "import json; print(json.load(open('${OUTPUT_DIR}/best_run_selection.json'))['winner_run_id'])")
echo "🏆 Winner ID: ${WINNER_ID}"

# Map Winner ID to Config (Robust Mapping)
if [[ "${WINNER_ID}" == "exp5_1" ]]; then EXT_CONFIG="${EXPERIMENT_DIR}/exp5_1_unitstd.yaml"; fi
if [[ "${WINNER_ID}" == "exp5_2" ]]; then EXT_CONFIG="${EXPERIMENT_DIR}/exp5_2_gradflow.yaml"; fi
if [[ "${WINNER_ID}" == "exp5_3" ]]; then EXT_CONFIG="${EXPERIMENT_DIR}/exp5_3_kendall.yaml"; fi
if [[ "${WINNER_ID}" == "exp5_4" ]]; then EXT_CONFIG="${EXPERIMENT_DIR}/exp5_4_pinn.yaml"; fi

EXT_RUN_DIR="${OUTPUT_DIR}/${WINNER_ID}_extended"
RESUME_CKPT="${OUTPUT_DIR}/${WINNER_ID}/checkpoints/best_model.pth"
EXT_LOG="${OUTPUT_DIR}/${WINNER_ID}_ext.log"

echo "Extending ${WINNER_ID} to 350 epochs..."
echo "Resume: ${RESUME_CKPT}"

nohup $PYTHON_CMD -m src.main \
    --config "${EXT_CONFIG}" \
    --epochs 350 \
    --seed 42 \
    --run-dir "${EXT_RUN_DIR}" \
    --resume-checkpoint "${RESUME_CKPT}" \
    > "${EXT_LOG}" 2>&1 &
    
EXT_PID=$!
wait $EXT_PID

if [ ! -f "${EXT_RUN_DIR}/best_metrics.json" ]; then
    echo "❌ [Failure] Extension run failed."
    exit 1
fi

echo "========================================================"
echo "🎉 Benchmark Suite Complete."
echo "   Summary: ${SUMMARY_JSON}"
echo "   Selection: ${OUTPUT_DIR}/best_run_selection.json"
echo "   Winner: ${WINNER_ID}"
echo "   Failed Runs: ${FAILED_RUNS}"
echo "========================================================"
