#!/bin/bash
# run_exp5_queue.sh
# Robust, sequential runner for Experiment Suite 5.
# Handles directory resolution, error checking, and automated aggregation.

set -euo pipefail

# 1. Resolve Root Directory (independent of where script is called)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
echo "Working Directory: $(pwd)"

# 2. Config
EXPERIMENT_DIR="configs/experiments_5_loss_study"
OUTPUT_DIR="outputs_exp5"
mkdir -p "${OUTPUT_DIR}"

# 3. Experiments
echo "========================================================"
echo "🚀 Starting Experiment Suite 5 (Sequential Queue)"
echo "========================================================"

run_experiment() {
    local config_name=$1
    local run_id=$2
    local config_path="${EXPERIMENT_DIR}/${config_name}"
    local run_dir="${OUTPUT_DIR}/${run_id}"
    local log_file="${OUTPUT_DIR}/${run_id}.log"
    
    echo "[Queue] Starting ${run_id} using ${config_name}..."
    echo "        Log: ${log_file}"
    
    # Run in background to decouple, but wait immediately
    # We use python -m src.main
    nohup python -m src.main \
        --config "${config_path}" \
        --epochs 100 \
        --seed 42 \
        --run-dir "${run_dir}" \
        > "${log_file}" 2>&1 &
        
    local pid=$!
    wait $pid
    
    # Check if run produced the best_metrics.json (Success Contract)
    if [ ! -f "${run_dir}/best_metrics.json" ]; then
        echo "❌ [Failure] ${run_id} did not produce best_metrics.json."
        echo "   Check log: ${log_file}"
        # We fail fast? Or continue? Prompt said "fail immediately... unless configured"
        # We will fail fast to prevent wasting time on broken pipeline.
        exit 1
    else
        echo "✅ [Success] ${run_id} completed."
    fi
}

# Run 4 Experiments Sequentially (Safer than parallel for debugging)
run_experiment "exp5_1_unitstd.yaml" "exp5_1"
run_experiment "exp5_2_gradflow.yaml" "exp5_2"
run_experiment "exp5_3_kendall.yaml" "exp5_3"
run_experiment "exp5_4_pinn.yaml" "exp5_4"

echo "========================================================"
echo "📊 Running Aggregator (Phase 1 Analysis)"
echo "========================================================"

AGG_LOG="${OUTPUT_DIR}/aggregation.log"
python scripts/select_best_run.py --suite_dir "${OUTPUT_DIR}" > "${AGG_LOG}" 2>&1 || {
    echo "❌ Aggregation Failed. See ${AGG_LOG}"
    cat "${AGG_LOG}"
    exit 1
}

echo "✅ Aggregation Complete."
cat "${AGG_LOG}"

# 4. Extension Run
echo "========================================================"
echo "🚀 Phase 2: Extension Run"
echo "========================================================"

# Read Winner from JSON (Robust Python One-Liner)
WINNER_ID=$(python -c "import json; print(json.load(open('${OUTPUT_DIR}/best_run_selection.json'))['winner_run_id'])")
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

nohup python -m src.main \
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
echo "   Selection: ${OUTPUT_DIR}/best_run_selection.json"
echo "   Winner: ${WINNER_ID}"
echo "   Tail logs: tail -f ${OUTPUT_DIR}/*.log"
echo "========================================================"
