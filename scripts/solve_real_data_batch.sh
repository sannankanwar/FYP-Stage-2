#!/bin/bash
# Batch process all real data files using solve_real_data.py
# Usage: ./scripts/solve_real_data_batch.sh <checkpoint_path>

set -u

CHECKPOINT=${1:-"outputs/test/dry_run_coord_gfx/checkpoints/best_model.pth"}
INPUT_DIR="real_data"
OUTPUT_DIR="outputs/real_data_solutions"

mkdir -p "$OUTPUT_DIR"

echo "Starting Batch Processing of Real Data..."
echo "Checkpoint: $CHECKPOINT"
echo "Input Dir: $INPUT_DIR"

# Loop over CSV files
for file in "$INPUT_DIR"/*.csv; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        echo "Queuing $filename..."
        
        # Run in background via nohup? 
        # Or sequential? sequential is safer for GPU memory (unless we have multiple GPUs).
        # User asked for nohup script to "do all".
        # We can wrap the whole loop in nohup, or run each in background.
        # Given 94 files, running 94 parallel GPU processes will crash the machine.
        # We should run SEQUENTIALLY inside a SINGLE nohup block.
        
        python scripts/solve_real_data.py \
            --input_file "$file" \
            --checkpoint "$CHECKPOINT" \
            --output_dir "$OUTPUT_DIR" \
            --pop_size 50 \
            --max_iter 100 \
            --crop_size 1024
            
    fi
done

echo "Batch Processing Complete!"
