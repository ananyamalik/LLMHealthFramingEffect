#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=patch_gen
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -o logs/gen_output_%j.txt            # Standard output file
#SBATCH -e logs/gen_error_%j.txt             # Standard error file

# Job 2 of 2. Needs the finished layer sweep (layer_sweep_summary.json) from run_layer_sweep.sh.
# Submit from the mechanistic-search folder after the sweep finishes, or chain it:
#   sbatch --dependency=afterok:<sweep_job_id> run_patched_generation.sh
# Resumes from patched_generations.jsonl, so resubmitting after a timeout continues where it left off.

source ./patching_config.sh
set -eo pipefail

if [ ! -f "$OUTPUT_DIR/sweep/layer_sweep_summary.json" ]; then
    echo "Missing $OUTPUT_DIR/sweep/layer_sweep_summary.json -- run run_layer_sweep.sh to completion first." >&2
    exit 1
fi

echo "Stage 2: patched generation at the top-3 layers from the sweep"
python3 -u run_patched_generation.py \
    --model "$MODEL" \
    --input_path "$INPUT_PATH" \
    --sweep_dir "$OUTPUT_DIR/sweep" \
    --output_dir "$OUTPUT_DIR/generation" \
    --layers auto \
    --top_k 3 \
    --max_new_tokens $MAX_NEW_TOKENS

conda deactivate
