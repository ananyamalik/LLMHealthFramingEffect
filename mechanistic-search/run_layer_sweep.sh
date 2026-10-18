#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=patch_sweep
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -o logs/sweep_output_%j.txt          # Standard output file
#SBATCH -e logs/sweep_error_%j.txt           # Standard error file

# Job 1 of 2. Submit from the mechanistic-search folder (logs/ must exist):
#   mkdir -p logs && sbatch run_layer_sweep.sh
# Resumes from layer_sweep_pairs.jsonl, so resubmitting after a timeout continues where it left off.

source ./patching_config.sh
set -e

echo "Stage 1: layer sweep"
python3 -u run_layer_sweep.py \
    --model "$MODEL" \
    --input_path "$INPUT_PATH" \
    --output_dir "$OUTPUT_DIR/sweep" \
    --num_reviews $NUM_REVIEWS \
    --max_new_tokens $MAX_NEW_TOKENS \
    --num_last_tokens 1

python3 -u plot_layer_sweep.py --sweep_dir "$OUTPUT_DIR/sweep"

conda deactivate
