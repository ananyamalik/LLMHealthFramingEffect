#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=patch_comma
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -o logs/comma_output_%j.txt          # Standard output file
#SBATCH -e logs/comma_error_%j.txt           # Standard error file

# Comma of "My doctor said," patched into the bare claim, before and after it (layers 0, 5, ..., 30; 15 reviews).
# Submit from the mechanistic-search folder (logs/ must exist):
#   mkdir -p logs && sbatch run_comma_patching.sh
# Extra arguments are passed to python, e.g. a 1-review check first: sbatch run_comma_patching.sh --debug
# Resumes from comma_patching_pairs.jsonl, so resubmitting after a timeout continues where it left off.

export CONTRAST=doctor
source ./patching_config.sh
set -eo pipefail

python3 -u run_comma_patching.py \
    --model "$MODEL" \
    --input_path "$INPUT_PATH" \
    --output_dir "$OUTPUT_DIR/comma" \
    --num_reviews 15 \
    --sample_pool $NUM_REVIEWS \
    --max_new_tokens $MAX_NEW_TOKENS \
    --layer_stride 5 \
    "$@"

conda deactivate
