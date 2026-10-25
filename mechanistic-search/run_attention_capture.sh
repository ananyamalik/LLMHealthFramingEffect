#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=attn_capture
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                         # any GPU type
#SBATCH --exclude=d1025
#SBATCH -o logs/attn_capture_output_%j.txt   # Standard output file
#SBATCH -e logs/attn_capture_error_%j.txt    # Standard error file

# Attention per generation step in the clean runs and the patched run 2 (full answers). Arguments go to python.
# Submit from mechanistic-search/:
#   mkdir -p logs
#   sbatch run_attention_capture.sh --roles doctor --framings negative --review_ids CD006827 CD007956
#   sbatch run_attention_capture.sh --max_reviews 2
#   sbatch --export=ALL,MODEL=allenai/OLMo-2-1124-7B-Instruct,MODEL_TAG=olmo2-7B-instruct run_attention_capture.sh --max_reviews 2
# Resumes from <output_dir>/<role>/attention.jsonl.

source ./patching_config.sh
set -eo pipefail

python3 -u run_attention_capture.py \
    --model "$MODEL" \
    --input_path "$INPUT_PATH" \
    --output_dir "$OUTPUT_DIR/attention" \
    --num_reviews $NUM_REVIEWS \
    "$@"

conda deactivate
