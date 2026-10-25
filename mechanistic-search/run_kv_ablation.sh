#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=kv_ablation
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                         # any GPU type
#SBATCH --exclude=d1025
#SBATCH -o logs/kv_ablation_output_%j.txt    # Standard output file
#SBATCH -e logs/kv_ablation_error_%j.txt     # Standard error file

# Hide or replace (with doctor's) the role's keys/values in the clean run 1 (full answers). Arguments go to python.
# Submit from mechanistic-search/:
#   mkdir -p logs
#   sbatch run_kv_ablation.sh --roles friend --max_reviews 5
#   sbatch --export=ALL,MODEL=allenai/OLMo-2-1124-7B-Instruct,MODEL_TAG=olmo2-7B-instruct run_kv_ablation.sh --roles enemy
# Resumes from <output_dir>/<role>/ablation.jsonl.

source ./patching_config.sh
set -eo pipefail

python3 -u run_kv_ablation.py \
    --model "$MODEL" \
    --input_path "$INPUT_PATH" \
    --output_dir "$OUTPUT_DIR/kv_ablation" \
    --num_reviews $NUM_REVIEWS \
    "$@"

conda deactivate
