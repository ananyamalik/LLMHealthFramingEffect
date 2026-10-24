#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=attn_ablation
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -o logs/ablation_output_%j.txt       # Standard output file
#SBATCH -e logs/ablation_error_%j.txt        # Standard error file

# Attention ablation: block attention to "My doctor said," from the answer / the prompt / both (30 reviews, 180 pairs).
# Submit from the mechanistic-search folder, on a GPU with 24GB+ (Llama 3 8B does not fit on a 16GB T4):
#   mkdir -p logs && sbatch --gres=gpu:a100:1 run_attention_ablation.sh
# Extra arguments are passed to python, e.g. a 1-review check first: sbatch run_attention_ablation.sh --debug
# Resumes from attention_ablation_pairs.jsonl, so resubmitting after a timeout continues where it left off.

export CONTRAST=doctor
source ./patching_config.sh
set -eo pipefail

python3 -u run_attention_ablation.py \
    --model "$MODEL" \
    --input_path "$INPUT_PATH" \
    --output_dir "$OUTPUT_DIR/attention_ablation" \
    --num_reviews $NUM_REVIEWS \
    --sample_pool $NUM_REVIEWS \
    --max_new_tokens $MAX_NEW_TOKENS \
    "$@"

conda deactivate
