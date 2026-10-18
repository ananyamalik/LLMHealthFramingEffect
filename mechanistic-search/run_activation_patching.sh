#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=act_patch
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -o output_%j.txt                     # Standard output file
#SBATCH -e error_%j.txt                      # Standard error file

# Submit from the mechanistic-search folder:  cd mechanistic-search && sbatch run_activation_patching.sh
# Both stages resume from their .jsonl files, so resubmitting after a timeout continues where it left off.

module purge
module load explorer anaconda3/2024.06

source activate base
source activate llm_health_framing_effect
conda activate llm_health_framing_effect

export HF_HOME="/scratch/$USER/.cache"
export HUGGINGFACE_HUB_CACHE="/scratch/$USER/.cache"
export XDG_CACHE_HOME="/scratch/$USER/.cache"

MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_TAG="llama3-8B-instruct"
QUESTION_SET="extracted"   # "extracted" (technical questions) or "simplified" (plain language questions)
INPUT_PATH="../code/outputs/questions/qwen3_thinking-4B/${QUESTION_SET}/cochrane_review_data_final_with_questions.json"
OUTPUT_DIR="./outputs/${MODEL_TAG}/${QUESTION_SET}"

NUM_REVIEWS=30      # x 12 question types = 360 pairs; use -1 for all reviews
MAX_NEW_TOKENS=512

echo "Stage 1: layer sweep"
python3 -u run_layer_sweep.py \
    --model "$MODEL" \
    --input_path "$INPUT_PATH" \
    --output_dir "$OUTPUT_DIR/sweep" \
    --num_reviews $NUM_REVIEWS \
    --max_new_tokens $MAX_NEW_TOKENS \
    --num_last_tokens 1

python3 -u plot_layer_sweep.py --sweep_dir "$OUTPUT_DIR/sweep"

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
