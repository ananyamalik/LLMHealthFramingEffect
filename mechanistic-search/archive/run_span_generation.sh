#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=span_gen
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -o logs/spangen_output_%j.txt        # Standard output file
#SBATCH -e logs/spangen_error_%j.txt         # Standard error file

# Patched generations for the span contrasts (20 pairs each, 7 conditions in total) for the evidence-direction check.
# Needs the finished span runs. Submit from the mechanistic-search folder, on a GPU with 24GB+:
#   mkdir -p logs && sbatch --gres=gpu:1 --exclude=d1025 run_span_generation.sh
# Resumes from span_generations.jsonl, so resubmitting after a timeout continues where it left off.

source ./patching_config.sh
set -eo pipefail

python3 -u run_span_generation.py \
    --model "$MODEL" \
    --input_path "$INPUT_PATH" \
    --span_root "./outputs/${MODEL_TAG}/${QUESTION_SET}" \
    --output_dir "./outputs/${MODEL_TAG}/${QUESTION_SET}/span_generation" \
    --max_new_tokens $MAX_NEW_TOKENS \
    "$@"

conda deactivate
