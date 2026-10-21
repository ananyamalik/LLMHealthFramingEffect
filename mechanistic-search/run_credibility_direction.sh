#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=credibility
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -o logs/credibility_output_%j.txt          # Standard output file
#SBATCH -e logs/credibility_error_%j.txt           # Standard error file

# Credibility direction: behavioral effect of 8 sources, a direction at the source token that orders them, and steering
# friend runs along it (Yes/No readout, 30 reviews x 6 claims). Arguments go to python. Submit from the
# mechanistic-search folder, on a GPU with 24GB+:
#   mkdir -p logs
#   sbatch --gres=gpu:a100:1 run_credibility_direction.sh
#   sbatch --gres=gpu:a100:1 run_credibility_direction.sh --determiner indefinite --output_dir ./outputs/llama3-8B-instruct/extracted/credibility_indefinite
# Behavior and steering resume from their .jsonl files, so resubmitting after a timeout continues where it left off.
# To rerun only the CPU analysis: add --stages direction (the model is not loaded).

source ./patching_config.sh
set -eo pipefail

python3 -u run_credibility_direction.py \
    --model "$MODEL" \
    --input_path "$INPUT_PATH" \
    --output_dir "$OUTPUT_DIR/credibility" \
    --num_reviews $NUM_REVIEWS \
    --sample_pool $NUM_REVIEWS \
    "$@"

conda deactivate
