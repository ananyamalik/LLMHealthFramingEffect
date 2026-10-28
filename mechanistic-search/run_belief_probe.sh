#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=belief_probe
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                         # any GPU type
#SBATCH --exclude=d1025
#SBATCH -o logs/belief_probe_output_%j.txt  # Standard output file
#SBATCH -e logs/belief_probe_error_%j.txt   # Standard error file

# Belief probe: does the full-answer (question-only) prompt's residual stream, along the claim, already carry the
# model's yes/no belief that the treatment works? All reviews and 3 roles by default. Arguments go to python.
# Submit from mechanistic-search/:
#   mkdir -p logs
#   sbatch run_belief_probe.sh
#   sbatch run_belief_probe.sh --num_reviews 15 --permutations 5          # quick check
#   sbatch --export=ALL,MODEL=allenai/OLMo-2-1124-7B-Instruct,MODEL_TAG=olmo2-7B-instruct run_belief_probe.sh

source ./patching_config.sh
set -eo pipefail

python3 -u run_belief_probe.py \
    --model "$MODEL" \
    --input_path "$INPUT_PATH" \
    --output_dir "./outputs/${MODEL_TAG}/belief_probe" \
    "$@"

conda deactivate
