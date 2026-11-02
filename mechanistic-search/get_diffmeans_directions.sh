#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=diffmeans
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                         # any GPU type
#SBATCH --exclude=d1025
#SBATCH -o logs/diffmeans_output_%j.txt     # Standard output file
#SBATCH -e logs/diffmeans_error_%j.txt      # Standard error file

# Difference-of-means framing and role directions (framing, shared "has a role", role-specific, role contrasts) in the
# question-only prompt's residual stream. Runs from scratch: collects the activations (all reviews x both framings x
# bare claim + doctor/friend/enemy, every layer and anchor) into <output_dir>/activations.pt, then fits the directions.
# A resubmit (e.g. --anchors last) reuses activations.pt and skips the model. Arguments go to python.
# Submit from mechanistic-search/:
#   mkdir -p logs
#   sbatch get_diffmeans_directions.sh
#   sbatch get_diffmeans_directions.sh --num_reviews 15                  # quick check
#   sbatch --export=ALL,MODEL=allenai/OLMo-2-1124-7B-Instruct,MODEL_TAG=olmo2-7B-instruct get_diffmeans_directions.sh
# Both models at once: bash submit_diffmeans.sh

source ./patching_config.sh
set -eo pipefail

python3 -u get_diffmeans_directions.py \
    --model "$MODEL" \
    --input_path "$INPUT_PATH" \
    --output_dir "./outputs/${MODEL_TAG}/diffmeans" \
    "$@"

conda deactivate
