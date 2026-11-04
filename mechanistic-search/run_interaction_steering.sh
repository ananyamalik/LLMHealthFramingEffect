#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --job-name=interaction_steering
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                                   # any GPU type
#SBATCH --exclude=d1025
#SBATCH -o logs/interaction_steering_output_%j.txt  # Standard output file
#SBATCH -e logs/interaction_steering_error_%j.txt   # Standard error file

# Role x framing interaction from steering, layers 12-20: forward (steer framing, does the role effect change?),
# reverse (steer a role vector, does the framing effect change?), each against a random direction of the same strength.
# Uses the directions and test split of run_framing_role_mediation.sh, so run that first. Submit from mechanistic-search/:
#   mkdir -p logs
#   sbatch --export=ALL,MODEL=allenai/OLMo-2-1124-7B-Instruct,MODEL_TAG=olmo2-7B-instruct run_interaction_steering.sh
# Resumable (records.jsonl), so resubmit after a timeout.

source ./patching_config.sh
set -eo pipefail

python3 -u run_interaction_steering.py \
    --model "$MODEL" \
    --input_path "$INPUT_PATH" \
    --directions_path "./outputs/${MODEL_TAG}/framing_role_mediation/directions.pt" \
    --output_dir "./outputs/${MODEL_TAG}/framing_role_interaction_steering" \
    "$@"

conda deactivate