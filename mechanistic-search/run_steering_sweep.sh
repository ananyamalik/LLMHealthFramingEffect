#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=steering_sweep
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                                   # any GPU type
#SBATCH --exclude=d1025
#SBATCH -o logs/steering_sweep_output_%j.txt          # Standard output file
#SBATCH -e logs/steering_sweep_error_%j.txt           # Standard error file

# Steering sweep on the bare claim (yes/no D): beta x layer x token site, for the diffmeans framing / role directions,
# random controls, and the directions fit at the steered token. 25 sampled held-out reviews (50 prompts) by default.
# Needs the diffmeans directions (get_diffmeans_directions.sh); submit_steering_sweep.sh builds them if missing.
# Submit from mechanistic-search/:
#   mkdir -p logs
#   bash submit_steering_sweep.sh                                            # one model (OLMo 2 7B by default)
#   sbatch run_steering_sweep.sh --num_reviews 2 --layers 8 16 24            # quick check, Llama 3 8B
#   sbatch --export=ALL,MODEL=allenai/OLMo-2-1124-7B-Instruct,MODEL_TAG=olmo2-7B-instruct run_steering_sweep.sh
# Resumable (one file per prompt in prompts/), so resubmit after a timeout. A different grid needs a new OUT_NAME.

source ./patching_config.sh
set -eo pipefail

OUT_NAME="${OUT_NAME:-steering_sweep}"
python3 -u run_steering_sweep.py \
    --model "$MODEL" \
    --input_path "$INPUT_PATH" \
    --diffmeans_dir "./outputs/${MODEL_TAG}/diffmeans" \
    --output_dir "./outputs/${MODEL_TAG}/${OUT_NAME}" \
    "$@"

conda deactivate
