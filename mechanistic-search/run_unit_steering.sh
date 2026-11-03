#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=unit_steering
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                                   # any GPU type
#SBATCH --exclude=d1025
#SBATCH -o logs/unit_steering_output_%j.txt           # Standard output file
#SBATCH -e logs/unit_steering_error_%j.txt            # Standard error file

# Extracts unit-length framing and doctor/friend/enemy vectors at the last prompt token (training reviews), then
# steers the bare claim's last prompt token with each and saves the written answers (test reviews), for every
# SCALES entry in turn (default: residual, then proj_std; the directions are fit once and reused).
# Further arguments go to run_unit_steering.py. Submit from mechanistic-search/:
#   mkdir -p logs
#   sbatch --export=ALL,MODEL=allenai/OLMo-2-1124-7B-Instruct,MODEL_TAG=olmo2-7B-instruct run_unit_steering.sh
#   sbatch --export=ALL,MODEL=allenai/OLMo-2-1124-7B-Instruct,MODEL_TAG=olmo2-7B-instruct run_unit_steering.sh --num_reviews 1
# Resumes from <output_dir>/<scale>/answers.jsonl, so resubmit after a timeout.

source ./patching_config.sh
set -eo pipefail

SCALES="${SCALES:-residual proj_std}"
for scale in $SCALES; do
    python3 -u run_unit_steering.py \
        --model "$MODEL" \
        --input_path "$INPUT_PATH" \
        --output_dir "./outputs/${MODEL_TAG}/unit_steering" \
        --scale "$scale" \
        "$@"
done

conda deactivate
