#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=risk_dir
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -o logs/riskdir_output_%j.txt        # Standard output file
#SBATCH -e logs/riskdir_error_%j.txt         # Standard error file

# Risk direction: fit risk/valence directions, projection test on doctor/friend/bare prompts, steering the doctor prompt.
# Submit from the mechanistic-search folder, on a GPU with 24GB+:
#   mkdir -p logs && sbatch --gres=gpu:1 --exclude=d1025 run_risk_direction.sh
# Extra arguments go to python, e.g. --steer_source friend, or --debug for a 4-review check.

source ./patching_config.sh
set -eo pipefail

python3 -u run_risk_direction.py \
    --model "$MODEL" \
    --input_path "$INPUT_PATH" \
    --output_dir "./outputs/${MODEL_TAG}/${QUESTION_SET}/risk_direction" \
    --num_reviews $NUM_REVIEWS \
    --sample_pool $NUM_REVIEWS \
    "$@"

conda deactivate
