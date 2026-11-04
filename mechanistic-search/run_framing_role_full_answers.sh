#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=framing_role_full
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                                   # any GPU type
#SBATCH --exclude=d1025
#SBATCH -o logs/framing_role_full_output_%j.txt  # Standard output file
#SBATCH -e logs/framing_role_full_error_%j.txt   # Standard error file

# Full written answers ("Question: <claim>", greedy) for the key conditions of run_framing_role_mediation.py:
# steer: unsteered / -framing push / push + role clamped or ablated / push + random clamped, at layers 16 20 24.
# patch: "My doctor said," activations patched into "My friend said," at a few (positions, layer) groups.
# Further arguments go to run_framing_role_full_answers.py. Submit from mechanistic-search/:
#   mkdir -p logs
#   sbatch --export=ALL,MODEL=allenai/OLMo-2-1124-7B-Instruct,MODEL_TAG=olmo2-7B-instruct run_framing_role_full_answers.sh
#   sbatch --export=ALL,MODEL=allenai/OLMo-2-1124-7B-Instruct,MODEL_TAG=olmo2-7B-instruct run_framing_role_full_answers.sh --experiment patch
# Resumable (steer_answers.jsonl / patch_answers.jsonl), so resubmit after a timeout.

source ./patching_config.sh
set -eo pipefail

python3 -u run_framing_role_full_answers.py \
    --model "$MODEL" \
    --input_path "$INPUT_PATH" \
    --output_dir "./outputs/${MODEL_TAG}/framing_role_mediation" \
    "$@"

conda deactivate
