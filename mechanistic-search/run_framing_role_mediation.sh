#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --job-name=framing_role_mediation
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                                   # any GPU type
#SBATCH --exclude=d1025
#SBATCH -o logs/framing_role_mediation_output_%j.txt  # Standard output file
#SBATCH -e logs/framing_role_mediation_error_%j.txt   # Standard error file

# Yes/no prompt. steer: framing steered negative at the last prompt token; how much the projection on the role vectors
# changes downstream, and how much of the effect is left when the role direction is blocked at later layers.
# patch: "My doctor said," activations patched into "My friend said," per position and layer, under each framing.
# Further arguments go to run_framing_role_mediation.py. Submit from mechanistic-search/:
#   mkdir -p logs
#   sbatch --export=ALL,MODEL=allenai/OLMo-2-1124-7B-Instruct,MODEL_TAG=olmo2-7B-instruct run_framing_role_mediation.sh
#   sbatch --export=ALL,MODEL=allenai/OLMo-2-1124-7B-Instruct,MODEL_TAG=olmo2-7B-instruct run_framing_role_mediation.sh --experiment patch --source_role doctor --target_role enemy
# Resumable (records.jsonl / pairs.jsonl), so resubmit after a timeout.

source ./patching_config.sh
set -eo pipefail

python3 -u run_framing_role_mediation.py \
    --model "$MODEL" \
    --input_path "$INPUT_PATH" \
    --output_dir "./outputs/${MODEL_TAG}/framing_role_mediation" \
    "$@"

conda deactivate
