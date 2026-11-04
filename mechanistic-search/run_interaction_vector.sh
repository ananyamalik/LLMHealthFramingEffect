#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --job-name=interaction_vector
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                                   # any GPU type
#SBATCH --exclude=d1025
#SBATCH -o logs/interaction_vector_output_%j.txt  # Standard output file
#SBATCH -e logs/interaction_vector_error_%j.txt   # Standard error file

# Role x framing interaction vector h(neg, role) - h(neg, none) - h(pos, role) + h(pos, none) at the last prompt token,
# per layer: size vs a shuffled-label null, alignment with the yes/no answer direction, and how much lies outside the
# framing + role directions. Submit from mechanistic-search/:
#   mkdir -p logs
#   sbatch --export=ALL,MODEL=allenai/OLMo-2-1124-7B-Instruct,MODEL_TAG=olmo2-7B-instruct run_interaction_vector.sh
# states.pt is reused if present, so rerunning only redoes the analysis.

source ./patching_config.sh
set -eo pipefail

python3 -u run_interaction_vector.py \
    --model "$MODEL" \
    --input_path "$INPUT_PATH" \
    --output_dir "./outputs/${MODEL_TAG}/framing_role_interaction" \
    "$@"

conda deactivate