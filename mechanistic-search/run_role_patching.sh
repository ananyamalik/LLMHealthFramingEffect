#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=role_patch
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                         # any GPU type
#SBATCH --exclude=d1025
#SBATCH -o logs/role_patch_output_%j.txt     # Standard output file
#SBATCH -e logs/role_patch_error_%j.txt      # Standard error file

# "My <role> said, <claim>" -> "<claim>": last-token and comma patching at every layer.
# First argument is the answer type (yes_no or full), any further arguments go to python. Submit from mechanistic-search/:
#   mkdir -p logs
#   sbatch run_role_patching.sh yes_no                    # all three roles in one job
#   sbatch run_role_patching.sh full                      # all roles; add --roles doctor for one role
#   sbatch run_role_patching.sh yes_no --debug            # 1-review check
# Or submit everything at once with: bash submit_role_patching.sh
# Resumes from <output_dir>/<answer>/<role>/pairs.jsonl, so resubmitting after a timeout continues where it left off.

ANSWER="${1:?usage: sbatch run_role_patching.sh <yes_no|full> [extra python args]}"
shift
source ./patching_config.sh
set -eo pipefail

python3 -u run_role_patching.py \
    --answer "$ANSWER" \
    --model "$MODEL" \
    --input_path "$INPUT_PATH" \
    --output_dir "$OUTPUT_DIR/$ANSWER" \
    --num_reviews $NUM_REVIEWS \
    "$@"

conda deactivate
