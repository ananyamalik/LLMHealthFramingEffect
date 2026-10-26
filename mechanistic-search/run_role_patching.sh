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
# Prompt: question only by default; PROMPT=articles sbatch ... for the long prompt with article summaries.
# Patch every generated token too (full only): PATCH_GENERATION=1 sbatch run_role_patching.sh full
# Resumes from <output_dir>/<answer>/<role>/pairs.jsonl, so resubmitting after a timeout continues where it left off.

ANSWER="${1:?usage: sbatch run_role_patching.sh <yes_no|full> [extra python args]}"
shift
source ./patching_config.sh
set -eo pipefail

# PROMPT=question_only (default): just "Question: <question>" -> outputs/<model>/role_patching_question_only/
# PROMPT=articles: instructions + article summaries          -> outputs/<model>/role_patching/
PROMPT="${PROMPT:-question_only}"
RUN_DIR="$OUTPUT_DIR"
if [ "$PROMPT" != "articles" ]; then
    RUN_DIR="${OUTPUT_DIR}_${PROMPT}"
fi
# PATCH_GENERATION=1 (full only): also patch every generated token -> outputs/<model>/role_patching_<prompt>_every_token/
EXTRA_ARGS=()
if [ -n "$PATCH_GENERATION" ]; then
    RUN_DIR="${RUN_DIR}_every_token"
    EXTRA_ARGS+=(--patch_generation)
fi
echo "prompt: $PROMPT | patch generation: ${PATCH_GENERATION:-no} | outputs: $RUN_DIR/$ANSWER"

python3 -u run_role_patching.py \
    --answer "$ANSWER" \
    --prompt "$PROMPT" \
    --model "$MODEL" \
    --input_path "$INPUT_PATH" \
    --output_dir "$RUN_DIR/$ANSWER" \
    --num_reviews $NUM_REVIEWS \
    "${EXTRA_ARGS[@]}" \
    "$@"

conda deactivate
