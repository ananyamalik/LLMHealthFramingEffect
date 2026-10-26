#!/bin/bash
# Submits the role-patching jobs for both models: one yes/no job and one full-answer job each (all roles, 15 reviews).
# Prompt: question only by default; PROMPT=articles bash submit_role_patching.sh for the long prompt.
# For a bigger sample: NUM_REVIEWS=50 bash submit_role_patching.sh (then split full answers per role with --roles).
# Run from mechanistic-search/ on the login node:  bash submit_role_patching.sh
# When they finish:  python plot_role_patching.py

MODELS=(   # "<hf name> <output tag>"
    "meta-llama/Meta-Llama-3-8B-Instruct llama3-8B-instruct"
    "allenai/OLMo-2-1124-7B-Instruct olmo2-7B-instruct"
)

mkdir -p logs
for entry in "${MODELS[@]}"; do
    read -r model tag <<< "$entry"
    exports="ALL,MODEL=$model,MODEL_TAG=$tag,NUM_REVIEWS=${NUM_REVIEWS:-15},PROMPT=${PROMPT:-question_only}"
    sbatch --export="$exports" --job-name="${tag}_yesno" run_role_patching.sh yes_no
    sbatch --export="$exports" --job-name="${tag}_full" run_role_patching.sh full
done
squeue -u "$USER"
