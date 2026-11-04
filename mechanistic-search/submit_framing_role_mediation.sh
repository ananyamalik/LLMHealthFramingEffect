#!/bin/bash
# Submits the framing -> role mediation jobs, two GPU jobs per model:
#   yes/no answer   run_framing_role_mediation.sh     (projection, role block, doctor -> friend patching)
#   full answers    run_framing_role_full_answers.sh  (the written answers for the same steering and patching)
# Both write to outputs/<tag>/framing_role_mediation/ (different files) and resume after a timeout.
# Default: Llama 3 8B Instruct only (OLMo is done). Run from mechanistic-search/ on the login node:
#   bash submit_framing_role_mediation.sh
#   MODELS_TO_RUN="llama3-8B-instruct olmo2-7B-instruct" bash submit_framing_role_mediation.sh
# Extra arguments go to both scripts (e.g. --num_reviews 2 for a quick check of both jobs).

declare -A MODELS=(   # <output tag> -> <hf name>
    ["llama3-8B-instruct"]="meta-llama/Meta-Llama-3-8B-Instruct"
    ["olmo2-7B-instruct"]="allenai/OLMo-2-1124-7B-Instruct"
)
MODELS_TO_RUN="${MODELS_TO_RUN:-llama3-8B-instruct}"

mkdir -p logs
for tag in $MODELS_TO_RUN; do
    model="${MODELS[$tag]}"
    sbatch --export="ALL,MODEL=$model,MODEL_TAG=$tag" --job-name="${tag}_framing_role_yes_no" run_framing_role_mediation.sh "$@"
    sbatch --export="ALL,MODEL=$model,MODEL_TAG=$tag" --job-name="${tag}_framing_role_full" run_framing_role_full_answers.sh "$@"
done
squeue -u "$USER"
