#!/bin/bash
# Submits the diffmeans steering jobs for both models, one GPU job each for:
#   yes_no (last + question tokens), full answers steered at the last token, full answers steered at every question token
# If a model has no directions yet, one get_diffmeans_directions.sh job builds them first and its steering jobs wait
# for it (so they do not all rebuild them). Extra arguments go to run_diffmeans_steering.py for every job.
# Run from mechanistic-search/ on the login node:  bash submit_diffmeans_steering.sh
#   bash submit_diffmeans_steering.sh --num_reviews 2         # quick check of every job

MODELS=(   # "<hf name> <output tag>"
    "meta-llama/Meta-Llama-3-8B-Instruct llama3-8B-instruct"
    "allenai/OLMo-2-1124-7B-Instruct olmo2-7B-instruct"
)
JOBS=(     # "<answer> <positions>"
    "yes_no last question"
    "full last"
    "full question"
)

mkdir -p logs
for entry in "${MODELS[@]}"; do
    read -r model tag <<< "$entry"
    dir="./outputs/${tag}/diffmeans"
    dependency=""
    if [ ! -f "$dir/yes_no/directions.pt" ] || [ ! -f "$dir/last/directions.pt" ]; then
        id=$(sbatch --parsable --export="ALL,MODEL=$model,MODEL_TAG=$tag" --job-name="${tag}_diffmeans" get_diffmeans_directions.sh)
        echo "$tag: no directions yet, building them in job $id first"
        dependency="--dependency=afterok:$id"
    fi
    for job in "${JOBS[@]}"; do
        read -r answer positions <<< "$job"
        sbatch $dependency --export="ALL,MODEL=$model,MODEL_TAG=$tag,ANSWER=$answer,POSITIONS=$positions" \
            --job-name="${tag}_steer_${answer}_${positions// /_}" run_diffmeans_steering.sh "$@"
    done
done
squeue -u "$USER"
