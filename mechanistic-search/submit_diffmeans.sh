#!/bin/bash
# Submits get_diffmeans_directions.sh for both models at once: one GPU job each, outputs in
# outputs/<model tag>/diffmeans/ (collects that model's activations on the first run, reuses them after).
# Extra arguments go to python for both jobs, e.g. a quick check:  bash submit_diffmeans.sh --num_reviews 15
# Run from mechanistic-search/ on the login node:  bash submit_diffmeans.sh

MODELS=(   # "<hf name> <output tag>"
    "meta-llama/Meta-Llama-3-8B-Instruct llama3-8B-instruct"
    "allenai/OLMo-2-1124-7B-Instruct olmo2-7B-instruct"
)

mkdir -p logs
for entry in "${MODELS[@]}"; do
    read -r model tag <<< "$entry"
    sbatch --export="ALL,MODEL=$model,MODEL_TAG=$tag" --job-name="${tag}_diffmeans" get_diffmeans_directions.sh "$@"
done
squeue -u "$USER"
