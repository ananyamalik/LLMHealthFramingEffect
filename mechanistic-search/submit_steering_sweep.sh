#!/bin/bash
# Submits the steering sweep (run_steering_sweep.sh) for one model, one GPU job (default OLMo 2 7B; override with
# MODEL=... MODEL_TAG=...). If the model has no diffmeans directions yet, one get_diffmeans_directions.sh job builds
# them first and the sweep waits for it. Extra arguments go to run_steering_sweep.py.
# Run from mechanistic-search/ on the login node:
#   bash submit_steering_sweep.sh
#   OUT_NAME=sweep_test bash submit_steering_sweep.sh --num_reviews 2 --layers 8 16 24      # quick check
#   MODEL=meta-llama/Meta-Llama-3-8B-Instruct MODEL_TAG=llama3-8B-instruct bash submit_steering_sweep.sh

model="${MODEL:-allenai/OLMo-2-1124-7B-Instruct}"
tag="${MODEL_TAG:-olmo2-7B-instruct}"

mkdir -p logs
dir="./outputs/${tag}/diffmeans"
dependency=""
if [ ! -f "$dir/yes_no/directions.pt" ] || [ ! -f "$dir/verdict/directions.pt" ]; then
    id=$(sbatch --parsable --export="ALL,MODEL=$model,MODEL_TAG=$tag" --job-name="${tag}_diffmeans" get_diffmeans_directions.sh)
    echo "$tag: no directions yet, building them in job $id first"
    dependency="--dependency=afterok:$id"
fi
sbatch $dependency --export="ALL,MODEL=$model,MODEL_TAG=$tag" --job-name="${tag}_steering_sweep" run_steering_sweep.sh "$@"
squeue -u "$USER"
