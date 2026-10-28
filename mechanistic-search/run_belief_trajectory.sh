#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --job-name=belief_trajectory
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                         # any GPU type
#SBATCH --exclude=d1025
#SBATCH -o logs/belief_trajectory_output_%j.txt  # Standard output file
#SBATCH -e logs/belief_trajectory_error_%j.txt   # Standard error file

# Belief trajectory: the belief probe read out at every token of the question-only prompt and the generated full
# answer, for 5 reviews x 2 framings x (bare + 3 roles). Trains the probe on outputs/<model>/belief_probe/activations.pt
# if run_belief_probe.sh has run, else on 100 other reviews collected in this job. Arguments go to python.
# Submit from mechanistic-search/:
#   mkdir -p logs
#   sbatch run_belief_trajectory.sh
#   sbatch run_belief_trajectory.sh --plot_layer 16 --roles doctor
#   sbatch --export=ALL,MODEL=allenai/OLMo-2-1124-7B-Instruct,MODEL_TAG=olmo2-7B-instruct run_belief_trajectory.sh

source ./patching_config.sh
set -eo pipefail

python3 -u run_belief_trajectory.py \
    --model "$MODEL" \
    --input_path "$INPUT_PATH" \
    --output_dir "./outputs/${MODEL_TAG}/belief_trajectory" \
    --probe_dir "./outputs/${MODEL_TAG}/belief_probe" \
    "$@"

conda deactivate
