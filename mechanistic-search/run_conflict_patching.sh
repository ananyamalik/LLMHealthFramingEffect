#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=patch_conflict
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -o logs/conflict_output_%j.txt          # Standard output file
#SBATCH -e logs/conflict_error_%j.txt           # Standard error file

# Conflicting doctor claims ("My doctor said A, but my doctor also said B"), Yes/No readout: behaviour + per-claim
# patching at every layer (30 reviews x 2 wordings). Any arguments go to python. Submit from the mechanistic-search
# folder, on a GPU with 24GB+:
#   mkdir -p logs
#   sbatch --gres=gpu:a100:1 run_conflict_patching.sh
# Resumes from conflict_patching_pairs.jsonl, so resubmitting after a timeout continues where it left off.

export CONTRAST=doctor_conflict   # outputs in outputs/<model>/extracted/doctor_conflict/yes_no
source ./patching_config.sh
set -eo pipefail

python3 -u run_conflict_patching.py \
    --model "$MODEL" \
    --input_path "$INPUT_PATH" \
    --output_dir "$OUTPUT_DIR/yes_no" \
    --num_reviews $NUM_REVIEWS \
    --sample_pool $NUM_REVIEWS \
    "$@"

conda deactivate
