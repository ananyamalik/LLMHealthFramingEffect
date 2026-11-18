#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --job-name=bilinear
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                                   # any GPU type
#SBATCH --exclude=d1025
#SBATCH -o logs/bilinear_output_%j.txt  # Standard output file
#SBATCH -e logs/bilinear_error_%j.txt   # Standard error file

# Bilinear check of role x framing with intermediate framings (run_bilinear_check.py): collects last-token residuals
# for 7 framing levels x (bare + doctor/friend/enemy) x all 368 reviews (~10k short forward passes, no generation),
# then analyzes on CPU. Submit from mechanistic-search/:
#   mkdir -p logs
#   sbatch --export=ALL,MODEL=allenai/OLMo-2-1124-7B-Instruct,MODEL_TAG=olmo2-7B-instruct run_bilinear_check.sh
#   ... run_bilinear_check.sh --num_reviews 100       # smaller sample
# acts.pt is reused if present; re-run only the analysis on CPU with --analyze_only. Arguments go to python.

source ./patching_config.sh
set -eo pipefail

python3 -u run_bilinear_check.py \
    --model "$MODEL" \
    --input_path "$INPUT_PATH" \
    --output_dir "./outputs/${MODEL_TAG}/bilinear_check" \
    "$@"

conda deactivate
