#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=subject_probe
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                         # any GPU type
#SBATCH --exclude=d1025
#SBATCH -o logs/subject_probe_output_%j.txt  # Standard output file
#SBATCH -e logs/subject_probe_error_%j.txt   # Standard error file

# Last-token subject probes trained on MMLU-Pro (expert speaker / no speaker / mixed), applied to the medical claims
# (all reviews by default). Arguments go to python. Submit from mechanistic-search/:
#   mkdir -p logs
#   sbatch run_subject_probe.sh
#   sbatch run_subject_probe.sh --per_category 50 --num_reviews 15        # quick check
#   sbatch --export=ALL,MODEL=allenai/OLMo-2-1124-7B-Instruct,MODEL_TAG=olmo2-7B-instruct run_subject_probe.sh

source ./patching_config.sh
set -eo pipefail
python -c "import datasets" || { echo "ERROR: 'datasets' is missing: pip install datasets" >&2; exit 1; }

python3 -u run_subject_probe.py \
    --model "$MODEL" \
    --input_path "$INPUT_PATH" \
    --output_dir "./outputs/${MODEL_TAG}/subject_probe" \
    "$@"

conda deactivate
