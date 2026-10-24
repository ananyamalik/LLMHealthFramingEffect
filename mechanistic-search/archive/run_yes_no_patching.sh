#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=patch_yesno
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -o logs/yesno_output_%j.txt          # Standard output file
#SBATCH -e logs/yesno_error_%j.txt           # Standard error file

# Yes/No single-token readout: last-token and differing-token patching at every layer (30 reviews, no generation).
# First argument is the contrast, any further arguments go to python. Submit from the mechanistic-search folder,
# on a GPU with 24GB+ (Llama 3 8B does not fit on a 16GB T4):
#   mkdir -p logs
#   sbatch --gres=gpu:a100:1 --job-name=yesno_framing run_yes_no_patching.sh framing
#   sbatch --gres=gpu:a100:1 --job-name=yesno_doctor  run_yes_no_patching.sh doctor
#   sbatch --gres=gpu:a100:1 --job-name=yesno_docfr   run_yes_no_patching.sh doctor_friend
# Resumes from yes_no_patching_pairs.jsonl, so resubmitting after a timeout continues where it left off.

YESNO_CONTRAST="${1:?usage: sbatch run_yes_no_patching.sh <framing|doctor|doctor_friend|doctor_lawyer> [extra python args]}"
shift
export CONTRAST="$YESNO_CONTRAST"   # framing -> outputs/<model>/extracted/yes_no, others -> .../<contrast>/yes_no
source ./patching_config.sh
set -eo pipefail

python3 -u run_yes_no_patching.py \
    --contrast "$YESNO_CONTRAST" \
    --model "$MODEL" \
    --input_path "$INPUT_PATH" \
    --output_dir "$OUTPUT_DIR/yes_no" \
    --num_reviews $NUM_REVIEWS \
    --sample_pool $NUM_REVIEWS \
    "$@"

conda deactivate
