#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=diffmeans_steering
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                                   # any GPU type
#SBATCH --exclude=d1025
#SBATCH -o logs/diffmeans_steering_output_%j.txt      # Standard output file
#SBATCH -e logs/diffmeans_steering_error_%j.txt       # Standard error file

# Steer the bare claim (held-out reviews) with the diffmeans framing and role (doctor_minus_enemy) directions,
# separately and together. ANSWER=yes_no (default; one forward pass per condition) or full (the written answer,
# scored by a yes/no follow-up; a generation per condition). POSITIONS: "last", "question" or both (default).
# Runs from scratch: builds the directions first if they do not exist (get_diffmeans_directions.py, ~1h).
# Further arguments go to run_diffmeans_steering.py. Submit from mechanistic-search/:
#   mkdir -p logs
#   bash submit_diffmeans_steering.sh                                       # both models, every answer and position
#   sbatch run_diffmeans_steering.sh --num_reviews 5                         # quick check, Llama 3 8B, yes/no
#   ANSWER=full POSITIONS=last sbatch run_diffmeans_steering.sh --num_reviews 2 --max_new_tokens 64
#   sbatch --export=ALL,MODEL=allenai/OLMo-2-1124-7B-Instruct,MODEL_TAG=olmo2-7B-instruct run_diffmeans_steering.sh
# Resumes from <output_dir>/<answer>/<positions>/pairs.jsonl, so resubmit after a timeout.

source ./patching_config.sh
set -eo pipefail

DIFFMEANS_DIR="./outputs/${MODEL_TAG}/diffmeans"
ANSWER="${ANSWER:-yes_no}"
POSITIONS="${POSITIONS:-last question}"
echo "directions: $DIFFMEANS_DIR | answer: $ANSWER | positions: $POSITIONS"

if [ ! -f "$DIFFMEANS_DIR/yes_no/directions.pt" ] || [ ! -f "$DIFFMEANS_DIR/last/directions.pt" ]; then
    echo "No directions in $DIFFMEANS_DIR: building them first"
    python3 -u get_diffmeans_directions.py \
        --model "$MODEL" \
        --input_path "$INPUT_PATH" \
        --output_dir "$DIFFMEANS_DIR"
fi

for positions in $POSITIONS; do
    python3 -u run_diffmeans_steering.py \
        --model "$MODEL" \
        --input_path "$INPUT_PATH" \
        --diffmeans_dir "$DIFFMEANS_DIR" \
        --output_dir "./outputs/${MODEL_TAG}/diffmeans_steering" \
        --answer "$ANSWER" \
        --positions "$positions" \
        "$@"
done

conda deactivate
