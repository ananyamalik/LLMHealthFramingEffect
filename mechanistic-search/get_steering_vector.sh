#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=steering_vector
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                              # any GPU type
#SBATCH --exclude=d1025
#SBATCH -o logs/steering_vector_output_%j.txt     # Standard output file
#SBATCH -e logs/steering_vector_error_%j.txt      # Standard error file

# Difference-of-means "My <role> said," vector (run 1 - run 2 at the last prompt token), added to "<claim>" at its last
# prompt token at every layer. First argument is the answer type (yes_no or full), further arguments go to python.
# Submit from mechanistic-search/:
#   mkdir -p logs
#   sbatch get_steering_vector.sh yes_no                                  # doctor, 100 reviews (80 fit / 20 test)
#   sbatch get_steering_vector.sh full --alphas 1 2 --generate_layers 8 16
#   sbatch get_steering_vector.sh yes_no --roles doctor friend enemy
#   NUM_REVIEWS=-1 sbatch get_steering_vector.sh yes_no                   # all 368 reviews
# Prompt: question only by default; PROMPT=articles sbatch ... for the long prompt with article summaries.
# Resumes from <output_dir>/<answer>/<role>/pairs.jsonl and reuses vectors.pt, so resubmit after a timeout.

ANSWER="${1:?usage: sbatch get_steering_vector.sh <yes_no|full> [extra python args]}"
shift
NUM_REVIEWS="${NUM_REVIEWS:-100}"   # more than patching_config.sh's 15: the vector needs training reviews
source ./patching_config.sh
set -eo pipefail

PROMPT="${PROMPT:-question_only}"
RUN_DIR="./outputs/${MODEL_TAG}/steering_${PROMPT}"
echo "prompt: $PROMPT | reviews: $NUM_REVIEWS | outputs: $RUN_DIR/$ANSWER"

python3 -u get_steering_vector.py \
    --answer "$ANSWER" \
    --prompt "$PROMPT" \
    --model "$MODEL" \
    --input_path "$INPUT_PATH" \
    --output_dir "$RUN_DIR/$ANSWER" \
    --num_reviews $NUM_REVIEWS \
    "$@"

conda deactivate
