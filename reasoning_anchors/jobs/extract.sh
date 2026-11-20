#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=ra_extract
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -o logs/ra_extract_%j.txt
#SBATCH -e logs/ra_extract_%j.err

# Step 1 of 2: generate (reasoning + answer) and record prompt + generation activations for every question.
# Each question's files are uploaded to S3 (s3://framing-interaction/reasoning_anchors/<model_tag>/) as soon as it
# finishes and deleted locally; a rerun resumes from the generations.jsonl in S3. Usually submitted by jobs/submit.sh.
#   sbatch jobs/extract.sh --num_reviews 60            (from reasoning_anchors/)
# Any GPU works (bf16 on A100+, fp16 on older cards). Extra arguments go to extract_reasoning_states.py.

source jobs/common.sh
set -eo pipefail

"$PY" -u code/extract_reasoning_states.py \
    --model "$MODEL" --model_tag "$MODEL_TAG" --input_path "$INPUT_PATH" \
    --output_dir "$WORK_DIR" "${S3_ARGS[@]}" "$@"
