#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=ra_extract_cpu
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --partition=short
#SBATCH -o logs/ra_extract_cpu_%j.txt
#SBATCH -e logs/ra_extract_cpu_%j.err

# Step 1 of 2, on a CPU node (fp32): same as extract.sh. For Qwen3-0.6B this is faster than a GPU job, which spends its
# time on one-token steps and activation copies and leaves the GPU idle (Research Computing cancels idle GPU jobs).
# Usually submitted by jobs/submit.sh (the default). Extra arguments go to extract_reasoning_states.py.
#   sbatch jobs/extract_cpu.sh --num_reviews 100       (from reasoning_anchors/)

source jobs/common.sh
set -eo pipefail

"$PY" -u code/extract_reasoning_states.py \
    --model "$MODEL" --model_tag "$MODEL_TAG" --input_path "$INPUT_PATH" \
    --output_dir "$WORK_DIR" "${S3_ARGS[@]}" --device cpu "$@"
