#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=patch_span
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -o logs/span_output_%j.txt           # Standard output file
#SBATCH -e logs/span_error_%j.txt            # Standard error file

# Span patching: the differing tokens ("diff", or "claim" for doctor_masked) or every position after them ("after"),
# at layers 0, 5, ..., 30; 15 reviews.
# First argument is the contrast, any further arguments go to python. Submit from the mechanistic-search folder:
#   mkdir -p logs
#   sbatch --job-name=patch_span_framing run_span_patching.sh framing              # positive control
#   sbatch --job-name=patch_span_docfriend run_span_patching.sh doctor_friend
#   sbatch --job-name=patch_span_docmask run_span_patching.sh doctor_masked     # prefix hidden by the attention mask
#   sbatch run_span_patching.sh framing --debug                                    # 1-review check
# Resumes from span_patching_pairs.jsonl, so resubmitting after a timeout continues where it left off.

SPAN_CONTRAST="${1:?usage: sbatch run_span_patching.sh <framing|doctor_friend> [extra python args]}"
shift
export CONTRAST="$SPAN_CONTRAST"   # framing -> outputs/<model>/extracted/span, others -> .../<contrast>/span
source ./patching_config.sh
set -eo pipefail

python3 -u run_span_patching.py \
    --contrast "$SPAN_CONTRAST" \
    --model "$MODEL" \
    --input_path "$INPUT_PATH" \
    --output_dir "$OUTPUT_DIR/span" \
    --num_reviews 15 \
    --sample_pool $NUM_REVIEWS \
    --max_new_tokens $MAX_NEW_TOKENS \
    --layer_stride 5 \
    "$@"

conda deactivate
