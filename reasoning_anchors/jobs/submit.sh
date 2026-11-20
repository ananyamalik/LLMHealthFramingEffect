#!/bin/bash
# Submits the two steps: extraction (GPU), then the analysis (CPU), which starts only if the extraction succeeds.
# From reasoning_anchors/:
#   bash jobs/submit.sh --num_reviews 60
#   MODEL=Qwen/Qwen3-1.7B MODEL_TAG=qwen3-1.7B bash jobs/submit.sh --num_reviews 60
# Arguments go to the extraction step. Outputs: s3://framing-interaction/reasoning_anchors/<model_tag>/
# (S3_URI=... for another location, S3_URI= to keep everything in outputs/ on Explorer).
set -e
cd "$(dirname "$0")/.."
mkdir -p logs
extract_id=$(sbatch --parsable --export=ALL jobs/extract.sh "$@")
analyze_id=$(sbatch --parsable --export=ALL --dependency=afterok:"$extract_id" jobs/analyze.sh)
echo "extract job $extract_id -> analyze job $analyze_id (starts after $extract_id succeeds)"
echo "outputs: ${S3_URI-s3://framing-interaction/reasoning_anchors/${MODEL_TAG:-qwen3-0.6B}}"
