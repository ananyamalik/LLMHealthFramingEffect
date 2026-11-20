#!/bin/bash
# Submits NUM_SHARDS extraction jobs (default 2) that run simultaneously, each on its own GPU and its own share of the
# reviews, then the analysis (CPU), which starts only after all of them succeed. From reasoning_anchors/:
#   bash jobs/submit.sh --num_reviews 60
#   NUM_SHARDS=4 bash jobs/submit.sh --num_reviews 120
#   MODEL=Qwen/Qwen3-1.7B MODEL_TAG=qwen3-1.7B bash jobs/submit.sh --num_reviews 60
# Arguments go to every extraction job. Shard i runs reviews i, i + n, i + 2n, ... of the sample and skips questions
# already in S3 (from any shard or earlier run), so rerunning the same command after a timeout only does what is missing.
# Outputs: s3://framing-interaction/reasoning_anchors/<model_tag>/ (S3_URI=... elsewhere, S3_URI= to stay in outputs/).
set -e
cd "$(dirname "$0")/.."
mkdir -p logs
NUM_SHARDS="${NUM_SHARDS:-2}"

TAG="${MODEL_TAG:-qwen3-0.6B}"
ids=()
for ((i = 0; i < NUM_SHARDS; i++)); do
    # With S3 each job gets its own working folder (scratch is shared between nodes): a job's resume download must not
    # overwrite the record file another job is appending to. Without S3 they share outputs/<tag>/ on purpose.
    if [ -n "${S3_URI-default}" ]; then
        export WORK_DIR="/scratch/${USER}/reasoning_anchors/${TAG}/shard_${i}"
    fi
    id=$(sbatch --parsable --export=ALL --job-name="ra_extract_${i}" jobs/extract.sh \
        --shard "$i" --num_shards "$NUM_SHARDS" "$@")
    ids+=("$id")
    echo "extract shard $i/$NUM_SHARDS: job $id   (log logs/ra_extract_${id}.txt)"
done
unset WORK_DIR   # the analysis uses its own default folder
deps=$(IFS=:; echo "${ids[*]}")
analyze_id=$(sbatch --parsable --export=ALL --dependency=afterok:"$deps" jobs/analyze.sh)
echo "analyze: job $analyze_id, starts after jobs ${deps//:/, } all succeed   (log logs/ra_analyze_${analyze_id}.txt)"
echo "outputs: ${S3_URI-s3://framing-interaction/reasoning_anchors/${MODEL_TAG:-qwen3-0.6B}}"
