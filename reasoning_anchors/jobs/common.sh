#!/bin/bash
# Shared setup for extract.sh and analyze.sh (sourced, not run). Sets MODEL, MODEL_TAG, S3_URI, WORK_DIR, INPUT_PATH, PY
# and activates the env: on SLURM through mechanistic-search's patching_config.sh, locally (LOCAL=1) through conda.

export MODEL="${MODEL:-Qwen/Qwen3-0.6B}" MODEL_TAG="${MODEL_TAG:-qwen3-0.6B}"   # set before the shared config, which keeps them
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-2}"                       # region of the framing-interaction bucket
S3_URI="${S3_URI-s3://framing-interaction/reasoning_anchors/${MODEL_TAG}}"         # S3_URI= (empty) keeps everything local

if [ -n "$LOCAL" ]; then
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-personaLLM}"
    export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
    PY="$CONDA_PREFIX/bin/python"   # not python3: a system python3 can come first on PATH
    INPUT_PATH="../code/outputs/questions/qwen3_thinking-4B/extracted/cochrane_review_data_final_with_questions.json"
    DEFAULT_WORK_DIR="./outputs/${MODEL_TAG}"
else
    source ../mechanistic-search/patching_config.sh   # cds to SLURM_SUBMIT_DIR (reasoning_anchors/), activates the env
    PY=python3
    DEFAULT_WORK_DIR="/scratch/${USER}/reasoning_anchors/${MODEL_TAG}"   # temporary when S3 is used
fi
# Without S3 the work dir is where the run lives, so keep it in outputs/
[ -z "$S3_URI" ] && DEFAULT_WORK_DIR="./outputs/${MODEL_TAG}"
WORK_DIR="${WORK_DIR:-$DEFAULT_WORK_DIR}"
mkdir -p "$WORK_DIR"

if [ -n "$S3_URI" ]; then
    # Fail fast if boto3 or the AWS credentials are missing, before loading a model
    "$PY" -c "import boto3; boto3.client('sts').get_caller_identity()" \
        || { echo "ERROR: boto3 or AWS credentials not available (pip install boto3; aws configure)" >&2; exit 1; }
    S3_ARGS=(--s3_uri "$S3_URI")
else
    S3_ARGS=()
fi
echo "model: $MODEL | work dir: $WORK_DIR | s3: ${S3_URI:-none}"
