#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=reasoning_anchors
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -o logs/reasoning_anchors_output_%j.txt
#SBATCH -e logs/reasoning_anchors_error_%j.txt

# Extract prompt + generation activations, then fit the framing / role / interaction vectors.
#
# Cluster (submit from reasoning_anchors/, which SLURM_SUBMIT_DIR then points at):
#   mkdir -p logs
#   sbatch jobs/run_extract.sh --num_reviews 60
#   sbatch --export=ALL,MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,MODEL_TAG=r1-distill-qwen-1.5B jobs/run_extract.sh
#
# Local (no SLURM; CPU, fp32), from reasoning_anchors/:
#   LOCAL=1 CONDA_ENV=personaLLM nohup bash jobs/run_extract.sh > logs/extract_local.log 2>&1 &
#
# Extra arguments go to extract_reasoning_states.py. It resumes from generations/generations.jsonl, so rerun after a
# timeout or interruption. Needs transformers >= 4.51 (Qwen3).

export MODEL="${MODEL:-Qwen/Qwen3-0.6B}" MODEL_TAG="${MODEL_TAG:-qwen3-0.6B}"

if [ -n "$LOCAL" ]; then
    cd "$(dirname "$0")/.."
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-personaLLM}"
    export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
    DTYPE=float32
    PY="$CONDA_PREFIX/bin/python"   # not python3: a system python3 can come first on PATH
    INPUT_PATH="../code/outputs/questions/qwen3_thinking-4B/extracted/cochrane_review_data_final_with_questions.json"
else
    source ../mechanistic-search/patching_config.sh   # cds to SLURM_SUBMIT_DIR, activates the env; keeps MODEL / MODEL_TAG
    DTYPE=bfloat16
    PY=python3
fi
set -eo pipefail
echo "model: $MODEL | outputs: ./outputs/${MODEL_TAG}"

"$PY" -u code/extract_reasoning_states.py \
    --model "$MODEL" --model_tag "$MODEL_TAG" --input_path "$INPUT_PATH" --dtype "$DTYPE" \
    --output_dir "./outputs/${MODEL_TAG}" "$@"
"$PY" -u code/reasoning_vectors.py --run_dir "./outputs/${MODEL_TAG}" --token_projections
