#!/bin/bash
# Shared environment + settings for the two activation patching jobs (sourced by both job scripts).

module purge
module load explorer anaconda3/2024.06

# `conda activate` only works in batch jobs after loading conda's shell functions
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate personaLLM

# Ignore packages in ~/.local so they can't shadow the conda env's (this caused the NumPy 1.x/2.x crash)
export PYTHONNOUSERSITE=1

export HF_HOME="/scratch/$USER/.cache"
export HUGGINGFACE_HUB_CACHE="/scratch/$USER/.cache"
export XDG_CACHE_HOME="/scratch/$USER/.cache"

# Fail fast if the env is missing anything
python3 -c "import torch, transformers, dotenv, jinja2, numpy, matplotlib; print('python', __import__('sys').executable, '| torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())" || exit 1

MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_TAG="llama3-8B-instruct"
QUESTION_SET="extracted"   # "extracted" (technical questions) or "simplified" (plain language questions)
INPUT_PATH="../code/outputs/questions/qwen3_thinking-4B/${QUESTION_SET}/cochrane_review_data_final_with_questions.json"
OUTPUT_DIR="./outputs/${MODEL_TAG}/${QUESTION_SET}"

NUM_REVIEWS=30      # x 12 question types = 360 pairs; use -1 for all reviews
MAX_NEW_TOKENS=512
