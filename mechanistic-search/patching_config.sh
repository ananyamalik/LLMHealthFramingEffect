#!/bin/bash
# Shared environment + settings for the patching jobs (sourced by run_role_patching.sh).

cd "$SLURM_SUBMIT_DIR"

# Load the module, then conda's shell hook, then activate. Override the env with: CONDA_ENV=myenv sbatch ...
CONDA_ENV="${CONDA_ENV:-emotional-safety}"
module load anaconda3/2024.06
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# Stop if the env didn't actually activate (otherwise python silently falls back to the cluster's base anaconda)
case "$(python -c 'import sys; print(sys.prefix)')" in
    */envs/"$CONDA_ENV") echo "python: $(which python) ($(python --version))" ;;
    *) echo "ERROR: conda env '$CONDA_ENV' is not active (python is $(which python)). Check 'conda env list'." >&2; exit 1 ;;
esac

# Ignore packages in ~/.local so they can't shadow the conda env's (this caused the NumPy 1.x/2.x crash)
export PYTHONNOUSERSITE=1

export HF_HOME="/scratch/$USER/.cache"
export HUGGINGFACE_HUB_CACHE="/scratch/$USER/.cache"
export XDG_CACHE_HOME="/scratch/$USER/.cache"

# Fail fast if the env is missing anything
python -c "import torch, transformers, dotenv, jinja2, numpy, matplotlib; print('torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())" || exit 1

# Override per job with --export=ALL,MODEL=...,MODEL_TAG=... (see submit_role_patching.sh). OLMo 3 needs transformers >= 4.57.
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
MODEL_TAG="${MODEL_TAG:-llama3-8B-instruct}"
INPUT_PATH="../code/outputs/questions/qwen3_thinking-4B/extracted/cochrane_review_data_final_with_questions.json"
OUTPUT_DIR="./outputs/${MODEL_TAG}/role_patching"
NUM_REVIEWS="${NUM_REVIEWS:-15}"   # seeded sample of reviews, x 2 framings = 30 pairs per role (dataset has 368)
echo "model: $MODEL | outputs: $OUTPUT_DIR"
