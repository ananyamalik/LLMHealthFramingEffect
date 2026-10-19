#!/bin/bash
# Shared environment + settings for the two activation patching jobs (sourced by both job scripts).

cd "$SLURM_SUBMIT_DIR"

# Same conda setup as the working rejudge/attention jobs: load the module, then conda's shell hook, then activate.
# Override the env with: CONDA_ENV=myenv sbatch run_layer_sweep.sh
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
python -c "import torch, transformers, dotenv, jinja2, numpy, matplotlib; print('python', __import__('sys').executable, '| torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())" || exit 1

MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_TAG="llama3-8B-instruct"
QUESTION_SET="extracted"   # "extracted" (technical questions) or "simplified" (plain language questions)
INPUT_PATH="../code/outputs/questions/qwen3_thinking-4B/${QUESTION_SET}/cochrane_review_data_final_with_questions.json"
# Which prompts are patched (source -> target). Override per job with --export=ALL,CONTRAST=doctor (see run_doctor_patching.sh)
#   framing: positive framing -> negative framing
#   doctor:  "My doctor said <sentence>" -> "<sentence>" (needs QUESTION_SET="extracted"; simplified has no claim-style questions)
CONTRAST="${CONTRAST:-framing}"
OUTPUT_DIR="./outputs/${MODEL_TAG}/${QUESTION_SET}"
if [ "$CONTRAST" != "framing" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}/${CONTRAST}"
fi
echo "contrast: $CONTRAST | outputs: $OUTPUT_DIR"

NUM_REVIEWS=30      # framing: x 12 question types = 360 pairs; doctor: x 3 distinct claims x 2 framings = 180; -1 for all
MAX_NEW_TOKENS=512
