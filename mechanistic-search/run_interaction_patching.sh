#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --job-name=inter_patch
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                                   # any GPU type
#SBATCH --exclude=d1025
#SBATCH -o logs/inter_patch_output_%j.txt  # Standard output file
#SBATCH -e logs/inter_patch_error_%j.txt   # Standard error file

# Patching test of the role x framing interaction vector on the written answer (run_interaction_patching.py).
# Default: a subsample for experimentation, 10 test reviews x layers 16 18 20 x strengths lambda 1 4 16, conditions
# inter inter_perp review random none+inter none+random (~3.3k answers). Outputs in interaction_patching/last_lambda1_4_16/.
# Finer sweep: --lambdas 1 2 4 8 16 (or e.g. --lambdas -4 -1 1 4 16 for both directions); fewer layers: --layers 16.
# Needs outputs/<model_tag>/diffmeans/activations.pt; the vectors (role_framing_vectors.py, CPU) are fit first if missing.
# Submit from mechanistic-search/:
#   mkdir -p logs
#   sbatch --export=ALL,MODEL=allenai/OLMo-2-1124-7B-Instruct,MODEL_TAG=olmo2-7B-instruct run_interaction_patching.sh
#   sbatch --export=ALL,MODEL=allenai/OLMo-2-1124-7B-Instruct,MODEL_TAG=olmo2-7B-instruct run_interaction_patching.sh --steer_generation
#   ... run_interaction_patching.sh --num_reviews 2 --layers 16              # smoke check
# Resumable: resubmit the same command after a timeout. Arguments go to run_interaction_patching.py.

source ./patching_config.sh
set -eo pipefail

DIFFMEANS="./outputs/${MODEL_TAG}/diffmeans"
VECTORS="${DIFFMEANS}/last/role_framing_vectors"
if [ ! -f "${VECTORS}/doctor/directions.pt" ]; then
    python3 -u role_framing_vectors.py --diffmeans_dir "$DIFFMEANS" --in_memory --roles doctor friend enemy
fi

python3 -u run_interaction_patching.py \
    --model "$MODEL" \
    --input_path "$INPUT_PATH" \
    --vectors_dir "$VECTORS" \
    --output_dir "./outputs/${MODEL_TAG}/interaction_patching" \
    "$@"

conda deactivate
