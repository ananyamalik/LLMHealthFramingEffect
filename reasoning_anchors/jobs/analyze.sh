#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --job-name=ra_analyze
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --partition=short
#SBATCH -o logs/ra_analyze_%j.txt
#SBATCH -e logs/ra_analyze_%j.err

# Step 2 of 2 (CPU only): framing / role / interaction vectors at every site, transfer, behavior, token projections.
# Reads the run from S3 (trajectories streamed one at a time), uploads analysis/ back to S3. Usually submitted by
# jobs/submit.sh after the extraction job; on its own it uses whatever is already in S3:
#   sbatch jobs/analyze.sh                              (from reasoning_anchors/)
# Extra arguments go to reasoning_vectors.py.

source jobs/common.sh
set -eo pipefail

"$PY" -u code/reasoning_vectors.py --run_dir "$WORK_DIR" "${S3_ARGS[@]}" --token_projections "$@"

# The local copy was only a working folder: S3 has the run and now the analysis
if [ -n "$S3_URI" ] && [ -z "$LOCAL" ]; then
    rm -rf "$WORK_DIR"
fi
