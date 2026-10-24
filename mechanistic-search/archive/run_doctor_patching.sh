#!/bin/bash
# "My doctor said <sentence>" -> "<sentence>" activation patching. Not a job itself: submits the two existing patching
# jobs with CONTRAST=doctor, chained so generation starts only if the sweep succeeds. Run from the mechanistic-search folder:
#   bash run_doctor_patching.sh
# <sentence> is each claim-style question (doctor, ai, testimonials, journals, friend, family) with its attribution
# removed, in both framings. Outputs go to outputs/<model_tag>/extracted/doctor/ (see patching_config.sh).
# Both stages resume, so after a timeout resubmit the stage that stopped with the same --export.

set -eo pipefail
mkdir -p logs

sweep_jid=$(sbatch --parsable --export=ALL,CONTRAST=doctor --job-name=patch_sweep_doctor run_layer_sweep.sh)
gen_jid=$(sbatch --parsable --export=ALL,CONTRAST=doctor --job-name=patch_gen_doctor \
    --dependency=afterok:$sweep_jid run_patched_generation.sh)

echo "Submitted doctor sweep: $sweep_jid"
echo "Submitted doctor patched generation: $gen_jid (after $sweep_jid)"
