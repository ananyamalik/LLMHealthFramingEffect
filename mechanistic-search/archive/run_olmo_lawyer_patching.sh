#!/bin/bash
# Two variants of the doctor -> friend patching, each run with both readouts (span = long-form, yes_no = Yes/No).
# Not a job itself: submits the existing span / Yes/No jobs. Run from the mechanistic-search folder:
#   bash run_olmo_lawyer_patching.sh
#   1. OLMo:   doctor -> friend with a small OLMo model instead of Llama 3 8B
#              outputs in outputs/olmo2-1B-instruct/extracted/doctor_friend/{span,yes_no}/
#   2. Lawyer: doctor -> lawyer with Llama 3 8B
#              outputs in outputs/llama3-8B-instruct/extracted/doctor_lawyer/{span,yes_no}/
# Every job resumes, so after a timeout resubmit the same sbatch line.

set -eo pipefail
mkdir -p logs

OLMO_MODEL="allenai/OLMo-2-0425-1B-Instruct"   # 16 layers, 4k context (pairs longer than that are skipped and logged)
OLMO_TAG="olmo2-1B-instruct"
OLMO_EXPORT="ALL,MODEL=$OLMO_MODEL,MODEL_TAG=$OLMO_TAG"

# 1. OLMo, doctor -> friend. Span patching uses every layer (--layer_stride 1) since OLMo has only 16.
olmo_span=$(sbatch --parsable --export="$OLMO_EXPORT" --job-name=olmo_span_docfr \
    run_span_patching.sh doctor_friend --layer_stride 1)
olmo_yesno=$(sbatch --parsable --export="$OLMO_EXPORT" --job-name=olmo_yesno_docfr \
    run_yes_no_patching.sh doctor_friend)

# 2. Llama 3 8B, doctor -> lawyer (24GB+ GPU, like the other Llama runs)
lawyer_span=$(sbatch --parsable --gres=gpu:a100:1 --job-name=patch_span_doclaw \
    run_span_patching.sh doctor_lawyer)
lawyer_yesno=$(sbatch --parsable --gres=gpu:a100:1 --job-name=yesno_doclaw \
    run_yes_no_patching.sh doctor_lawyer)

echo "Submitted OLMo doctor -> friend:  span $olmo_span, yes/no $olmo_yesno"
echo "Submitted Llama doctor -> lawyer: span $lawyer_span, yes/no $lawyer_yesno"
