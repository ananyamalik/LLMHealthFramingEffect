# Reasoning anchors: framing × role vectors in a small reasoning model

The framing / role / interaction analysis of `mechanistic-search/` (`run_interaction_vector.py`,
`role_framing_vectors.py`, `get_diffmeans_directions.py`), run on a small local **reasoning** model, with the residual
stream recorded on the prompt **and at every token the model generates** (its reasoning trace and its answer).

Default model: `Qwen/Qwen3-0.6B` with thinking on (28 layers, d_model 1024). It runs on a laptop CPU.
`Qwen/Qwen3-1.7B` and `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` also work (`--model ... --model_tag ...`).

## Setup

The prompts are the same as mechanistic-search's question-only prompts (`prompts/question_only.jinja2`). Each review
gets both framings × (bare claim, doctor, friend, enemy), which is 8 questions:

```text
Question: [My <role> said, ]amantadine will|won't work for fatigue in multiple sclerosis. What do you think?
```

The model writes greedily: `<think> reasoning </think>`, then its answer. If the reasoning has not closed after
`--max_think_tokens` (512), `</think>` is forced (budget forcing). Those rows are flagged `forced_think_end`, and the
answer gets up to `--max_answer_tokens` (192) more.

**Residual stream** = the output of each decoder layer (as in `patching_utils.cache_residuals`). It is recorded with
forward hooks *during* KV-cached generation: the prefill pass gives every prompt token, and each decode step gives the
token just fed in. `gen_states[:, i]` is the residual at generated token i, which predicts token i + 1.

**Sites** (one `[layers, d_model]` vector per question):

| group | sites |
|---|---|
| prompt | `intervention`, `verdict` (` work`), `claim_end` (`.`), `question_end` (`?`), `last` (starts the generation), `yes_no` (last token of the yes/no prompt, thinking off) |
| reasoning | `think_bin_0..9` (10 equal slices of the trace, mean of each), `think_end` (`</think>`), `think_mean` |
| answer | `answer_first`, `answer_bin_0..3`, `answer_last`, `answer_mean`; `gen_mean` = every generated token |

**Behavior**, per question (one forward pass each, thinking off):
- `d_nothink` = log p(Yes) − log p(No) on the yes/no prompt.
- `d_after` = the same yes/no question asked as a follow-up turn after the model's own answer.

## Vectors (`reasoning_vectors.py`)

Per site, role and layer, from per-review cells (A00 = `X won't work`, A01 = `X will work`, A10 / A11 = the same
after `My <role> said,`), as in `role_framing_vectors.py`:

- `f` = A01 − A00 (framing), `d_pos` = A11 − A01, `d_neg` = A10 − A00, `role` = (`d_pos` + `d_neg`) / 2
- `inter` = `d_pos` − `d_neg` = (A11 − A10) − (A01 − A00), split into `inter_par` (along f) and `inter_perp`.
  Its sign is the opposite of `run_interaction_vector.py`'s I. Here > 0 along f means the role widens the framing gap.
- Significance: a sign-flip null for |mean f|, |mean role| and |mean inter|. Swapping a factor's two levels within a
  review flips the sign of that review's vector. `*_snr` = |mean| / null 95th percentile, so values above 1 are above
  review-to-review noise.
- `gain` = inter·f̂ / |f|, `perp_frac`, `frac_outside` (the part of inter outside span(f, role)), `cos_answer` (with the
  yes/no logit-lens direction), and `cos_f_last` / `cos_inter_last` (whether the prompt-time direction is still the
  same direction at this site).
- **Transfer** (`transfer.csv`, leave one review out): directions fit at `last` on the other reviews, read out on the
  held-out review at every site:
  - framing AUC
  - role AUC
  - t statistic of the held-out interaction projected on the `last` interaction direction
  
  This tests whether the framing and the role × framing signal set up before the first generated token are still
  carried through the reasoning.
- `--token_projections`: every generated token projected on the `last` vectors (`projections/<key>.npz`), plotted
  over the relative position in the reasoning trace.

Caveat: at generation sites the 8 questions of a review have different texts (each wrote its own reasoning), so a
generation-site vector compares "where the model is at 30% of its reasoning" across conditions, not matched tokens.
The prompt sites are matched token for token, as in mechanistic-search.

## Layout

```text
reasoning_anchors/
  code/extract_reasoning_states.py   step 1 (GPU): prompt + generation activations and the generated text
  code/reasoning_vectors.py          step 2 (CPU): framing / role / interaction vectors at every site
  code/s3_store.py                   S3 helper, also a CLI to copy a run folder to / from the bucket
  jobs/submit.sh                     submits step 1, then step 2 (starts only if step 1 succeeds)
  jobs/extract.sh, jobs/analyze.sh   the two SLURM jobs (also run locally with LOCAL=1)
  jobs/common.sh                     shared env / paths / S3 settings
```

## Storage: S3

Runs are stored in `s3://framing-interaction/reasoning_anchors/<model_tag>/` (bucket region us-east-2), with the same
layout as in "Outputs" below. Explorer only holds a temporary working folder (`/scratch/$USER/reasoning_anchors/<tag>`):

- **extract.sh** uploads each question's activations as soon as it finishes and deletes the local copy, and re-uploads
  `generations.jsonl` / `.csv` after every question. A resubmit resumes from the `generations.jsonl` in S3.
- **analyze.sh** downloads everything except the trajectories, streams the trajectories one at a time for the token
  projections, uploads `analysis/` back to S3, and deletes its working folder.

`S3_URI=s3://other/prefix` writes somewhere else; `S3_URI=` (empty) keeps everything in `outputs/` without S3.

**One-time setup on Explorer:** the env needs boto3 and AWS credentials. Both jobs check this first and stop with an
error if either is missing.

```bash
conda activate emotional-safety && pip install boto3
aws configure          # or put the key in ~/.aws/credentials; region us-east-2
python -c "import boto3; print(boto3.client('sts').get_caller_identity()['Arn'])"
```

## Running

On Explorer, from `reasoning_anchors/`:

```bash
bash jobs/submit.sh --num_reviews 60               # extract (GPU, 8 h) -> analyze (CPU, partition short)
MODEL=Qwen/Qwen3-1.7B MODEL_TAG=qwen3-1.7B bash jobs/submit.sh --num_reviews 60
sbatch jobs/analyze.sh                              # rerun only the analysis on what is in S3
```

- Arguments to `submit.sh` go to the extraction step. If it times out, run `bash jobs/submit.sh` with the same arguments
  again: questions already in S3 are skipped. The 16-review sample is the first 16 of the 60-review sample (same seed),
  so going from 16 to 60 only runs the 44 new reviews.
- Logs: `logs/ra_extract_<jobid>.txt` (progress bar: tokens/s, forced `</think>`, d_after per question) and
  `logs/ra_analyze_<jobid>.txt`.
- Any GPU works (bf16 on A100+, fp16 on V100/T4/P100, chosen automatically).

**Copying a run** (works on the Mac with the AWS CLI, or anywhere with `code/s3_store.py`; reruns skip finished files):

```bash
# existing outputs on Explorer -> S3
python code/s3_store.py upload outputs/qwen3-0.6B s3://framing-interaction/reasoning_anchors/qwen3-0.6B
# S3 -> Mac, without the per-token trajectories (several GB)
aws s3 sync s3://framing-interaction/reasoning_anchors/qwen3-0.6B outputs/qwen3-0.6B --exclude "activations/trajectories/*"
# one review's trajectories
aws s3 sync s3://framing-interaction/reasoning_anchors/qwen3-0.6B/activations/trajectories \
    outputs/qwen3-0.6B/activations/trajectories --exclude "*" --include "CD006827_*"
```

**Local (Mac, CPU, fp32)**: the same scripts with `LOCAL=1` (env `personaLLM`, which has boto3):

```bash
mkdir -p logs
LOCAL=1 nohup bash jobs/extract.sh --num_reviews 16 > logs/extract_local.log 2>&1 &   # writes to S3
LOCAL=1 bash jobs/analyze.sh
LOCAL=1 S3_URI= bash jobs/extract.sh                 # no S3: everything in outputs/<model_tag>/
```

About 5 tokens/s on this MacBook (2–3 min per question). Disk per question: about 1.5 MB of sites and about 30–40 MB of
trajectories (`--traj_layers 6 12 18 24 27` or `--no_trajectories` to save less).

## Outputs (`outputs/<model_tag>/`)

```text
generations/
  generations.jsonl     per question: prompt text, raw generation (with <think> tags), reasoning, answer, token
                        counts and spans, anchor positions, d_nothink, d_after, forced_think_end, tokens/s
  generations.csv       the same without positions/raw text, one row per question (easy to read in a spreadsheet)
activations/
  sites/<key>.pt        {site: [layers, d_model] fp16}: prompt anchors + reasoning / answer sites
  trajectories/<key>.pt prompt_states [layers, prompt tokens, d] and gen_states [layers, generated tokens, d] (fp16):
                        the residual stream at every prompt token and every generated token; prompt_ids, gen_ids,
                        think_start / think_end / answer_start
  answer_dir.pt         yes/no logit-lens direction
meta.json
analysis/
  directions.pt         {site: {f, framing, framing_<role>, d_pos_<role>, d_neg_<role>, role_<role>, inter_<role>,
                        inter_par_<role>, inter_perp_<role>: [layers, d_model]}}
  summary.csv           site x role x layer metrics above
  transfer.csv          leave-one-out readout of the `last` directions at every site
  behavior.csv          framing gap and role x framing interaction in d_after / d_nothink; reasoning lengths
  interaction_sites.png per role: inter snr, gain, frac_outside (site x layer)
  timeline_L<l>.png     framing / role / interaction snr along prompt -> reasoning -> answer
  transfer_last.png     leave-one-out framing AUC and interaction projection along the timeline
  token_projection_L<l>.png, projections/<key>.npz
```

Keys are `<review_id>_<framing>_<role>`, the same in all three folders.
