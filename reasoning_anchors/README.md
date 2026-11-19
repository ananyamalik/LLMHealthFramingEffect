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
  code/extract_reasoning_states.py   runs the model: prompt + generation activations and the generated text
  code/reasoning_vectors.py          CPU analysis: framing / role / interaction vectors at every site
  jobs/run_extract.sh                both steps, locally (LOCAL=1) or on SLURM
  outputs/<model_tag>/               see below
```

## Running

Local (CPU, fp32), in the background, from `reasoning_anchors/`:

```bash
mkdir -p logs
LOCAL=1 CONDA_ENV=personaLLM nohup bash jobs/run_extract.sh > logs/extract_local.log 2>&1 &
tail -f logs/extract_local.log        # progress bar: tokens/s, forced </think>, d_after per question
```

Extra arguments go to the extraction step, e.g. `bash jobs/run_extract.sh --num_reviews 8 --max_think_tokens 384`.
The analysis runs automatically when the extraction finishes. To rerun only the analysis (it uses every review that
already has all 8 questions, so it also works mid-run):

```bash
conda activate personaLLM
python code/reasoning_vectors.py --run_dir outputs/qwen3-0.6B --token_projections
```

- Env: `personaLLM` (python 3.10, torch 2.2, transformers 4.57; Qwen3 needs >= 4.51). The job script calls the env's
  own python, because the system `python3` comes first on this machine's PATH.
- Speed on this MacBook (i5, fp32): about 5 tokens/s for Qwen3-0.6B, so roughly 2–3 min per question at the default
  budgets and 4–6 h for 16 reviews x 8 questions. If the run stops, rerun the same command: it resumes from
  `generations/generations.jsonl`.
- Disk: about 1.5 MB of sites per question, and about 60 KB per generated token for full trajectories at all 28 layers
  (about 40 MB for a 700-token question, about 5 GB for the default run). `--traj_layers 6 12 18 24 27` or
  `--no_trajectories` saves less.
- Cluster: `mkdir -p logs && sbatch jobs/run_extract.sh --num_reviews 60` from `reasoning_anchors/` (mechanistic-search's
  `patching_config.sh` env, bf16 on GPU).

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
