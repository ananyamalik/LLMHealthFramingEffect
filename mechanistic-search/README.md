# Mechanistic search: role patching

Does the role in `My <role> said, <claim>` change the answer, and where is that information carried?

Earlier experiments (layer sweep, span/comma patching, credibility and risk directions, conflict, attention ablation)
are in `archive/`, with their own README. They were written to run from this folder, so move a script back here
before running it again. Their outputs are still in `outputs/`.

## Setup

Each review gives one claim, in two framings: the `doctor` question with "My doctor says " removed.

| framing  | claim                                                                          |
|----------|--------------------------------------------------------------------------------|
| positive | `amantadine will work for fatigue in multiple sclerosis. What do you think?`  |
| negative | `amantadine won't work for fatigue in multiple sclerosis. What do you think?` |

For each role (`doctor`, `friend`, `enemy`) and framing:

- run 1 (source): `My <role> said, <claim>`
- run 2 (target): `<claim>`

The claim and everything after it (article summaries, chat template) are the same tokens in both runs; the script
checks this.

**Patching setups.** The residual stream of run 1 is written into run 2 at one layer L, for every layer, one setup at
a time:

- `last`: last prompt token of run 1 -> last prompt token of run 2
- `comma`: the `,` of `My <role> said,` in run 1 -> the token just before the claim in run 2 (`**:` from
  `**QUESTION**:`, the position where the role would start)

**Answer versions** (`--answer`):

- `yes_no`: `prompts/question_answering_yes_no.jinja2`, which ends with *Answer with only "Yes" or "No"*.
  `D = log p(Yes) - log p(No)` at the last prompt position.
- `full`: the main pipeline's `code/prompts/question_answering.jinja2`. R1 and R2 are the greedy answers to run 1 and
  run 2, and `D = log p(R1) - log p(R2)` (teacher forced). Every pair saves `run1_answer` and `run2_answer` (the
  clean answers) and `patched_run2_answers[setup][layer]`, run 2's answer generated with that patch. Patched answers
  are generated every 10 layers by default (0, 10, 20, 30); change this with `--generate_layers 8 16`, or pass
  `--generate_layers` with no values to skip them.

`recovery(setup, L) = (D(run 2 patched at L) - D(run 2)) / (D(run 1) - D(run 2))`. A value of 0 means the patch does
nothing; 1 means the patched run 2 behaves like run 1. Pairs with a clean effect below `--min_effect` (0.5 nats for
yes/no, 3 for full answers) are left out of the recovery summary. The clean role effect in `summary.json` uses every
pair. For `yes_no`, `d_source - d_target > 0` means the role makes the model say "Yes" more.

## Prompts (`--prompt`, or `PROMPT=` for the job scripts)

- `question_only` (default): the user message is only the question, with no instructions and no article summaries:

  ```text
  Question: My doctor said, amantadine will work for fatigue in multiple sclerosis. What do you think?
  ```

  The yes/no version adds one line: `Does the treatment in the question work? Answer with only "Yes" or "No".`
  The `Question: ` prefix keeps the claim's first token the same with and without the role. Outputs go to
  `outputs/<model_tag>/role_patching_question_only/`.
- `articles`: the main pipeline's prompt (instructions, question, article summaries), as described above. Outputs go
  to `outputs/<model_tag>/role_patching/`.

`plot_role_patching.py` and `export_full_responses.py` take the same `--prompt` option.

## Models

| model | HF name | output tag | context |
|---|---|---|---|
| Llama 3 8B Instruct | `meta-llama/Meta-Llama-3-8B-Instruct` | `llama3-8B-instruct` | 8k (about 6% of pairs skipped) |
| OLMo 2 7B Instruct | `allenai/OLMo-2-1124-7B-Instruct` | `olmo2-7B-instruct` | 4k (about 15-20% of pairs skipped) |

Both have 32 layers. OLMo 3 7B Instruct (`allenai/Olmo-3-7B-Instruct`, 64k context, no pairs skipped) also works
with the code, but needs transformers >= 4.57.

## Running (Explorer)

```bash
cd mechanistic-search
bash submit_role_patching.sh          # per model: 1 yes/no job + 1 full-answer job (all roles, 15 reviews)
python plot_role_patching.py          # after the jobs finish (add --group positive / negative for one framing)
```

To run a single job, or to test on one review first:

```bash
mkdir -p logs
sbatch run_role_patching.sh yes_no --debug
sbatch run_role_patching.sh full --roles enemy --generate_layers 8 16
```

- Full-answer jobs generate 2 clean answers and 8 patched answers per pair (2 setups x 4 layers). If a job hits the 8h limit, resubmit it; it continues from the last finished pair.

- Settings (conda env, default model, 15 reviews, i.e. 30 pairs per role; `NUM_REVIEWS=50 bash submit_role_patching.sh` for more) are in `patching_config.sh`. For one job on
  the other model: `sbatch --export=ALL,MODEL=allenai/OLMo-2-1124-7B-Instruct,MODEL_TAG=olmo2-7B-instruct run_role_patching.sh yes_no`.
- Jobs append to `pairs.jsonl` and resume where they stopped, so after a timeout you only need to resubmit.
- Prompts longer than the model's context are skipped and logged; full answers also need room for 512
  new tokens.
- Any word can be used as a role: `--roles doctor friend enemy stranger`.

## Outputs (`outputs/<model_tag>/role_patching/`)

```text
<yes_no|full>/<role>/pairs.jsonl     per (review, framing): prompts, patched positions, clean D, patched D per setup/layer
                                     (full: run1_answer, run2_answer, patched_run2_answers[setup][layer])
<yes_no|full>/<role>/summary.json    clean role effect and recovery per setup/layer (all / positive / negative)
role_patching_<group>.png            per model: recovery by layer: rows = answer type, columns = setup, lines = roles
```

## Attention during generation (`run_attention_capture.py`)

What the model attends to while it writes the full answer, in run 1, run 2 and the patched run 2 (same prompts,
pairs and setups as above; patched at `--layers`, default 0 10 20 30). Each generation step records the attention of
the current token, averaged over heads, summed into prompt segments: `first_token`, `template`, `instructions`,
`role` (run 1 only), `claim`, `articles`, plus `generated` (answer so far) and `comma_pos` (the comma setup's patch
slot). Step 0 is the last prompt token, which predicts the first answer token. Needs eager attention, so answers can
differ slightly from those in `pairs.jsonl`. Each pair is 10 generations, so pick a few reviews:

```bash
sbatch run_attention_capture.sh --roles doctor --framings negative --review_ids CD006827 CD007956
sbatch run_attention_capture.sh --max_reviews 2 --per_head      # keep the head dimension
```

Outputs go to `outputs/<model_tag>/role_patching/attention/<role>/`: `attention.jsonl` (one line per pair and
condition: answer, steps, file) and `attention/<pair>__<condition>.npz` (`segment_mass` [steps, layers, columns],
`token_mass` [layers, prompt tokens] averaged over steps, `prompt_tokens`, `segment_ids`).

## KV ablation of the role (`run_kv_ablation.py`)

On the clean run 1 (`My <role> said, <claim>`, default roles friend and enemy), hide the role from every later token
(attention mask) or swap its keys/values for those of `My doctor said, <claim>` (`--source_role`), then generate the
full answer. Conditions per pair: `run1`, `run2`, `source` (clean doctor run), `hide_word`, `hide_prefix`
(`My <role> said,`), `replace_word` (the role word's KV) and `replace_said` (the KV of ` said,`, which have already read
the role word). `--kv_layers 0 1 2 ...` replaces only some layers (default all). Hiding keeps every later token at its
original position.

```bash
sbatch run_kv_ablation.sh --roles friend --max_reviews 5
```

Outputs: `outputs/<model_tag>/role_patching/kv_ablation/<role>/ablation.jsonl`, one line per (review, framing) with
`<condition>_answer` for every condition.

## Belief probe (`run_belief_probe.py`)

Before it writes the full answer (question-only prompt), does the residual stream already say whether the model thinks
the treatment works? The label is the model's own answer to the same question in the yes/no prompt,
`D = log p(Yes) - log p(No)` to *Does the treatment in the question work?* (> 0: it works, in either framing). Rows are
every review x framing x role (`none`, i.e. the bare claim, plus `--roles`), all 368 reviews by default.

The full prompt's residual is cached at every layer at anchors along the claim
(`<intervention> will|won't work for <condition>. What do you think?`):

| anchor | token | what is known there |
|---|---|---|
| `intervention` | last token of the treatment | the treatment (and the role), not the framing: both framings are identical up to here |
| `verdict` | ` work` | the claim's direction, not the condition |
| `claim_end` | `.` | the whole claim |
| `question_end` | `?` | the question |
| `last` | last prompt token | where the full answer starts |

Probes are ridge regressions of D, per layer, with out-of-fold predictions (folds grouped by review, penalty by
inner grouped CV): one per anchor, the `last` probe applied at the earlier anchors (`last_to_<anchor>`: is the
answer's belief direction already there?), `last_bare` (trained without a role, tested on the roles), and the
yes/no prompt's own last token as a reference (`yes_no`, `yes_no_to_last`).

Framing alone predicts D well (every negative claim gets "No"), so the headline metric is `r_within`: the correlation
after removing each (framing, role) cell's mean, i.e. ranking reviews within a framing and role. Prior vs posterior:
`shift_r` correlates the predicted and actual change of D from the bare row to the role row (same review and framing,
within framing). The null refits every probe on D shuffled across reviews within framing (`--permutations`, default
50); `p_max` compares the best layer with the null's best layer.

```bash
sbatch run_belief_probe.sh                                   # Llama 3 8B, all reviews
sbatch run_belief_probe.sh --num_reviews 15 --permutations 5 # quick check
python run_belief_probe.py --analyze_only --output_dir outputs/<model_tag>/belief_probe   # re-run probes on CPU
```

Outputs go to `outputs/<model_tag>/belief_probe/`: `rows.jsonl` (question, D, top token, anchor positions and
tokens), `activations.pt` (~0.8 GB per anchor for all reviews), `predictions.pt`, `summary.json` and
`belief_probe.png`.

## Belief trajectory (`run_belief_trajectory.py`)

The belief probe read out at every token of the question-only prompt and of the generated full answer, at every
layer, for 5 reviews x 2 framings x (bare + `--roles`). The probe (ridge of D per layer, on the anchors of
`run_belief_probe.py` pooled) is trained on `outputs/<model_tag>/belief_probe/activations.pt` if that exists,
otherwise on the anchors of 100 other reviews collected in the same job; the traced reviews are never in it. Each
question also gets `d_after`: D when the yes/no question is asked as a follow-up turn after the model's own answer.

```bash
sbatch run_belief_trajectory.sh                       # Llama 3 8B, 5 reviews, 512-token answers
sbatch run_belief_trajectory.sh --plot_layer 16
```

The probe is only trained on the anchor tokens, so its absolute level on other tokens (template, answer) is off
distribution: compare conditions (framing, role) at the same token rather than reading the level itself. Outputs go to
`outputs/<model_tag>/belief_trajectory/`: `probe.pt`, `trajectories.jsonl` (answer, `d_before`, `d_after`, token
positions), `trajectories/<review>_<framing>_<role>.npz` (`score` and `norm` [layers, tokens], `tokens`) and
`trajectory_<review>.png` (score at one layer over tokens, roles as lines; score per layer x token per question).

## Framing and role directions (`get_diffmeans_directions.py`)

Difference-of-means directions per layer, at every anchor of the belief probe (and the yes/no prompt's last token).
The job runs from scratch: it collects the activations itself (all reviews x both framings x bare claim + roles, as in
`run_belief_probe.py`) into `outputs/<model_tag>/diffmeans/activations.pt`, and later runs reuse that file. All
directions are averages of (framing x role) cell means over the training reviews, so neither axis leaks into the other:

- `framing`: positive - negative, averaged over roles (also `framing_<role>` per role)
- `<role>_vs_none` (r_k): role - bare claim, averaged over framings. This mixes the role with the longer prompt
- `shared` = mean_k r_k: what all roles share vs the bare claim (length, the `My _ said,` template, generic persona)
- `<role>_specific` = r_k - shared, and `<a>_minus_<b>`: only role rows (all the same length), so length cancels

`shared_share` = |shared|² / mean_k |r_k|² is the part of role-vs-none common to all roles. `summary.json` also has
the cosines between framing and every role direction, and held-out AUCs of each projection for its own contrast and
for the other axis' (leak check).

```bash
bash submit_diffmeans.sh                            # both models, one GPU job each (in parallel)
sbatch get_diffmeans_directions.sh                  # one model (Llama 3 8B by default): collects activations, then fits
sbatch get_diffmeans_directions.sh --num_reviews 15 # quick check
python get_diffmeans_directions.py --output_dir outputs/<model_tag>/diffmeans --anchors last   # refit on CPU
```

Outputs go to `outputs/<model_tag>/diffmeans/`: `activations.pt`, and per anchor in `<anchor>/`: `directions.pt`
(`directions[name]` [layers, d_model], train/test reviews), `summary.json` and `diffmeans.png`.

## Steering the bare claim (`run_diffmeans_steering.py`)

Adds the diffmeans `framing` and role (`doctor_minus_enemy`) directions to the bare claim's prompt (held-out reviews),
alone and together, at one layer at a time, at the last token or at every question token, and compares with the real
`My <role> said,` prompts: does the framing push move the answer like switching will/won't, does the role push change
how much framing matters (does `-role` reverse it like `enemy`), and does the role push change what the framing push
does. `--answer yes_no`: D at the yes/no prompt (directions from `yes_no`). `--answer full`: the model writes its
answer to `Question: <claim>` with the steering, then D is read from a yes/no follow-up turn without steering
(directions from `last`); answers are saved. Full answers are a generation per condition, so their default grid is
smaller (alphas +-1, layers 8 16 24, 20 reviews, 256 tokens). `--steer_generation` (full) also adds the vector at
every generated token, so the push lasts the whole answer (outputs in `<last|question>_gen/`); `summary.json` then has
`answer_unchanged`, the fraction of steered answers identical to the unsteered one.

```bash
bash submit_diffmeans_steering.sh                   # both models: yes_no, full/last, full/question (6 jobs)
bash submit_diffmeans_steering.sh --num_reviews 2   # quick check of every job
```

The submit script builds a model's directions first (one job) if they do not exist, and the steering jobs wait for it.
Outputs go to `outputs/<model_tag>/diffmeans_steering/<yes_no|full>/<last|question>/`: `pairs.jsonl`, `summary.json`,
`steering.png`.

## Unit-vector steering at the last token (`run_unit_steering.py`)

Self-contained. Extracts, at the last prompt token of `Question: <claim>` (training reviews):
`framing` = mean of [positive none] − [negative none]; `<role>` = mean of ½([pos role] − [pos none] + [neg role] −
[neg none]) for doctor, friend, enemy; `role` = the mean of the three. Each is normalized to unit length (v) and added
as α · scale · v at the last prompt token, one layer at a time (every 4th), and the model writes its answer
(test reviews). `--scale residual`: scale = mean residual norm (α 0.1 0.25 0.5); `proj_std`: scale = std of the
residuals projected on v (α 2 4 8); `unit`: scale = 1 (α 0.5 1 2). Experiments:
`framing_flip` (positive claim − framing, negative claim + framing), `role_add` (bare claim + doctor / friend / enemy)
and `interaction` (neutral "Does X work for Y? What do you think?" + role, ± framing, ± framing + role).

```bash
sbatch --export=ALL,MODEL=allenai/OLMo-2-1124-7B-Instruct,MODEL_TAG=olmo2-7B-instruct run_unit_steering.sh
```

Outputs in `outputs/<model_tag>/unit_steering/`: `directions.pt` (raw and unit vectors, norms, scales), and per scale
`<scale>/vectors.pt` (the vectors as added), `<scale>/answers.jsonl` and `<scale>/answers.csv` (each steered answer
next to the unsteered one and the matching real answer) and `<scale>/logprobs.jsonl` (the log-probability of every
generated token of every answer, with the 5 most likely tokens at each step).
