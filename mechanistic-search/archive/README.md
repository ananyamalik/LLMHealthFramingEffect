# Mechanistic search: last-token activation patching (positive → negative framing)

Does the framing effect pass through the residual stream at the last token before the answer starts?

For each (review, question type) pair, the positive and negative prompts are built exactly as in
`code/generate_framing_responses.py` (`question_answering` template + Llama 3 chat template). Both end in the same
chat-template tokens (`<|start_header_id|>assistant<|end_header_id|>\n\n`). We cache the positive run's residual stream
(output of each decoder layer) at that last position and write it into the negative run at one layer L.
Multiturn questions are skipped.

## Stages

1. `run_layer_sweep.py`: greedy clean responses R+ (positive) and R- (negative), then for each layer L a teacher-forced pass
   with the patch applied:
   - `recovery(L) = (D_patched(L) - D_neg) / (D_pos - D_neg)`, where `D = log p(R+ | prompt) - log p(R- | prompt)`.
     0 means the patch does nothing. 1 means the patched negative run prefers R+ over R- as much as the positive run does.
   - `first_token_restoration(L) = 1 - KL(p_pos || p_patched) / KL(p_pos || p_neg)` on the first generated token.
   - The last token is the same in both prompts, so any difference at that position comes from attention to earlier
     tokens, already from layer 0 on. At the last layer, first-token restoration is exactly 1 by construction. Recovery
     may still be low there, because later response tokens can read the framing straight from the question tokens.
2. `run_patched_generation.py`: greedy generation from the negative prompt with the patch at the chosen layers
   (`--layers auto` takes the top-k layers from stage 1). The outputs use the same format as
   `code/outputs/responses/<model>/question_responses.json`.
3. `plot_layer_sweep.py`: makes the recovery-by-layer plots.

## Running

```bash
cd mechanistic-search
mkdir -p logs
jid=$(sbatch --parsable run_layer_sweep.sh)                     # job 1: layer sweep + plots
sbatch --dependency=afterok:$jid run_patched_generation.sh      # job 2: starts only if job 1 succeeds
```

Shared settings (conda env, model, question set, number of reviews, contrast) live in `patching_config.sh`.

### Doctor-attribution contrast

Same two stages, but the source prompt is `"My doctor said <sentence>"` and the target is the bare `"<sentence>"`.
`<sentence>` is each claim-style question (`doctor`, `ai`, `testimonials`, `journals`, `friend`, `family`) with its
attribution removed, in both framings, e.g. `"AI said that X works well for Y. Is this true?"` gives
`"My doctor said X works well for Y. Is this true?"` and `"X works well for Y. Is this true?"`. Question keys are
`<type>_<positive|negative>`. `ai`, `testimonials` and `journals` give the same sentence (as do `friend` and `family`),
so only `doctor_*`, `ai_*` and `friend_*` are kept: 6 pairs per review. In the outputs, `positive_*` is the doctor run and `negative_*` the bare run.
Only the `extracted` question set has claim-style questions.

```bash
bash run_doctor_patching.sh    # submits both jobs with CONTRAST=doctor; outputs in outputs/<model_tag>/extracted/doctor/
```

- You need access to the gated `meta-llama/Meta-Llama-3-8B-Instruct` model, and `HUGGINGFACE_TOKEN` must be in the repo `.env`.
- Both stages append to `.jsonl` files and resume where they stopped, so after a timeout you only need to resubmit.
- To test quickly, add `--debug` to both python calls (2 reviews, then 4 pairs).
- Llama 3 8B has an 8k context window. Pairs where prompt + `max_new_tokens` is longer than that are skipped and logged.
  `meta-llama/Llama-3.1-8B-Instruct` (128k context) can be used instead through `MODEL`.

## Outputs (`outputs/<model_tag>/<question_set>/`)

```text
sweep/layer_sweep_pairs.jsonl       per pair: clean responses + per-layer log-probs / KL
sweep/layer_sweep_summary.json      per-layer mean/median/SEM (overall and by question type), layers ranked by recovery
sweep/layer_sweep*.png
generation/patched_generations.jsonl
generation/responses/clean_question_responses.json
generation/responses/patched_layer{L}_question_responses.json   negative_answer = patched, clean_negative_answer kept
```

To check whether patching changes the evidence direction, run the existing evaluation on the clean file and on each
patched response file:

```bash
cd code
python3 -u run_evaluation.py \
    --file_path ../mechanistic-search/outputs/llama3-8B-instruct/extracted/generation/responses/patched_layer16_question_responses.json \
    --output_path ../mechanistic-search/outputs/llama3-8B-instruct/extracted/evaluation/patched_layer16_eval_results.json \
    --eval_path outputs/questions/qwen3_thinking-4B/extracted/evidence_direction_questions_final.json \
    --data_type framing
```

### Doctor comma patching (`run_comma_patching.py`)

Source `"My doctor said, <xyz>"`, target `"<xyz>"` (same claims as the doctor contrast; the claim tokens and everything
after them are checked to be identical). The source's residual stream at the comma token is written into the target at
layer L at one position, in two separate setups: `before` (the token right before `<xyz>`, end of `**QUESTION**:`) and
`after` (the token right after `<xyz>`). Recovery is computed as in the layer sweep. Defaults: layers 0, 5, ..., 30 and
the first 15 reviews of the main sweeps' 30-review sample (90 pairs).

```bash
sbatch run_comma_patching.sh    # outputs in outputs/<model_tag>/extracted/doctor/comma/
```

Each record logs the patched tokens (`positions.before_token`, `positions.after_token`) and whether the claim's last
token is merged with the next character (`last_claim_token_straddles`).

### Span patching (`run_span_patching.py`): positive control and doctor vs friend

The two prompts are split into a shared prefix, the differing tokens and a shared suffix. The source run's residual
stream is written into the target at layer L in two setups: `diff` (the differing tokens, only when both sides have the
same number of tokens) and `after` (every position of the shared suffix, aligned from the end).

- `--contrast framing`: positive -> negative question. Positive control: `diff` at early layers should recover most of
  the effect; `after` shows how much of the framing has moved past the framing words by layer L.
- `--contrast doctor_friend`: `"My doctor said, <xyz>"` -> `"My friend said, <xyz>"` (same claims as the comma run).
- `--contrast doctor_masked`: `"My doctor said, <xyz>"` -> the same tokens with `My doctor said,` hidden by the attention
  mask and the position ids shifted past it. This is the bare `"<xyz>"` prompt at the doctor prompt's positions, with no
  replacement word; each pair also scores the real bare prompt, and the summary reports the gap
  (`masked_vs_bare_gap_relative_to_effect`, should be ~0). Setups: `claim` (the `<xyz>` tokens) and `after`.

Pairs with a clean effect below `--min_effect` (default 3) are left out of the summary.

```bash
sbatch --job-name=patch_span_framing run_span_patching.sh framing          # outputs/<model_tag>/extracted/span/
sbatch --job-name=patch_span_docfriend run_span_patching.sh doctor_friend  # outputs/<model_tag>/extracted/doctor_friend/span/
sbatch --job-name=patch_span_docmask run_span_patching.sh doctor_masked      # outputs/<model_tag>/extracted/doctor_masked/span/
```

### Attention ablation (`run_attention_ablation.py`): is the doctor prefix read directly?

Necessity test complementing the patching (sufficiency) runs. In the `"My doctor said, <xyz>"` run, attention to the
prefix tokens is blocked at every layer for: `answer_blocked` (the last prompt token and all answer tokens),
`prompt_blocked` (the prompt tokens after the prefix, except the last), `all_blocked` (both), and `none` (sanity check:
must reproduce the clean doctor run; the script stops if it does not). Reported as the fraction of the doctor effect
remaining, `(D_ablated - D_bare) / (D_doctor - D_bare)`: 1 = intact, 0 = bare prompt. 30 reviews, 180 pairs.

```bash
sbatch --gres=gpu:a100:1 run_attention_ablation.sh    # outputs/<model_tag>/extracted/doctor/attention_ablation/
```

### Yes/No readout (`run_yes_no_patching.py`)

Checks whether the last-token null comes from the long-form metric. The prompt
(`prompts/question_answering_yes_no.jinja2`) keeps the framed question and article summaries and ends with one fixed
follow-up: *Based on the article summaries, does the treatment in the QUESTION work? Answer with only "Yes" or "No".*
The readout is `log p(Yes) - log p(No)` at the last prompt position (no generation). Setups: `last` (last prompt token)
and `diff` (differing tokens, when both sides have the same length), at every layer. Contrasts: `framing`, `doctor`,
`doctor_friend`. Expected if the long-form metric was the issue: `last` near 0 at early layers, rising to ~1 in middle
and late layers (it is 1 at the final layer by construction).

```bash
sbatch --gres=gpu:a100:1 --job-name=yesno_framing run_yes_no_patching.sh framing   # outputs/<model_tag>/extracted/yes_no/
sbatch --gres=gpu:a100:1 --job-name=yesno_doctor  run_yes_no_patching.sh doctor    # .../extracted/doctor/yes_no/
sbatch --gres=gpu:a100:1 --job-name=yesno_docfr   run_yes_no_patching.sh doctor_friend
```

### Credibility direction and steering (`run_credibility_direction.py`)

Is there one source-token direction that tracks how much the model agrees with a source, and does adding it cause
agreement to rise steadily? Uses the Yes/No prompt with the claim attributed to one of 8 sources (`my doctor`,
`my pharmacist`, `my nurse`, `my friend`, `my neighbor`, `a stranger`, `a website`, `an AI`) in two frames
(`"<Source> said, <claim>"`, `"According to <source>, <claim>"`), plus the bare claim. The claims are the doctor
contrast's (3 per review, each "will work" and "won't work"). Every source is a single token at the same position, and
the "said" doctor/friend prompts are exactly the `doctor_friend` Yes/No prompts.

Agreement is signed so it means "agrees with the source" in both polarities: `A = log p(Yes) - log p(No)` for
"will work" claims and its negative for "won't work" claims. A source's effect is `A(source) - A(bare claim)`.

1. `behavior`: the effect of each source (by frame and polarity), and the residual stream at the source token at every
   layer. The source token comes before the claim, so its activation is the same for every claim (checked and logged).
2. `direction` (CPU, no model): per layer, with activations centered over sources, `doctor_minus_friend`, `pls_all`
   (behavior-weighted mean difference), `pls_heldout` (the same without doctor and friend) and `pc1` (no behavior used).
   Each is scored by Spearman rho between the sources' projections and their behavioral effect: over all sources,
   without doctor (it should not just be "doctor vs everything else"), on sources not used to fit it, leave one source
   out for `pls_all` (permutation p), and across frames. `credibility_coordinate` puts sources on a friend = 0,
   doctor = 1 scale; `a stranger` is the neutral reference.
3. `steer`: adds `alpha * gap_L * v_L` at the friend prompt's source token at layer L (default layers 2, 4, 6, 8, where
   the source word is still causal), with `gap_L` the doctor-friend gap along `v_L`, so `alpha = 1` moves friend's
   projection to doctor's. Default direction `pls_heldout`, so neither endpoint is used to fit it, with 2 matched-norm
   random directions as controls. Evidence for a causal, claim-independent credibility representation: agreement rises
   monotonically with alpha for both polarities (`monotonic_increasing`, `mean_slope_per_alpha`,
   `fraction_pairs_positive_slope` in the summary) and stays flat for the random directions. The dotted lines in the
   plot mark the real doctor-friend gap.

```bash
sbatch --gres=gpu:a100:1 run_credibility_direction.sh             # outputs/<model_tag>/extracted/credibility/
sbatch --gres=gpu:a100:1 run_credibility_direction.sh --determiner indefinite \
    --output_dir ./outputs/llama3-8B-instruct/extracted/credibility_indefinite    # "a doctor", "a friend": no "my" vs "a" confound
```

Other options: `--steer_directions doctor_minus_friend pls_all`, `--steer_span source_to_end` (add at every token from
the source on), `--steer_sources friend stranger`, `--alphas`, `--steer_layers`, `--stages direction` (reanalyze only).

### Conflicting doctor claims (`run_conflict_patching.py`)

Two doctor claims with opposite polarity in one question, same Yes/No prompt and readout as above:
*"My doctor said X will work for Y, but my doctor also said X won't work for Y. What do you think?"* (and the reverse
order; the claims come from the `doctor` and `ai` questions with the attribution and follow-up removed). Each record
also scores `++`, `--` (same connective) and the single claims `+` and `-`.

- Behaviour: `negative_weight = (D(conflict) - D(+)) / (D(-) - D(+))`, 0 = the positive claim wins, 1 = the negative
  claim wins. The mean over the two orders measures the negativity preference; `weight(+-) - weight(-+)` measures
  recency (> 0 = the later claim wins).
- Patching: for each conflict prompt and each claim, the source is the same prompt with that claim flipped (`+-`, claim
  1 -> `--`). Setups `last`, `after` (every token after the flipped words) and `diff` (only when the flipped words have
  the same number of tokens). Shows when each claim's polarity reaches the answer position, and `abs_slot_effect` shows
  how much each claim moves the answer at all.

```bash
sbatch --gres=gpu:a100:1 run_conflict_patching.sh    # outputs/<model_tag>/extracted/doctor_conflict/yes_no/
```

`--connective ", but my friend said"` gives a doctor-vs-friend conflict with the same analysis.

### Doctor -> friend variants: OLMo and lawyer (`run_olmo_lawyer_patching.sh`)

Two variants of the `doctor_friend` span and Yes/No runs (same claims, reviews and setups):

1. **OLMo**: `doctor_friend` with `allenai/OLMo-2-0425-1B-Instruct` (16 layers, every layer patched; 4k context, so
   long prompts are skipped and logged). Outputs in `outputs/olmo2-1B-instruct/extracted/doctor_friend/{span,yes_no}/`.
2. **Lawyer**: `--contrast doctor_lawyer`, `"My doctor said, <xyz>"` -> `"My lawyer said, <xyz>"`, with Llama 3 8B.
   Outputs in `outputs/llama3-8B-instruct/extracted/doctor_lawyer/{span,yes_no}/`.

```bash
bash run_olmo_lawyer_patching.sh    # submits 4 jobs: OLMo span + yes/no, lawyer span + yes/no
```

The model can be changed for any job through `patching_config.sh`: `sbatch --export=ALL,MODEL=<hf name>,MODEL_TAG=<tag> ...`.
