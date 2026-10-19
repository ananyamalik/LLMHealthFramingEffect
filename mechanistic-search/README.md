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
