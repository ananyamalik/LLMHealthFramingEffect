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

## Models

| model | HF name | output tag | context |
|---|---|---|---|
| Llama 3 8B Instruct | `meta-llama/Meta-Llama-3-8B-Instruct` | `llama3-8B-instruct` | 8k (about 6% of pairs skipped) |
| OLMo 3 7B Instruct | `allenai/Olmo-3-7B-Instruct` | `olmo3-7B-instruct` | 64k (no pairs skipped) |

Both have 32 layers. OLMo 3 needs transformers >= 4.57. OLMo 2 7B Instruct (`allenai/OLMo-2-1124-7B-Instruct`) also
works with the code, but its 4k context skips about 15-20% of pairs.

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
  the other model: `sbatch --export=ALL,MODEL=allenai/Olmo-3-7B-Instruct,MODEL_TAG=olmo3-7B-instruct run_role_patching.sh yes_no`.
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
