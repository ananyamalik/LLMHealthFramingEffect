"""
Doctor-comma patching.

Source prompt: "My doctor said, <xyz>"     target prompt: "<xyz>"
where <xyz> is a claim-style question with its attribution removed (same pairs as --contrast doctor in
run_layer_sweep.py), so the claim text and its tokens are identical in both prompts.

We cache the source run's residual stream at the comma token of "My doctor said," and write it into the target run at
layer L at one position, in two separate setups:
    before: the token right before <xyz> in the target (end of "**QUESTION**:")
    after:  the token right after <xyz> in the target (start of the line before "**ARTICLE SUMMARIES**")
The patch overwrites that token's residual stream; the target prompt itself is unchanged.

Metric (teacher forced, same as run_layer_sweep.py):
    R+ = greedy response to the source prompt, R- = greedy response to the target prompt
    D(prompt) = log p(R+ | prompt) - log p(R- | prompt)
    recovery(setup, L) = (D(target, patched) - D(target)) / (D(source) - D(target))

Outputs (in --output_dir):
    comma_patching_pairs.jsonl     one line per pair (clean responses, patched positions, per-setup per-layer log-probs)
    comma_patching_summary.json    per-setup per-layer aggregates (overall and by positive/negative claim)
    comma_patching.png
"""
import argparse
import json
import os
import gc

import torch
from tqdm import tqdm

from patching_utils import (
    DEFAULT_MODEL_NAME, SEED, DOCTOR_PREFIX, PROMPT_TEMPLATE_NAME, render_prompt, set_global_seed,
    load_model_and_tokenizer, get_decoder_layers, get_context_length, format_rct_inputs, build_prompt_ids,
    load_dataset, sample_reviews, get_prompt_pairs, cache_residuals, patch_residual, greedy_generate,
    response_logprobs,
)
from run_layer_sweep import MIN_EFFECT_SIZE, pair_key, load_done_pairs, aggregate

PAIRS_FILE_NAME = "comma_patching_pairs.jsonl"
SUMMARY_FILE_NAME = "comma_patching_summary.json"
SETUPS = ["before", "after"]
QUESTION_MARKER = "**QUESTION**: "


def encode_prompt(tokenizer, question: str, rct_inputs: list[dict], device) -> tuple[torch.Tensor, object, int]:
    """
    Prompt ids plus the encoding with character offsets, and the character index where the question starts.
    The ids are checked against build_prompt_ids so positions match the runs exactly.
    """
    input_text = render_prompt(PROMPT_TEMPLATE_NAME, template_dir="./prompts", question=question, abstracts=rct_inputs)
    text = tokenizer.apply_chat_template([{"role": "user", "content": input_text}], tokenize=False, add_generation_prompt=True)
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)

    input_ids = build_prompt_ids(tokenizer, question, rct_inputs, device)
    if encoding["input_ids"] != input_ids[0].tolist():
        raise ValueError("Offset tokenization does not match the chat-template ids")

    marker_index = text.find(QUESTION_MARKER + question)
    if marker_index == -1:
        raise ValueError(f"Question not found in the rendered prompt: {question}")
    return input_ids, encoding, marker_index + len(QUESTION_MARKER)


def claim_token_span(encoding, claim_start: int, claim_end: int) -> tuple[int, int]:
    """(first, last) token index of the claim characters [claim_start, claim_end)."""
    return encoding.char_to_token(claim_start), encoding.char_to_token(claim_end - 1)


def find_positions(tokenizer, src_ids, src_enc, src_q_start, tgt_ids, tgt_enc, tgt_q_start, claim: str) -> dict:
    """Comma position in the source, and the before/after positions around the claim in the target."""
    comma_char = src_q_start + len(DOCTOR_PREFIX)
    comma_pos = src_enc.char_to_token(comma_char)
    if tokenizer.decode([src_ids[0, comma_pos].item()]) != ",":
        raise ValueError(f"Comma is not its own token: {tokenizer.decode([src_ids[0, comma_pos].item()])!r}")

    src_first, src_last = claim_token_span(src_enc, comma_char + 2, comma_char + 2 + len(claim))
    tgt_first, tgt_last = claim_token_span(tgt_enc, tgt_q_start, tgt_q_start + len(claim))
    if src_first != comma_pos + 1:
        raise ValueError("Claim does not start right after the comma token")
    # Same claim tokens and same suffix (articles + chat template) in both prompts
    if not torch.equal(src_ids[0, src_first:], tgt_ids[0, tgt_first:]):
        raise ValueError(f"Claim/suffix tokens differ between source and target for: {claim}")

    before_pos, after_pos = tgt_first - 1, tgt_last + 1
    claim_end_char = tgt_q_start + len(claim)
    return {
        "comma_pos": comma_pos,
        "before_pos": before_pos,
        "after_pos": after_pos,
        "before_token": tokenizer.decode([tgt_ids[0, before_pos].item()]),
        "after_token": tokenizer.decode([tgt_ids[0, after_pos].item()]),
        # True if the last claim token also holds characters after the claim (e.g. "?\n" merged into one token)
        "last_claim_token_straddles": tgt_enc["offset_mapping"][tgt_last][1] > claim_end_char,
    }


@torch.no_grad()
def get_comma_cache(model, src_ids, comma_pos: int) -> dict[int, torch.Tensor]:
    with cache_residuals(model, [comma_pos]) as cache:
        model.model(input_ids=src_ids, use_cache=False)
    return dict(cache)


def score_pair(model, src_ids, tgt_ids, src_resp, tgt_resp, positions: dict, layers: list[int]) -> dict:
    comma_cache = get_comma_cache(model, src_ids, positions["comma_pos"])

    def logps(prompt_ids):
        return {
            "logp_pos_resp": response_logprobs(model, prompt_ids, src_resp)[0].sum().item(),
            "logp_neg_resp": response_logprobs(model, prompt_ids, tgt_resp)[0].sum().item(),
        }

    result = {"clean": {"source_prompt": logps(src_ids), "target_prompt": logps(tgt_ids)}, "patched": {}}
    for setup in SETUPS:
        result["patched"][setup] = {}
        for layer_idx in layers:
            with patch_residual(model, layer_idx, [positions[f"{setup}_pos"]], comma_cache[layer_idx]):
                result["patched"][setup][str(layer_idx)] = logps(tgt_ids)
    return result


def compute_pair_recovery(record: dict) -> dict | None:
    """{setup: {layer: recovery}} for one pair; None if the pair has no effect to recover."""
    clean = record["metrics"]["clean"]
    d_src = clean["source_prompt"]["logp_pos_resp"] - clean["source_prompt"]["logp_neg_resp"]
    d_tgt = clean["target_prompt"]["logp_pos_resp"] - clean["target_prompt"]["logp_neg_resp"]
    effect = d_src - d_tgt
    if record["identical_responses"] or abs(effect) < MIN_EFFECT_SIZE:
        return None
    return {
        setup: {layer: ((m["logp_pos_resp"] - m["logp_neg_resp"]) - d_tgt) / effect for layer, m in by_layer.items()}
        for setup, by_layer in record["metrics"]["patched"].items()
    }


def summarize(pairs_path: str, summary_path: str) -> dict:
    with open(pairs_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    groups = {"all": {}, "positive": {}, "negative": {}}   # group -> setup -> layer -> values
    num_valid = 0
    for record in records:
        recovery = compute_pair_recovery(record)
        if recovery is None:
            continue
        num_valid += 1
        framing = record["question_key"].rsplit("_", 1)[1]
        for setup, by_layer in recovery.items():
            for layer, value in by_layer.items():
                for group in ["all", framing]:
                    groups[group].setdefault(setup, {}).setdefault(layer, []).append(value)

    summary = {
        "num_pairs": len(records),
        "num_pairs_with_effect": num_valid,
        "num_pairs_after_token_straddles": sum(r["positions"]["last_claim_token_straddles"] for r in records),
        "by_setup": {
            group: {
                setup: {l: aggregate(v) for l, v in sorted(by_layer.items(), key=lambda kv: int(kv[0]))}
                for setup, by_layer in by_setup.items()
            }
            for group, by_setup in groups.items()
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    return summary


def plot(summary: dict, output_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, group in zip(axes, ["all", "positive", "negative"]):
        for setup, by_layer in summary["by_setup"][group].items():
            layers = [int(l) for l in by_layer]
            ax.errorbar(layers, [by_layer[str(l)]["mean"] for l in layers], yerr=[by_layer[str(l)]["sem"] for l in layers],
                        marker="o", capsize=3, label=f"{setup} <xyz>")
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set_xlabel("Layer patched")
        ax.set_title(f"{group} claims")
        ax.legend()
    axes[0].set_ylabel("Mean recovery +/- SEM")
    fig.suptitle(f"Comma of 'My doctor said,' -> '<xyz>' (n={summary['num_pairs_with_effect']} pairs)")
    fig.tight_layout()
    path = os.path.join(output_dir, "comma_patching.png")
    fig.savefig(path, dpi=200)
    print(f"Saved {path}")


def main(args):
    set_global_seed(SEED)
    os.makedirs(args.output_dir, exist_ok=True)
    pairs_path = os.path.join(args.output_dir, PAIRS_FILE_NAME)
    summary_path = os.path.join(args.output_dir, SUMMARY_FILE_NAME)

    # First reviews of the same seeded sample as the main sweeps, so the reviews overlap with those runs
    dataset = sample_reviews(load_dataset(args.input_path), args.sample_pool, seed=args.seed)[:args.num_reviews]
    if args.debug:
        dataset = dataset[:1]
    print(f"Number of reviews: {len(dataset)}")

    model, tokenizer = load_model_and_tokenizer(args.model)
    layers = list(range(0, len(get_decoder_layers(model)), args.layer_stride))
    context_length = get_context_length(model)
    print(f"Patching layers: {layers}")

    done = load_done_pairs(pairs_path)
    print(f"Pairs already done (resuming): {len(done)}")

    with open(pairs_path, "a", encoding="utf-8") as out_file:
        for example in tqdm(dataset, desc="Reviews"):
            review_id = example["ReviewID"]
            rct_inputs = format_rct_inputs(example["Inputs"])

            for prompt_pair in get_prompt_pairs(example, "doctor", None):
                key = pair_key(review_id, prompt_pair["question_key"])
                if key in done:
                    continue
                claim = prompt_pair["target_question"]
                src_question = f"{DOCTOR_PREFIX}, {claim}"

                src_ids, src_enc, src_q_start = encode_prompt(tokenizer, src_question, rct_inputs, model.device)
                tgt_ids, tgt_enc, tgt_q_start = encode_prompt(tokenizer, claim, rct_inputs, model.device)
                if src_ids.shape[1] + args.max_new_tokens > context_length:
                    print(f"Skipping {key}: prompt + max_new_tokens exceeds context length ({context_length})")
                    continue
                positions = find_positions(tokenizer, src_ids, src_enc, src_q_start, tgt_ids, tgt_enc, tgt_q_start, claim)

                src_resp = greedy_generate(model, tokenizer, src_ids, args.max_new_tokens)
                tgt_resp = greedy_generate(model, tokenizer, tgt_ids, args.max_new_tokens)
                metrics = score_pair(model, src_ids, tgt_ids, src_resp, tgt_resp, positions, layers)

                record = {
                    "pair_key": key,
                    "review_id": review_id,
                    "question_key": prompt_pair["question_key"],
                    "source_question": src_question,
                    "target_question": claim,
                    "source_answer": tokenizer.decode(src_resp, skip_special_tokens=True).strip(),
                    "target_answer": tokenizer.decode(tgt_resp, skip_special_tokens=True).strip(),
                    "identical_responses": torch.equal(src_resp, tgt_resp),
                    "positions": positions,
                    "metrics": metrics,
                }
                out_file.write(json.dumps(record) + "\n")
                out_file.flush()

            gc.collect()
            torch.cuda.empty_cache()

    summary = summarize(pairs_path, summary_path)
    print(f"Pairs: {summary['num_pairs']} (with effect: {summary['num_pairs_with_effect']}; "
          f"last claim token merged with the next character: {summary['num_pairs_after_token_straddles']})")
    for setup, by_layer in summary["by_setup"]["all"].items():
        print(f"Mean recovery, comma written {setup} <xyz>:")
        for layer, r in by_layer.items():
            print(f"  layer {int(layer):>2}: {r['mean']:.3f} +/- {r['sem']:.3f} (n={r['n']})")
    plot(summary, args.output_dir)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch the comma of 'My doctor said,' into the bare claim, before or after it")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="HuggingFace model name.")
    parser.add_argument("--input_path", required=True, help="dataset with RCT inputs and framed questions (extracted question set).")
    parser.add_argument("--output_dir", required=True, help="directory to save the outputs.")
    parser.add_argument("--num_reviews", default=15, type=int, help="number of reviews (each gives 6 pairs).")
    parser.add_argument("--sample_pool", default=30, type=int, help="reviews are the first --num_reviews of the seeded sample of this size used by the main sweeps.")
    parser.add_argument("--max_new_tokens", default=512, type=int, help="max tokens for the clean greedy responses.")
    parser.add_argument("--layer_stride", default=5, type=int, help="patch every k-th layer starting at 0.")
    parser.add_argument("--seed", default=SEED, type=int, help="seed for sampling reviews.")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, help="only run 1 review.")
    args = parser.parse_args()

    print()
    print("Arguments Provided for Comma Patching:")
    for k, v in vars(args).items():
        print(f"{k:<18} {v}")
    print()

    main(args)
