"""
Span patching: patch the tokens that differ between two prompts, or everything after them.

Contrasts (--contrast):
    framing        source = positive question, target = negative question (all single-turn types). Positive control:
                   patching the differing words at early layers should recover most of the framing effect.
    doctor_friend  source = "My doctor said, <xyz>", target = "My friend said, <xyz>", with the same claims as
                   run_comma_patching.py.
    doctor_masked  source = "My doctor said, <xyz>", target = the same tokens with "My doctor said," hidden by the
                   attention mask and the position ids shifted past it, which is the bare "<xyz>" prompt laid out at
                   the source's positions (checked against the real bare prompt for every pair). Setups:
                       claim: the <xyz> tokens     after: every position after <xyz>

For each pair, the token ids are split into a shared prefix, a differing span and a shared suffix. The source run's
residual stream is written into the target run at layer L, in two separate setups:
    diff:   the differing span, position by position (only when both spans have the same number of tokens)
    after:  every position of the shared suffix (rest of the question, article summaries, chat template), aligned from
            the end, so it is always possible

Metric (teacher forced, same as run_layer_sweep.py):
    R+ = greedy response to the source prompt, R- = greedy response to the target prompt (the bare prompt for
    doctor_masked)
    D(prompt) = log p(R+ | prompt) - log p(R- | prompt)
    recovery(setup, L) = (D(target, patched) - D(target)) / (D(source) - D(target))

Outputs (in --output_dir):
    span_patching_pairs.jsonl    one line per pair (clean responses, spans, per-setup per-layer log-probs)
    span_patching_summary.json   per-setup per-layer aggregates (overall and by group)
    span_patching.png
"""
import argparse
import json
import os
import gc

import torch
from tqdm import tqdm

from patching_utils import (
    DEFAULT_MODEL_NAME, SEED, DOCTOR_PREFIX, set_global_seed, load_model_and_tokenizer, get_decoder_layers,
    get_context_length, format_rct_inputs, build_prompt_ids, load_dataset, sample_reviews, get_prompt_pairs,
    cache_residuals, patch_residual, greedy_generate, response_logprobs, masked_prompt_inputs,
)
from run_layer_sweep import pair_key, load_done_pairs, aggregate
from run_comma_patching import encode_prompt, claim_token_span

PAIRS_FILE_NAME = "span_patching_pairs.jsonl"
SUMMARY_FILE_NAME = "span_patching_summary.json"
CONTRASTS = ["framing", "doctor_friend", "doctor_masked"]


def get_span_prompt_pairs(example: dict, contrast: str) -> list[dict]:
    """[{question_key, source_question, target_question}] for one review."""
    if contrast == "framing":
        return get_prompt_pairs(example, "framing", None)
    if contrast == "doctor_friend":
        return [
            {
                "question_key": p["question_key"],
                "source_question": f"My doctor said, {p['target_question']}",
                "target_question": f"My friend said, {p['target_question']}",
            }
            for p in get_prompt_pairs(example, "doctor", None)
        ]
    if contrast == "doctor_masked":
        return [
            {
                "question_key": p["question_key"],
                "source_question": f"{DOCTOR_PREFIX}, {p['target_question']}",
                "target_question": p["target_question"],
            }
            for p in get_prompt_pairs(example, "doctor", None)
        ]
    raise ValueError(f"Unknown contrast '{contrast}' (options: {CONTRASTS})")


def get_group(contrast: str, question_key: str) -> str:
    """Summary group besides 'all': question type for framing, positive/negative claim for doctor_friend."""
    return question_key if contrast == "framing" else question_key.rsplit("_", 1)[1]


def find_spans(tokenizer, src_ids: torch.Tensor, tgt_ids: torch.Tensor) -> dict:
    """Shared prefix / differing span / shared suffix of the two prompts."""
    src, tgt = src_ids[0].tolist(), tgt_ids[0].tolist()
    prefix = 0
    while prefix < min(len(src), len(tgt)) and src[prefix] == tgt[prefix]:
        prefix += 1
    suffix = 0
    while suffix < min(len(src), len(tgt)) - prefix and src[-1 - suffix] == tgt[-1 - suffix]:
        suffix += 1
    if suffix == 0:
        raise ValueError("Prompts share no suffix; the 'after' setup needs the article summaries + chat template")

    src_span, tgt_span = (prefix, len(src) - suffix), (prefix, len(tgt) - suffix)
    return {
        "src_span": src_span,
        "tgt_span": tgt_span,
        "src_span_text": tokenizer.decode(src[src_span[0]:src_span[1]]),
        "tgt_span_text": tokenizer.decode(tgt[tgt_span[0]:tgt_span[1]]),
        "diff_patchable": src_span[1] - src_span[0] == tgt_span[1] - tgt_span[0],
        "num_after_positions": suffix,
    }


def setup_positions(spans: dict) -> dict[str, tuple[list[int], list[int]] | None]:
    """{setup: (source positions, target positions)}, or None if the setup is not possible for this pair."""
    (s0, s1), (t0, t1) = spans["src_span"], spans["tgt_span"]
    n_after = spans["num_after_positions"]
    return {
        "diff": (list(range(s0, s1)), list(range(t0, t1))) if spans["diff_patchable"] else None,
        "after": (list(range(s1, s1 + n_after)), list(range(t1, t1 + n_after))),
    }


def prepare_masked_pair(tokenizer, src_ids, bare_ids, src_question: str, claim: str, rct_inputs, device) -> dict:
    """
    Positions for doctor_masked: the prefix tokens to hide, the claim tokens and everything after them (same indices in
    the source and the masked run, since both use the source's tokens).
    """
    spans = find_spans(tokenizer, src_ids, bare_ids)
    if spans["tgt_span"][0] != spans["tgt_span"][1]:
        raise ValueError(f"Bare prompt is not the doctor prompt minus its prefix: {spans['tgt_span_text']!r}")
    prefix_positions = list(range(*spans["src_span"]))

    _, src_enc, src_q_start = encode_prompt(tokenizer, src_question, rct_inputs, device)
    claim_start = src_q_start + len(f"{DOCTOR_PREFIX}, ")
    first, last = claim_token_span(src_enc, claim_start, claim_start + len(claim))
    if first != spans["src_span"][1]:
        raise ValueError("Claim does not start right after the masked prefix")

    attention_mask, position_ids = masked_prompt_inputs(src_ids, prefix_positions)
    claim_positions = list(range(first, last + 1))
    after_positions = list(range(last + 1, src_ids.shape[1]))
    return {
        "spans": {
            "masked_prefix_text": spans["src_span_text"],
            "masked_positions": prefix_positions,
            "claim_span": (first, last + 1),
            "num_after_positions": len(after_positions),
        },
        "positions": {"claim": (claim_positions, claim_positions), "after": (after_positions, after_positions)},
        "target_kwargs": {"prompt_attention_mask": attention_mask, "prompt_position_ids": position_ids},
    }


@torch.no_grad()
def get_source_cache(model, src_ids: torch.Tensor, positions: list[int], layers: list[int]) -> dict[int, torch.Tensor]:
    """Only the patched layers, kept on CPU (the 'after' setup covers ~1.8k positions); patch_residual moves them back."""
    with cache_residuals(model, positions, layer_indices=layers, device="cpu") as cache:
        model.model(input_ids=src_ids, use_cache=False)
    return dict(cache)


def score_pair(model, src_ids, tgt_ids, src_resp, tgt_resp, positions: dict, layers: list[int],
               target_kwargs: dict | None = None, bare_ids: torch.Tensor | None = None) -> dict:
    """
    target_kwargs (attention mask / position ids) apply to every target pass. With bare_ids, the real bare prompt is
    also scored, to check that the masked target matches it.
    """
    target_kwargs = target_kwargs or {}

    def logps(prompt_ids, **kwargs):
        return {
            "logp_pos_resp": response_logprobs(model, prompt_ids, src_resp, **kwargs)[0].sum().item(),
            "logp_neg_resp": response_logprobs(model, prompt_ids, tgt_resp, **kwargs)[0].sum().item(),
        }

    result = {"clean": {"source_prompt": logps(src_ids), "target_prompt": logps(tgt_ids, **target_kwargs)}, "patched": {}}
    if bare_ids is not None:
        result["clean"]["bare_prompt"] = logps(bare_ids)
    for setup, setup_pos in positions.items():
        if setup_pos is None:
            continue
        src_pos, tgt_pos = setup_pos
        cache = get_source_cache(model, src_ids, src_pos, layers)
        result["patched"][setup] = {}
        for layer_idx in layers:
            with patch_residual(model, layer_idx, tgt_pos, cache[layer_idx]):
                result["patched"][setup][str(layer_idx)] = logps(tgt_ids, **target_kwargs)
        del cache
    return result


def summarize(pairs_path: str, summary_path: str, contrast: str, min_effect: float) -> dict:
    with open(pairs_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    groups = {}   # group -> setup -> layer -> recovery values
    num_valid = 0
    for record in records:
        clean = record["metrics"]["clean"]
        d_src = clean["source_prompt"]["logp_pos_resp"] - clean["source_prompt"]["logp_neg_resp"]
        d_tgt = clean["target_prompt"]["logp_pos_resp"] - clean["target_prompt"]["logp_neg_resp"]
        effect = d_src - d_tgt
        if record["identical_responses"] or abs(effect) < min_effect:
            continue
        num_valid += 1
        for setup, by_layer in record["metrics"]["patched"].items():
            for layer, m in by_layer.items():
                value = ((m["logp_pos_resp"] - m["logp_neg_resp"]) - d_tgt) / effect
                for group in ["all", get_group(contrast, record["question_key"])]:
                    groups.setdefault(group, {}).setdefault(setup, {}).setdefault(layer, []).append(value)

    diff_patchable = {}
    for record in records:
        group = get_group(contrast, record["question_key"])
        counts = diff_patchable.setdefault(group, {"patchable": 0, "total": 0})
        counts["patchable"] += record["spans"].get("diff_patchable", False)
        counts["total"] += 1

    # doctor_masked: how closely the masked target reproduces the real bare prompt (difference in D, relative to effect)
    mask_check = []
    for record in records:
        clean = record["metrics"]["clean"]
        if "bare_prompt" not in clean:
            continue
        d = lambda m: m["logp_pos_resp"] - m["logp_neg_resp"]
        effect = d(clean["source_prompt"]) - d(clean["target_prompt"])
        if abs(effect) >= min_effect:
            mask_check.append(abs(d(clean["target_prompt"]) - d(clean["bare_prompt"])) / abs(effect))

    summary = {
        "contrast": contrast,
        "min_effect": min_effect,
        "num_pairs": len(records),
        "num_pairs_with_effect": num_valid,
        "diff_patchable_by_group": diff_patchable,
        "masked_vs_bare_gap_relative_to_effect": aggregate(mask_check) if mask_check else None,
        "by_group": {
            group: {
                setup: {l: aggregate(v) for l, v in sorted(by_layer.items(), key=lambda kv: int(kv[0]))}
                for setup, by_layer in by_setup.items()
            }
            for group, by_setup in sorted(groups.items(), key=lambda kv: kv[0] != "all")
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    return summary


def plot(summary: dict, output_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = {
        "framing": "Positive -> negative framing",
        "doctor_friend": "'My doctor said,' -> 'My friend said,'",
        "doctor_masked": "'My doctor said,' -> masked prefix",
    }
    setup_titles = {"diff": "differing tokens", "after": "all positions after them", "claim": "the <xyz> tokens"}
    setups = list(dict.fromkeys(s for by_setup in summary["by_group"].values() for s in by_setup))
    fig, axes = plt.subplots(1, len(setups), figsize=(6.5 * len(setups), 4.5), sharey=True, squeeze=False)
    for ax, setup in zip(axes[0], setups):
        for group, by_setup in summary["by_group"].items():
            if setup not in by_setup:
                continue
            by_layer = by_setup[setup]
            layers = [int(l) for l in by_layer]
            means = [by_layer[str(l)]["mean"] for l in layers]
            if group == "all":
                ax.errorbar(layers, means, yerr=[by_layer[str(l)]["sem"] for l in layers], marker="o", capsize=3,
                            color="black", linewidth=2, label="all", zorder=3)
            else:
                ax.plot(layers, means, marker=".", linewidth=1, alpha=0.7, label=group)
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.axhline(1, color="grey", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Layer patched")
        ax.set_title(f"{setup}: {setup_titles[setup]}")
        ax.legend(fontsize=7, ncol=2)
    axes[0][0].set_ylabel("Mean recovery")
    fig.suptitle(f"{titles[summary['contrast']]} span patching (n={summary['num_pairs_with_effect']} pairs, effect >= {summary['min_effect']})")
    fig.tight_layout()
    path = os.path.join(output_dir, "span_patching.png")
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

            for prompt_pair in get_span_prompt_pairs(example, args.contrast):
                key = pair_key(review_id, prompt_pair["question_key"])
                if key in done:
                    continue
                src_ids = build_prompt_ids(tokenizer, prompt_pair["source_question"], rct_inputs, model.device)
                tgt_ids = build_prompt_ids(tokenizer, prompt_pair["target_question"], rct_inputs, model.device)
                if max(src_ids.shape[1], tgt_ids.shape[1]) + args.max_new_tokens > context_length:
                    print(f"Skipping {key}: prompt + max_new_tokens exceeds context length ({context_length})")
                    continue
                src_resp = greedy_generate(model, tokenizer, src_ids, args.max_new_tokens)
                tgt_resp = greedy_generate(model, tokenizer, tgt_ids, args.max_new_tokens)

                if args.contrast == "doctor_masked":
                    # Target = source tokens with the prefix masked; R- is the bare prompt's response
                    masked = prepare_masked_pair(tokenizer, src_ids, tgt_ids, prompt_pair["source_question"],
                                                 prompt_pair["target_question"], rct_inputs, model.device)
                    spans = masked["spans"]
                    metrics = score_pair(model, src_ids, src_ids, src_resp, tgt_resp, masked["positions"], layers,
                                         target_kwargs=masked["target_kwargs"], bare_ids=tgt_ids)
                else:
                    spans = find_spans(tokenizer, src_ids, tgt_ids)
                    metrics = score_pair(model, src_ids, tgt_ids, src_resp, tgt_resp, setup_positions(spans), layers)

                record = {
                    "pair_key": key,
                    "review_id": review_id,
                    "question_key": prompt_pair["question_key"],
                    "contrast": args.contrast,
                    "source_question": prompt_pair["source_question"],
                    "target_question": prompt_pair["target_question"],
                    "source_answer": tokenizer.decode(src_resp, skip_special_tokens=True).strip(),
                    "target_answer": tokenizer.decode(tgt_resp, skip_special_tokens=True).strip(),
                    "identical_responses": torch.equal(src_resp, tgt_resp),
                    "spans": spans,
                    "metrics": metrics,
                }
                out_file.write(json.dumps(record) + "\n")
                out_file.flush()

            gc.collect()
            torch.cuda.empty_cache()

    summary = summarize(pairs_path, summary_path, args.contrast, args.min_effect)
    print(f"Pairs: {summary['num_pairs']} (effect >= {args.min_effect}: {summary['num_pairs_with_effect']})")
    if args.contrast == "doctor_masked":
        print(f"Masked target vs real bare prompt, |D gap| / effect: {summary['masked_vs_bare_gap_relative_to_effect']}")
    else:
        print(f"Pairs where the differing spans have equal length (diff setup possible): {summary['diff_patchable_by_group']}")
    for setup, by_layer in summary["by_group"].get("all", {}).items():
        print(f"Mean recovery, {setup}:")
        for layer, r in by_layer.items():
            print(f"  layer {int(layer):>2}: {r['mean']:.3f} +/- {r['sem']:.3f} (median {r['median']:.3f}, n={r['n']})")
    plot(summary, args.output_dir)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch the differing tokens, or all positions after them, from source into target")
    parser.add_argument("--contrast", required=True, choices=CONTRASTS, help="framing (positive control), doctor_friend, or doctor_masked.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="HuggingFace model name.")
    parser.add_argument("--input_path", required=True, help="dataset with RCT inputs and framed questions (extracted question set).")
    parser.add_argument("--output_dir", required=True, help="directory to save the outputs.")
    parser.add_argument("--num_reviews", default=15, type=int, help="number of reviews.")
    parser.add_argument("--sample_pool", default=30, type=int, help="reviews are the first --num_reviews of the seeded sample of this size used by the main sweeps.")
    parser.add_argument("--max_new_tokens", default=512, type=int, help="max tokens for the clean greedy responses.")
    parser.add_argument("--layer_stride", default=5, type=int, help="patch every k-th layer starting at 0.")
    parser.add_argument("--min_effect", default=3.0, type=float, help="pairs with |D(source) - D(target)| below this are left out of the summary (near-zero effects make recovery unstable).")
    parser.add_argument("--seed", default=SEED, type=int, help="seed for sampling reviews.")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, help="only run 1 review.")
    args = parser.parse_args()

    print()
    print("Arguments Provided for Span Patching:")
    for k, v in vars(args).items():
        print(f"{k:<18} {v}")
    print()

    main(args)
