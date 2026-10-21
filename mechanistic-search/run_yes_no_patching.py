"""
Single-token readout: last-token (and differing-word) patching with a Yes/No answer.

Same framed questions and article summaries as the main experiments, but the prompt ends with one fixed follow-up,
identical in both prompts (prompts/question_answering_yes_no.jinja2):
    Based on the article summaries, does the treatment in the QUESTION work? Answer with only "Yes" or "No".

Readout at the last prompt position (no generation):
    D(prompt) = log p(Yes) - log p(No)        ("Yes"/"yes" and "No"/"no" pooled)
    recovery(setup, L) = (D(target, patched at L) - D(target)) / (D(source) - D(target))

Setups (source residual stream written into the target at layer L, every layer):
    last:  the last prompt token (always possible)
    diff:  the differing tokens, position by position (only when both sides have the same number of tokens)

Contrasts (--contrast): framing (positive -> negative question), doctor ("My doctor said, <xyz>" -> "<xyz>"),
doctor_friend ("My doctor said, <xyz>" -> "My friend said, <xyz>").

With a single-token readout, last-token recovery is expected to be ~0 at early layers and to rise to ~1 at middle/late
layers (the last position is where the answer is computed). If so, the near-zero last-token result of the long-form
runs comes from the long-form metric, not from how this model handles framing.

Outputs (in --output_dir):
    yes_no_patching_pairs.jsonl    one line per pair (clean readouts, spans, per-setup per-layer D)
    yes_no_patching_summary.json   per-setup per-layer recovery (overall and by group)
    yes_no_patching.png
"""
import argparse
import json
import os
import gc

import torch
from tqdm import tqdm

from patching_utils import (
    DEFAULT_MODEL_NAME, SEED, DOCTOR_PREFIX, set_global_seed, load_model_and_tokenizer, get_decoder_layers,
    format_rct_inputs, build_prompt_ids, load_dataset, sample_reviews, get_prompt_pairs, cache_residuals,
    patch_residual,
)
from run_layer_sweep import pair_key, load_done_pairs, aggregate
from run_span_patching import find_spans, get_span_prompt_pairs, get_group

PAIRS_FILE_NAME = "yes_no_patching_pairs.jsonl"
SUMMARY_FILE_NAME = "yes_no_patching_summary.json"
CONTRASTS = ["framing", "doctor", "doctor_friend"]
TEMPLATE_NAME = "question_answering_yes_no"
TEMPLATE_DIR = "../mechanistic-search/prompts"   # relative to code/, where render_prompt resolves template dirs
YES_STRINGS, NO_STRINGS = ["Yes", "yes"], ["No", "no"]


def get_yes_no_prompt_pairs(example: dict, contrast: str) -> list[dict]:
    if contrast == "doctor":
        return [
            {
                "question_key": p["question_key"],
                "source_question": f"{DOCTOR_PREFIX}, {p['target_question']}",
                "target_question": p["target_question"],
            }
            for p in get_prompt_pairs(example, "doctor", None)
        ]
    return get_span_prompt_pairs(example, contrast)


def single_token_ids(tokenizer, strings: list[str]) -> list[int]:
    ids = []
    for s in strings:
        encoded = tokenizer.encode(s, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"{s!r} is not a single token: {encoded}")
        ids.append(encoded[0])
    return ids


@torch.no_grad()
def readout(model, prompt_ids: torch.Tensor, yes_ids: list[int], no_ids: list[int]) -> dict:
    """log p(Yes) - log p(No) at the last prompt position, plus the top-1 token."""
    hidden = model.model(input_ids=prompt_ids, use_cache=False).last_hidden_state[:, -1:, :]
    log_probs = torch.log_softmax(model.lm_head(hidden).float(), dim=-1)[0, 0]
    log_yes, log_no = torch.logsumexp(log_probs[yes_ids], 0).item(), torch.logsumexp(log_probs[no_ids], 0).item()
    return {"d": log_yes - log_no, "log_p_yes": log_yes, "log_p_no": log_no, "top_token_id": int(log_probs.argmax())}


@torch.no_grad()
def score_pair(model, src_ids, tgt_ids, spans: dict, layers: list[int], yes_ids, no_ids) -> dict:
    positions = {"last": ([src_ids.shape[1] - 1], [tgt_ids.shape[1] - 1])}
    if spans["diff_patchable"]:
        positions["diff"] = (list(range(*spans["src_span"])), list(range(*spans["tgt_span"])))

    result = {
        "clean": {"source_prompt": readout(model, src_ids, yes_ids, no_ids), "target_prompt": readout(model, tgt_ids, yes_ids, no_ids)},
        "patched": {},
    }
    for setup, (src_pos, tgt_pos) in positions.items():
        with cache_residuals(model, src_pos, layer_indices=layers) as cache:
            model.model(input_ids=src_ids, use_cache=False)
        result["patched"][setup] = {}
        for layer_idx in layers:
            with patch_residual(model, layer_idx, tgt_pos, cache[layer_idx]):
                result["patched"][setup][str(layer_idx)] = readout(model, tgt_ids, yes_ids, no_ids)["d"]
        del cache
    return result


def summarize(pairs_path: str, summary_path: str, contrast: str, min_effect: float, yes_no_ids: set[int]) -> dict:
    with open(pairs_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    groups = {}   # group -> setup -> layer -> recovery values
    num_valid = 0
    for record in records:
        clean = record["metrics"]["clean"]
        d_src, d_tgt = clean["source_prompt"]["d"], clean["target_prompt"]["d"]
        effect = d_src - d_tgt
        if abs(effect) < min_effect:
            continue
        num_valid += 1
        for setup, by_layer in record["metrics"]["patched"].items():
            for layer, d_patch in by_layer.items():
                value = (d_patch - d_tgt) / effect
                for group in ["all", get_group(contrast, record["question_key"])]:
                    groups.setdefault(group, {}).setdefault(setup, {}).setdefault(layer, []).append(value)

    effects = [r["metrics"]["clean"]["source_prompt"]["d"] - r["metrics"]["clean"]["target_prompt"]["d"] for r in records]
    top_is_yes_no = [r["metrics"]["clean"][p]["top_token_id"] in yes_no_ids for r in records for p in ["source_prompt", "target_prompt"]]
    summary = {
        "contrast": contrast,
        "min_effect": min_effect,
        "num_pairs": len(records),
        "num_pairs_with_effect": num_valid,
        "effect_d_source_minus_d_target": aggregate(effects),
        "fraction_source_more_yes": sum(e > 0 for e in effects) / len(effects) if effects else None,
        "fraction_top_token_yes_or_no": sum(top_is_yes_no) / len(top_is_yes_no) if top_is_yes_no else None,
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
        "doctor": "'My doctor said,' -> bare claim",
        "doctor_friend": "'My doctor said,' -> 'My friend said,'",
    }
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
                ax.plot(layers, means, marker=".", linewidth=1, alpha=0.6, label=group)
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.axhline(1, color="grey", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Layer patched")
        ax.set_title(f"{setup}: {'last prompt token' if setup == 'last' else 'differing tokens'}")
        ax.legend(fontsize=7, ncol=2)
    axes[0][0].set_ylabel("Mean recovery of log p(Yes) - log p(No)")
    fig.suptitle(f"{titles[summary['contrast']]}, Yes/No readout (n={summary['num_pairs_with_effect']} pairs, |effect| >= {summary['min_effect']})")
    fig.tight_layout()
    path = os.path.join(output_dir, "yes_no_patching.png")
    fig.savefig(path, dpi=200)
    print(f"Saved {path}")


def main(args):
    set_global_seed(SEED)
    os.makedirs(args.output_dir, exist_ok=True)
    pairs_path = os.path.join(args.output_dir, PAIRS_FILE_NAME)
    summary_path = os.path.join(args.output_dir, SUMMARY_FILE_NAME)

    dataset = sample_reviews(load_dataset(args.input_path), args.sample_pool, seed=args.seed)[:args.num_reviews]
    if args.debug:
        dataset = dataset[:1]
    print(f"Number of reviews: {len(dataset)}")

    model, tokenizer = load_model_and_tokenizer(args.model)
    layers = list(range(0, len(get_decoder_layers(model)), args.layer_stride))
    yes_ids, no_ids = single_token_ids(tokenizer, YES_STRINGS), single_token_ids(tokenizer, NO_STRINGS)
    print(f"Patching layers: {layers}")
    print(f"Yes ids {yes_ids}, No ids {no_ids}")

    done = load_done_pairs(pairs_path)
    print(f"Pairs already done (resuming): {len(done)}")

    with open(pairs_path, "a", encoding="utf-8") as out_file:
        for example in tqdm(dataset, desc="Reviews"):
            review_id = example["ReviewID"]
            rct_inputs = format_rct_inputs(example["Inputs"])

            for prompt_pair in get_yes_no_prompt_pairs(example, args.contrast):
                key = pair_key(review_id, prompt_pair["question_key"])
                if key in done:
                    continue
                src_ids = build_prompt_ids(tokenizer, prompt_pair["source_question"], rct_inputs, model.device,
                                           template_name=TEMPLATE_NAME, template_dir=TEMPLATE_DIR)
                tgt_ids = build_prompt_ids(tokenizer, prompt_pair["target_question"], rct_inputs, model.device,
                                           template_name=TEMPLATE_NAME, template_dir=TEMPLATE_DIR)
                spans = find_spans(tokenizer, src_ids, tgt_ids)
                metrics = score_pair(model, src_ids, tgt_ids, spans, layers, yes_ids, no_ids)

                record = {
                    "pair_key": key,
                    "review_id": review_id,
                    "question_key": prompt_pair["question_key"],
                    "contrast": args.contrast,
                    "source_question": prompt_pair["source_question"],
                    "target_question": prompt_pair["target_question"],
                    "source_top_token": tokenizer.decode([metrics["clean"]["source_prompt"]["top_token_id"]]),
                    "target_top_token": tokenizer.decode([metrics["clean"]["target_prompt"]["top_token_id"]]),
                    "spans": spans,
                    "metrics": metrics,
                }
                out_file.write(json.dumps(record) + "\n")
                out_file.flush()

            gc.collect()
            torch.cuda.empty_cache()

    summary = summarize(pairs_path, summary_path, args.contrast, args.min_effect, set(yes_ids + no_ids))
    print(f"Pairs: {summary['num_pairs']} (|effect| >= {args.min_effect}: {summary['num_pairs_with_effect']})")
    print(f"Effect D(source) - D(target): {summary['effect_d_source_minus_d_target']}")
    print(f"Fraction of pairs where the source prompt is more 'Yes': {summary['fraction_source_more_yes']}")
    print(f"Fraction of clean runs whose top token is Yes/No: {summary['fraction_top_token_yes_or_no']}")
    for setup, by_layer in summary["by_group"].get("all", {}).items():
        print(f"Mean recovery, {setup}:")
        for layer, r in by_layer.items():
            print(f"  layer {int(layer):>2}: {r['mean']:.3f} +/- {r['sem']:.3f} (median {r['median']:.3f}, n={r['n']})")
    plot(summary, args.output_dir)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Last-token / differing-token patching with a single-token Yes/No readout")
    parser.add_argument("--contrast", required=True, choices=CONTRASTS, help="framing, doctor, or doctor_friend.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="HuggingFace model name.")
    parser.add_argument("--input_path", required=True, help="dataset with RCT inputs and framed questions (extracted question set).")
    parser.add_argument("--output_dir", required=True, help="directory to save the outputs.")
    parser.add_argument("--num_reviews", default=30, type=int, help="number of reviews.")
    parser.add_argument("--sample_pool", default=30, type=int, help="reviews are the first --num_reviews of the seeded sample of this size used by the main sweeps.")
    parser.add_argument("--layer_stride", default=1, type=int, help="patch every k-th layer starting at 0.")
    parser.add_argument("--min_effect", default=0.5, type=float, help="pairs with |D(source) - D(target)| below this (in nats) are left out of the summary.")
    parser.add_argument("--seed", default=SEED, type=int, help="seed for sampling reviews.")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, help="only run 1 review.")
    args = parser.parse_args()

    print()
    print("Arguments Provided for Yes/No Patching:")
    for k, v in vars(args).items():
        print(f"{k:<18} {v}")
    print()

    main(args)
