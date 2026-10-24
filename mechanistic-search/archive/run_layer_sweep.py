"""
Stage 1: layer sweep.

For every (review, question type) pair and every layer L, patch the positive-framing residual stream at the last
prompt token into the negative-framing run at layer L, and measure how much of the framing effect is recovered.

Metric (teacher forced, no sampling):
    R+ = greedy response to the positive prompt, R- = greedy response to the negative prompt
    D(prompt) = log p(R+ | prompt) - log p(R- | prompt)          (summed over response tokens)
    recovery(L) = (D(neg, patched at L) - D(neg)) / (D(pos) - D(neg))
        0 -> patch does nothing, 1 -> patched negative run prefers R+ over R- as much as the positive run does.

We also log KL(p_pos || p) for the first-token distribution, for the clean negative run and each patched run.

--contrast doctor runs the same sweep with "My doctor said <sentence>" as the source (positive_*) and the bare
"<sentence>" as the target (negative_*); see CONTRASTS in patching_utils.py.

Outputs (in --output_dir):
    layer_sweep_pairs.jsonl   one line per pair (clean responses + per-layer metrics); used to resume and by stage 2
    layer_sweep_summary.json  per-layer aggregates (overall and per question type)
"""
import argparse
import json
import os
import math
import gc

import numpy as np
import torch
from tqdm import tqdm

from patching_utils import (
    DEFAULT_MODEL_NAME, SEED, set_global_seed, load_model_and_tokenizer, get_decoder_layers, get_context_length,
    format_rct_inputs, build_prompt_ids, load_dataset, sample_reviews, get_prompt_pairs, CONTRASTS,
    get_positive_cache, patch_residual, last_positions, greedy_generate, response_logprobs, kl_divergence,
)

PAIRS_FILE_NAME = "layer_sweep_pairs.jsonl"
SUMMARY_FILE_NAME = "layer_sweep_summary.json"
MIN_EFFECT_SIZE = 1e-3  # pairs with |D(pos) - D(neg)| below this have no framing effect to recover


def pair_key(review_id, question_key: str) -> str:
    return f"{review_id}_{question_key}"


def dedupe_records(records: list[dict]) -> list[dict]:
    """Drop records whose (review, source prompt, target prompt) was already seen (older doctor sweeps scored some twice)."""
    seen, out = set(), []
    for record in records:
        key = (record["review_id"], record["positive_question"], record["negative_question"])
        if key not in seen:
            seen.add(key)
            out.append(record)
    return out


def load_done_pairs(pairs_path: str) -> set[str]:
    if not os.path.exists(pairs_path):
        return set()
    with open(pairs_path, "r", encoding="utf-8") as f:
        return {json.loads(line)["pair_key"] for line in f if line.strip()}


def score_pair(model, pos_ids, neg_ids, pos_resp, neg_resp, layers: list[int], num_last_tokens: int) -> dict:
    """Teacher-forced clean and patched metrics for one pair."""
    pos_cache = get_positive_cache(model, pos_ids, num_last_tokens)
    neg_positions = last_positions(neg_ids.shape[1], num_last_tokens)

    lp_pos_rpos, first_pos = response_logprobs(model, pos_ids, pos_resp)
    lp_pos_rneg, _ = response_logprobs(model, pos_ids, neg_resp)
    lp_neg_rpos, first_neg = response_logprobs(model, neg_ids, pos_resp)
    lp_neg_rneg, _ = response_logprobs(model, neg_ids, neg_resp)

    result = {
        "clean": {
            "pos_prompt": {"logp_pos_resp": lp_pos_rpos.sum().item(), "logp_neg_resp": lp_pos_rneg.sum().item()},
            "neg_prompt": {"logp_pos_resp": lp_neg_rpos.sum().item(), "logp_neg_resp": lp_neg_rneg.sum().item()},
            "kl_first_token_neg": kl_divergence(first_pos, first_neg),
        },
        "patched": {},
    }

    for layer_idx in layers:
        with patch_residual(model, layer_idx, neg_positions, pos_cache[layer_idx]):
            lp_patch_rpos, first_patch = response_logprobs(model, neg_ids, pos_resp)
            lp_patch_rneg, _ = response_logprobs(model, neg_ids, neg_resp)
        result["patched"][str(layer_idx)] = {
            "logp_pos_resp": lp_patch_rpos.sum().item(),
            "logp_neg_resp": lp_patch_rneg.sum().item(),
            "kl_first_token": kl_divergence(first_pos, first_patch),
        }

    return result


def compute_pair_recovery(record: dict) -> dict[str, dict] | None:
    """Per-layer recovery metrics for one pair record; None if the pair has no framing effect to recover."""
    clean = record["metrics"]["clean"]
    d_pos = clean["pos_prompt"]["logp_pos_resp"] - clean["pos_prompt"]["logp_neg_resp"]
    d_neg = clean["neg_prompt"]["logp_pos_resp"] - clean["neg_prompt"]["logp_neg_resp"]
    effect = d_pos - d_neg
    if record["identical_responses"] or abs(effect) < MIN_EFFECT_SIZE:
        return None

    kl_neg = clean["kl_first_token_neg"]
    out = {}
    for layer, m in record["metrics"]["patched"].items():
        d_patch = m["logp_pos_resp"] - m["logp_neg_resp"]
        out[layer] = {
            "recovery": (d_patch - d_neg) / effect,
            "kl_first_token": m["kl_first_token"],
            "first_token_restoration": 1 - m["kl_first_token"] / kl_neg if kl_neg > 0 else float("nan"),
        }
    return out


def aggregate(values: list[float]) -> dict:
    arr = np.array([v for v in values if not math.isnan(v)], dtype=float)
    if len(arr) == 0:
        return {"mean": None, "median": None, "sem": None, "n": 0}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "sem": float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
        "n": int(len(arr)),
    }


def summarize(pairs_path: str, summary_path: str) -> dict:
    with open(pairs_path, "r", encoding="utf-8") as f:
        records = dedupe_records([json.loads(line) for line in f if line.strip()])

    per_layer = {}                # layer -> metric -> values
    per_qtype = {}                # question type -> layer -> recovery values
    num_valid = 0
    for record in records:
        recovery = compute_pair_recovery(record)
        if recovery is None:
            continue
        num_valid += 1
        for layer, metrics in recovery.items():
            for name, value in metrics.items():
                per_layer.setdefault(layer, {}).setdefault(name, []).append(value)
            per_qtype.setdefault(record["question_key"], {}).setdefault(layer, []).append(metrics["recovery"])

    layers_sorted = sorted(per_layer, key=int)
    summary = {
        "contrast": records[0].get("contrast", "framing") if records else None,
        "num_pairs": len(records),
        "num_pairs_with_effect": num_valid,
        "num_pairs_identical_responses": sum(r["identical_responses"] for r in records),
        "by_layer": {l: {name: aggregate(vals) for name, vals in per_layer[l].items()} for l in layers_sorted},
        "by_question_type": {
            q: {l: aggregate(per_qtype[q][l]) for l in sorted(per_qtype[q], key=int)} for q in sorted(per_qtype)
        },
    }
    ranked = sorted(layers_sorted, key=lambda l: summary["by_layer"][l]["recovery"]["mean"], reverse=True)
    summary["layers_ranked_by_recovery"] = [int(l) for l in ranked]

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    return summary


def main(args):
    set_global_seed(SEED)
    os.makedirs(args.output_dir, exist_ok=True)
    pairs_path = os.path.join(args.output_dir, PAIRS_FILE_NAME)
    summary_path = os.path.join(args.output_dir, SUMMARY_FILE_NAME)

    dataset = sample_reviews(load_dataset(args.input_path), args.num_reviews, seed=args.seed)
    if args.debug:
        dataset = dataset[:2]
    print(f"Number of reviews: {len(dataset)}")

    model, tokenizer = load_model_and_tokenizer(args.model)
    num_layers = len(get_decoder_layers(model))
    layers = list(range(0, num_layers, args.layer_stride))
    if num_layers - 1 not in layers:
        layers.append(num_layers - 1)
    context_length = get_context_length(model)
    print(f"Patching layers: {layers}")

    done = load_done_pairs(pairs_path)
    print(f"Pairs already done (resuming): {len(done)}")

    with open(pairs_path, "a", encoding="utf-8") as out_file:
        for example in tqdm(dataset, desc="Reviews"):
            review_id = example["ReviewID"]
            rct_inputs = format_rct_inputs(example["Inputs"])

            for prompt_pair in get_prompt_pairs(example, args.contrast, args.question_types):
                q_key = prompt_pair["question_key"]
                key = pair_key(review_id, q_key)
                if key in done:
                    continue
                pos_question, neg_question = prompt_pair["source_question"], prompt_pair["target_question"]
                pos_ids = build_prompt_ids(tokenizer, pos_question, rct_inputs, model.device)
                neg_ids = build_prompt_ids(tokenizer, neg_question, rct_inputs, model.device)

                longest = max(pos_ids.shape[1], neg_ids.shape[1]) + args.max_new_tokens
                if longest > context_length:
                    print(f"Skipping {key}: prompt + max_new_tokens ({longest}) exceeds context length ({context_length})")
                    continue

                pos_resp = greedy_generate(model, tokenizer, pos_ids, args.max_new_tokens)
                neg_resp = greedy_generate(model, tokenizer, neg_ids, args.max_new_tokens)

                metrics = score_pair(model, pos_ids, neg_ids, pos_resp, neg_resp, layers, args.num_last_tokens)

                record = {
                    "pair_key": key,
                    "review_id": review_id,
                    "question_key": q_key,
                    "contrast": args.contrast,
                    "positive_question": pos_question,
                    "negative_question": neg_question,
                    "positive_answer": tokenizer.decode(pos_resp, skip_special_tokens=True).strip(),
                    "negative_answer": tokenizer.decode(neg_resp, skip_special_tokens=True).strip(),
                    "pos_prompt_len": pos_ids.shape[1],
                    "neg_prompt_len": neg_ids.shape[1],
                    "pos_resp_len": len(pos_resp),
                    "neg_resp_len": len(neg_resp),
                    "identical_responses": torch.equal(pos_resp, neg_resp),
                    "num_last_tokens": args.num_last_tokens,
                    "metrics": metrics,
                }
                out_file.write(json.dumps(record) + "\n")
                out_file.flush()

            gc.collect()
            torch.cuda.empty_cache()

    summary = summarize(pairs_path, summary_path)
    print(f"Pairs: {summary['num_pairs']} (with framing effect: {summary['num_pairs_with_effect']})")
    print("Mean recovery by layer:")
    for layer, metrics in summary["by_layer"].items():
        r = metrics["recovery"]
        print(f"  layer {int(layer):>2}: {r['mean']:.3f} +/- {r['sem']:.3f} (n={r['n']})")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layer sweep of last-token activation patching (source -> target prompt)")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="HuggingFace model name.")
    parser.add_argument("--input_path", required=True, help="dataset with RCT inputs and framed questions (cochrane_review_data_final_with_questions.json).")
    parser.add_argument("--output_dir", required=True, help="directory to save the sweep outputs.")
    parser.add_argument("--num_reviews", default=30, type=int, help="number of reviews to sample (each gives one pair per question type). Use -1 for all.")
    parser.add_argument("--contrast", default="framing", choices=CONTRASTS, help="framing: positive -> negative framing. doctor: 'My doctor said <sentence>' -> '<sentence>'.")
    parser.add_argument("--question_types", nargs="*", default=None, help="subset of question types (default: all single-turn types; for --contrast doctor, all claim-style types).")
    parser.add_argument("--max_new_tokens", default=512, type=int, help="max tokens for the clean greedy responses.")
    parser.add_argument("--layer_stride", default=1, type=int, help="patch every k-th layer (the last layer is always included).")
    parser.add_argument("--num_last_tokens", default=1, type=int, help="number of final prompt tokens to patch (1 = last token only; up to 5 are identical chat-template tokens).")
    parser.add_argument("--seed", default=SEED, type=int, help="seed for sampling reviews.")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, help="only run 2 reviews.")
    args = parser.parse_args()
    if args.num_reviews == -1:
        args.num_reviews = None

    print()
    print("Arguments Provided for Layer Sweep:")
    for k, v in vars(args).items():
        print(f"{k:<18} {v}")
    print()

    main(args)
