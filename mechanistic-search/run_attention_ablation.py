"""
Attention ablation (necessity test): is the doctor prefix read directly by the answer?

Doctor prompt "My doctor said, <xyz>", bare prompt "<xyz>" (same claims as run_comma_patching.py). In the doctor run we
block attention to the prefix tokens ("My doctor said,") at every layer, for different sets of query positions:
    none            nothing blocked (same 4D-mask code path; must reproduce the clean doctor run)
    answer_blocked  positions that predict answer tokens: the last prompt token and every answer token
    prompt_blocked  prompt tokens after the prefix (claim, articles, chat template) except the last prompt token
    all_blocked     every position after the prefix (reference; close to the bare prompt)

Metric (teacher forced):
    R+ = greedy response to the doctor prompt, R- = greedy response to the bare prompt
    D(run) = log p(R+ | run) - log p(R- | run)
    gap_remaining(condition) = (D(doctor, ablated) - D(bare)) / (D(doctor) - D(bare))
        1 -> effect intact, 0 -> collapsed to the bare prompt
If answer_blocked collapses the gap, the doctor effect works through direct reads of the prefix by the answer.

Outputs (in --output_dir):
    attention_ablation_pairs.jsonl    one line per pair (clean responses, positions, log-probs per condition)
    attention_ablation_summary.json   gap_remaining per condition (overall and by positive/negative claim)
"""
import argparse
import json
import os
import gc

import torch
from tqdm import tqdm

from patching_utils import (
    DEFAULT_MODEL_NAME, SEED, DOCTOR_PREFIX, set_global_seed, load_model_and_tokenizer, get_context_length,
    format_rct_inputs, build_prompt_ids, load_dataset, sample_reviews, get_prompt_pairs, greedy_generate,
    response_logprobs, blocked_attention_mask,
)
from run_layer_sweep import pair_key, load_done_pairs, aggregate
from run_span_patching import find_spans

PAIRS_FILE_NAME = "attention_ablation_pairs.jsonl"
SUMMARY_FILE_NAME = "attention_ablation_summary.json"
CONDITIONS = ["none", "answer_blocked", "prompt_blocked", "all_blocked"]
SANITY_TOLERANCE = 0.05  # max |D(none) - D(doctor)| relative to the effect before the run is stopped


def blocked_queries(condition: str, prefix_end: int, prompt_len: int, seq_len: int) -> list[int]:
    """Query positions that cannot attend to the prefix (seq_len = prompt + response)."""
    return {
        "none": [],
        "answer_blocked": list(range(prompt_len - 1, seq_len)),
        "prompt_blocked": list(range(prefix_end, prompt_len - 1)),
        "all_blocked": list(range(prefix_end, seq_len)),
    }[condition]


def score_conditions(model, doc_ids, bare_ids, doc_resp, bare_resp, prefix_positions: list[int]) -> dict:
    prompt_len, prefix_end = doc_ids.shape[1], prefix_positions[-1] + 1

    def logp(prompt_ids, resp, **kwargs):
        return response_logprobs(model, prompt_ids, resp, **kwargs)[0].sum().item()

    def ablated(condition, resp):
        seq_len = prompt_len + resp.shape[0]
        mask = blocked_attention_mask(seq_len, blocked_queries(condition, prefix_end, prompt_len, seq_len),
                                      prefix_positions, model.dtype, doc_ids.device)
        return logp(doc_ids, resp, full_attention_mask=mask)

    result = {
        "clean": {
            "doctor_prompt": {"logp_pos_resp": logp(doc_ids, doc_resp), "logp_neg_resp": logp(doc_ids, bare_resp)},
            "bare_prompt": {"logp_pos_resp": logp(bare_ids, doc_resp), "logp_neg_resp": logp(bare_ids, bare_resp)},
        },
        "ablated": {},
    }
    for condition in CONDITIONS:
        result["ablated"][condition] = {"logp_pos_resp": ablated(condition, doc_resp), "logp_neg_resp": ablated(condition, bare_resp)}
    return result


def d(m: dict) -> float:
    return m["logp_pos_resp"] - m["logp_neg_resp"]


def gap_remaining(record: dict, min_effect: float) -> dict | None:
    clean = record["metrics"]["clean"]
    d_doc, d_bare = d(clean["doctor_prompt"]), d(clean["bare_prompt"])
    effect = d_doc - d_bare
    if record["identical_responses"] or abs(effect) < min_effect:
        return None
    return {c: (d(m) - d_bare) / effect for c, m in record["metrics"]["ablated"].items()}


def summarize(pairs_path: str, summary_path: str, min_effect: float) -> dict:
    with open(pairs_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    groups = {"all": {}, "positive": {}, "negative": {}}   # group -> condition -> values
    num_valid = 0
    for record in records:
        remaining = gap_remaining(record, min_effect)
        if remaining is None:
            continue
        num_valid += 1
        framing = record["question_key"].rsplit("_", 1)[1]
        for condition, value in remaining.items():
            for group in ["all", framing]:
                groups[group].setdefault(condition, []).append(value)

    summary = {
        "min_effect": min_effect,
        "num_pairs": len(records),
        "num_pairs_with_effect": num_valid,
        "gap_remaining": {g: {c: aggregate(v) for c, v in by_c.items()} for g, by_c in groups.items()},
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    return summary


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
    context_length = get_context_length(model)

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
                doc_question = f"{DOCTOR_PREFIX}, {claim}"
                doc_ids = build_prompt_ids(tokenizer, doc_question, rct_inputs, model.device)
                bare_ids = build_prompt_ids(tokenizer, claim, rct_inputs, model.device)
                if doc_ids.shape[1] + args.max_new_tokens > context_length:
                    print(f"Skipping {key}: prompt + max_new_tokens exceeds context length ({context_length})")
                    continue

                spans = find_spans(tokenizer, doc_ids, bare_ids)
                if spans["tgt_span"][0] != spans["tgt_span"][1]:
                    raise ValueError(f"Bare prompt is not the doctor prompt minus its prefix: {spans['tgt_span_text']!r}")
                prefix_positions = list(range(*spans["src_span"]))

                doc_resp = greedy_generate(model, tokenizer, doc_ids, args.max_new_tokens)
                bare_resp = greedy_generate(model, tokenizer, bare_ids, args.max_new_tokens)
                metrics = score_conditions(model, doc_ids, bare_ids, doc_resp, bare_resp, prefix_positions)

                # The 4D-mask path with nothing blocked must reproduce the clean doctor run
                effect = d(metrics["clean"]["doctor_prompt"]) - d(metrics["clean"]["bare_prompt"])
                sanity_gap = abs(d(metrics["ablated"]["none"]) - d(metrics["clean"]["doctor_prompt"]))
                if abs(effect) >= args.min_effect and sanity_gap > SANITY_TOLERANCE * abs(effect):
                    raise RuntimeError(f"4D mask with nothing blocked does not reproduce the doctor run for {key} "
                                       f"(|D gap| {sanity_gap:.3f}, effect {effect:.3f}); check the transformers version's mask handling")

                record = {
                    "pair_key": key,
                    "review_id": review_id,
                    "question_key": prompt_pair["question_key"],
                    "doctor_question": doc_question,
                    "bare_question": claim,
                    "doctor_answer": tokenizer.decode(doc_resp, skip_special_tokens=True).strip(),
                    "bare_answer": tokenizer.decode(bare_resp, skip_special_tokens=True).strip(),
                    "identical_responses": torch.equal(doc_resp, bare_resp),
                    "prefix_positions": prefix_positions,
                    "prefix_text": spans["src_span_text"],
                    "sanity_none_vs_clean": sanity_gap,
                    "metrics": metrics,
                }
                out_file.write(json.dumps(record) + "\n")
                out_file.flush()

            gc.collect()
            torch.cuda.empty_cache()

    summary = summarize(pairs_path, summary_path, args.min_effect)
    print(f"Pairs: {summary['num_pairs']} (effect >= {args.min_effect}: {summary['num_pairs_with_effect']})")
    print("Fraction of the doctor effect remaining (1 = intact, 0 = bare prompt):")
    for group, by_condition in summary["gap_remaining"].items():
        print(f"  [{group}]")
        for condition, r in by_condition.items():
            if r["n"]:
                print(f"    {condition:<15} mean {r['mean']:.3f} +/- {r['sem']:.3f}  median {r['median']:.3f}  (n={r['n']})")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Block attention to the 'My doctor said,' prefix from the answer and/or the prompt")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="HuggingFace model name.")
    parser.add_argument("--input_path", required=True, help="dataset with RCT inputs and framed questions (extracted question set).")
    parser.add_argument("--output_dir", required=True, help="directory to save the outputs.")
    parser.add_argument("--num_reviews", default=30, type=int, help="number of reviews (each gives 6 pairs).")
    parser.add_argument("--sample_pool", default=30, type=int, help="reviews are the first --num_reviews of the seeded sample of this size used by the main sweeps.")
    parser.add_argument("--max_new_tokens", default=512, type=int, help="max tokens for the clean greedy responses.")
    parser.add_argument("--min_effect", default=3.0, type=float, help="pairs with |D(doctor) - D(bare)| below this are left out of the summary.")
    parser.add_argument("--seed", default=SEED, type=int, help="seed for sampling reviews.")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, help="only run 1 review.")
    args = parser.parse_args()

    print()
    print("Arguments Provided for Attention Ablation:")
    for k, v in vars(args).items():
        print(f"{k:<18} {v}")
    print()

    main(args)
