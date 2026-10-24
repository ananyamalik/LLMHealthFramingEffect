"""
Patched generation for the span-patching contrasts, on a representative sample, for the evidence-direction check.

For each contrast, sample pairs from the span run (run_span_patching.py), then greedily generate the target prompt's
answer with the source residual stream patched in at a few (setup, layer) conditions that recovered the effect in the
teacher-forced runs. Clean answers are reused from the span run.

Default conditions:
    framing        diff @ 0, diff @ 5, after @ 10        (sampled from pairs where the diff setup is possible)
    doctor_friend  diff @ 0, after @ 10
    doctor_masked  claim @ 10, after @ 10                (target = doctor prompt with "My doctor said," masked)

Outputs (in --output_dir/<contrast>/), same layout as code/outputs/responses/<model>/question_responses.json so they can
be scored with code/run_evaluation.py --data_type framing:
    clean_question_responses.json                   positive_answer = clean source, negative_answer = clean target
    patched_<setup>_L<layer>_question_responses.json positive_answer = clean source, negative_answer = PATCHED target
                                                     (clean target kept as clean_negative_answer)
    span_generations.jsonl                           raw generations (used to resume)
    sanity_masked_vs_bare.json                       doctor_masked only: unpatched masked generation vs the bare answer
"""
import argparse
import json
import os
import random
import gc

import torch
from tqdm import tqdm

from patching_utils import (
    DEFAULT_MODEL_NAME, SEED, set_global_seed, load_model_and_tokenizer, format_rct_inputs, build_prompt_ids,
    load_dataset, patch_residual, greedy_generate, masked_prompt_inputs,
)
from run_span_patching import PAIRS_FILE_NAME as SPAN_PAIRS_FILE_NAME, setup_positions, get_source_cache, get_group

GENERATIONS_FILE_NAME = "span_generations.jsonl"
DATASET_COLUMNS_TO_DROP = ["Questions"]
DEFAULT_CONDITIONS = {
    "framing": [("diff", 0), ("diff", 5), ("after", 10)],
    "doctor_friend": [("diff", 0), ("after", 10)],
    "doctor_masked": [("claim", 10), ("after", 10)],
}


def load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def clean_effect(record: dict) -> float:
    clean = record["metrics"]["clean"]
    d = lambda m: m["logp_pos_resp"] - m["logp_neg_resp"]
    return d(clean["source_prompt"]) - d(clean["target_prompt"])


def sample_pairs(records: list[dict], contrast: str, conditions: list[tuple[str, int]], num_pairs: int,
                 min_effect: float, seed: int) -> list[dict]:
    """Stratified sample (equal share per group) of pairs where every condition's setup was run and the effect is clear."""
    needed = {setup for setup, _ in conditions}
    eligible = [r for r in records if not r["identical_responses"] and abs(clean_effect(r)) >= min_effect
                and needed <= set(r["metrics"]["patched"])]
    by_group = {}
    for r in eligible:
        by_group.setdefault(get_group(contrast, r["question_key"]), []).append(r)

    rng = random.Random(seed)
    for group_records in by_group.values():
        rng.shuffle(group_records)
    sample, round_idx = [], 0
    while len(sample) < num_pairs and any(round_idx < len(v) for v in by_group.values()):
        for group in sorted(by_group):
            if round_idx < len(by_group[group]) and len(sample) < num_pairs:
                sample.append(by_group[group][round_idx])
        round_idx += 1
    return sample


def pair_inputs(record: dict, contrast: str, tokenizer, rct_inputs, device) -> dict:
    """Prompt ids, patch positions per setup, and the target's generation inputs for one span record."""
    src_ids = build_prompt_ids(tokenizer, record["source_question"], rct_inputs, device)
    spans = record["spans"]
    if contrast == "doctor_masked":
        attention_mask, _ = masked_prompt_inputs(src_ids, spans["masked_positions"])
        claim = list(range(*spans["claim_span"]))
        after = list(range(spans["claim_span"][1], spans["claim_span"][1] + spans["num_after_positions"]))
        if after[-1] != src_ids.shape[1] - 1:
            raise ValueError(f"Rebuilt prompt does not match the span record for {record['pair_key']}")
        positions = {"claim": (claim, claim), "after": (after, after)}
        return {"src_ids": src_ids, "tgt_ids": src_ids, "tgt_attention_mask": attention_mask, "positions": positions}

    tgt_ids = build_prompt_ids(tokenizer, record["target_question"], rct_inputs, device)
    if (spans["src_span"][1] + spans["num_after_positions"] != src_ids.shape[1]
            or spans["tgt_span"][1] + spans["num_after_positions"] != tgt_ids.shape[1]):
        raise ValueError(f"Rebuilt prompts do not match the span record for {record['pair_key']}")
    return {"src_ids": src_ids, "tgt_ids": tgt_ids, "tgt_attention_mask": None, "positions": setup_positions(spans)}


def build_response_files(sample: list[dict], generations: dict, dataset_by_id: dict, conditions, out_dir: str) -> None:
    def assemble(answer_fn) -> list[dict]:
        by_review = {}
        for record in sample:
            answers = answer_fn(record)
            if answers is None:
                continue
            review_id = record["review_id"]
            if review_id not in by_review:
                example = {k: v for k, v in dataset_by_id[review_id].items() if k not in DATASET_COLUMNS_TO_DROP}
                example["ModelGeneratedAnswersWithQuestions"] = {}
                by_review[review_id] = example
            by_review[review_id]["ModelGeneratedAnswersWithQuestions"][record["question_key"]] = {
                "positive_question": record["source_question"],
                "negative_question": record["target_question"],
                **answers,
            }
        return list(by_review.values())

    def save(name, data):
        path = os.path.join(out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"Saved {path} ({sum(len(x['ModelGeneratedAnswersWithQuestions']) for x in data)} pairs)")

    save("clean_question_responses.json",
         assemble(lambda r: {"positive_answer": r["source_answer"], "negative_answer": r["target_answer"]}))
    for setup, layer in conditions:
        def patched(r, cond=f"{setup}_L{layer}"):
            answer = generations.get(r["pair_key"], {}).get(cond)
            if answer is None:
                return None
            return {"positive_answer": r["source_answer"], "negative_answer": answer, "clean_negative_answer": r["target_answer"]}
        save(f"patched_{setup}_L{layer}_question_responses.json", assemble(patched))


def run_contrast(model, tokenizer, contrast: str, args, dataset_by_id: dict) -> None:
    span_path = os.path.join(args.span_root, args.span_dirs[contrast], SPAN_PAIRS_FILE_NAME)
    records = load_jsonl(span_path)
    if not records:
        print(f"[{contrast}] no span records at {span_path}; skipping")
        return
    conditions = DEFAULT_CONDITIONS[contrast]
    sample = sample_pairs(records, contrast, conditions, 2 if args.debug else args.num_pairs, args.min_effect, args.seed)
    print(f"[{contrast}] sampled {len(sample)} pairs from {len(records)}; conditions {conditions}")

    out_dir = os.path.join(args.output_dir, contrast)
    os.makedirs(out_dir, exist_ok=True)
    generations_path = os.path.join(out_dir, GENERATIONS_FILE_NAME)
    generations = {}
    for row in load_jsonl(generations_path):
        generations.setdefault(row["pair_key"], {})[row["condition"]] = row["patched_target_answer"]

    sanity = []
    with open(generations_path, "a", encoding="utf-8") as out_file:
        for i, record in enumerate(tqdm(sample, desc=f"{contrast} pairs")):
            todo = [(s, l) for s, l in conditions if f"{s}_L{l}" not in generations.get(record["pair_key"], {})]
            check_mask = contrast == "doctor_masked" and i < args.sanity_pairs
            if not todo and not check_mask:
                continue
            rct_inputs = format_rct_inputs(dataset_by_id[record["review_id"]]["Inputs"])
            inputs = pair_inputs(record, contrast, tokenizer, rct_inputs, model.device)

            def generate_target():
                resp = greedy_generate(model, tokenizer, inputs["tgt_ids"], args.max_new_tokens, attention_mask=inputs["tgt_attention_mask"])
                return tokenizer.decode(resp, skip_special_tokens=True).strip()

            if check_mask:  # the unpatched masked run should write the bare prompt's answer
                masked_answer = generate_target()
                sanity.append({"pair_key": record["pair_key"], "matches_bare_answer": masked_answer == record["target_answer"]})

            for setup, layer in todo:
                src_pos, tgt_pos = inputs["positions"][setup]
                cache = get_source_cache(model, inputs["src_ids"], src_pos, [layer])
                with patch_residual(model, layer, tgt_pos, cache[layer]):
                    answer = generate_target()
                del cache
                condition = f"{setup}_L{layer}"
                generations.setdefault(record["pair_key"], {})[condition] = answer
                out_file.write(json.dumps({"pair_key": record["pair_key"], "condition": condition, "patched_target_answer": answer}) + "\n")
                out_file.flush()

            gc.collect()
            torch.cuda.empty_cache()

    if sanity:
        path = os.path.join(out_dir, "sanity_masked_vs_bare.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sanity, f, indent=4)
        print(f"[{contrast}] unpatched masked generation == bare answer in {sum(s['matches_bare_answer'] for s in sanity)}/{len(sanity)} pairs")

    for setup, layer in conditions:
        cond = f"{setup}_L{layer}"
        answers = [(generations.get(r["pair_key"], {}).get(cond), r) for r in sample]
        unchanged = sum(a == r["target_answer"] for a, r in answers if a is not None)
        to_source = sum(a == r["source_answer"] for a, r in answers if a is not None)
        print(f"[{contrast}] {cond}: patched answer identical to clean target in {unchanged}, to clean source in {to_source} of {len(sample)}")
    build_response_files(sample, generations, dataset_by_id, conditions, out_dir)


def main(args):
    set_global_seed(SEED)
    dataset_by_id = {ex["ReviewID"]: ex for ex in load_dataset(args.input_path)}
    model, tokenizer = load_model_and_tokenizer(args.model)
    for contrast in args.contrasts:
        run_contrast(model, tokenizer, contrast, args, dataset_by_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patched generation at selected span-patching conditions on a sample of pairs")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="HuggingFace model name (must match the span runs).")
    parser.add_argument("--input_path", required=True, help="same dataset file used by the span runs.")
    parser.add_argument("--span_root", required=True, help="outputs/<model_tag>/<question_set> folder holding the span runs.")
    parser.add_argument("--output_dir", required=True, help="directory to save patched generations (one subfolder per contrast).")
    parser.add_argument("--contrasts", nargs="+", default=list(DEFAULT_CONDITIONS), choices=list(DEFAULT_CONDITIONS), help="contrasts to run.")
    parser.add_argument("--num_pairs", default=20, type=int, help="pairs sampled per contrast (stratified by question type / claim framing).")
    parser.add_argument("--min_effect", default=3.0, type=float, help="only sample pairs with a clean teacher-forced effect at least this large.")
    parser.add_argument("--max_new_tokens", default=512, type=int, help="max tokens per generation (keep equal to the span runs).")
    parser.add_argument("--sanity_pairs", default=3, type=int, help="doctor_masked: pairs on which the unpatched masked generation is checked against the bare answer.")
    parser.add_argument("--seed", default=SEED, type=int, help="seed for sampling pairs.")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, help="only 2 pairs per contrast.")
    args = parser.parse_args()
    args.span_dirs = {"framing": "span", "doctor_friend": "doctor_friend/span", "doctor_masked": "doctor_masked/span"}

    print()
    print("Arguments Provided for Span Generation:")
    for k, v in vars(args).items():
        print(f"{k:<18} {v}")
    print()

    main(args)
