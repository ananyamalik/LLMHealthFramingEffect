"""
Stage 2: patched generation.

For the pairs scored in stage 1 (run_layer_sweep.py), generate the negative-framing response while the positive-framing
residual stream is patched into the last prompt token at layer L. Generation is greedy, matching the clean responses
saved in stage 1.

Outputs (in --output_dir/responses):
    clean_question_responses.json            positive_answer = clean positive, negative_answer = clean negative
    patched_layer{L}_question_responses.json positive_answer = clean positive, negative_answer = PATCHED negative
                                             (the clean negative is kept as clean_negative_answer)

Both use the same layout as code/outputs/responses/<model>/question_responses.json, so they can be scored with
code/run_evaluation.py --data_type framing.

For --contrast doctor sweeps, positive_* is the "My doctor said <sentence>" run and negative_* the bare "<sentence>" run,
so the patched negative_answer is the bare prompt with the doctor-attributed activations patched in. If patching removes the framing effect, the patched files should look more
like the baseline (positive vs. positive) evaluation than the clean file does.
"""
import argparse
import json
import os
import gc

import torch
from tqdm import tqdm

from patching_utils import (
    DEFAULT_MODEL_NAME, SEED, set_global_seed, load_model_and_tokenizer,
    format_rct_inputs, build_prompt_ids, load_dataset,
    get_positive_cache, patch_residual, last_positions, greedy_generate,
)
from run_layer_sweep import PAIRS_FILE_NAME, SUMMARY_FILE_NAME, dedupe_records

GENERATIONS_FILE_NAME = "patched_generations.jsonl"
DATASET_COLUMNS_TO_DROP = ["Questions"]


def load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def resolve_layers(layers_arg: list[str], top_k: int, sweep_dir: str) -> list[int]:
    """Either explicit layer indices or 'auto' = the top_k layers by mean recovery in the stage 1 summary."""
    if layers_arg != ["auto"]:
        return [int(l) for l in layers_arg]
    summary_path = os.path.join(sweep_dir, SUMMARY_FILE_NAME)
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    return sorted(summary["layers_ranked_by_recovery"][:top_k])


def build_response_files(pairs: list[dict], generations: dict, dataset_by_id: dict, layers: list[int], responses_dir: str) -> None:
    """Assemble outputs in the same layout as the main pipeline's question_responses.json."""
    os.makedirs(responses_dir, exist_ok=True)

    def assemble(answer_fn) -> list[dict]:
        by_review = {}
        for pair in pairs:
            answers = answer_fn(pair)
            if answers is None:
                continue
            review_id = pair["review_id"]
            if review_id not in by_review:
                example = {k: v for k, v in dataset_by_id[review_id].items() if k not in DATASET_COLUMNS_TO_DROP}
                example["ModelGeneratedAnswersWithQuestions"] = {}
                by_review[review_id] = example
            by_review[review_id]["ModelGeneratedAnswersWithQuestions"][pair["question_key"]] = {
                "positive_question": pair["positive_question"],
                "negative_question": pair["negative_question"],
                **answers,
            }
        return list(by_review.values())

    clean = assemble(lambda p: {"positive_answer": p["positive_answer"], "negative_answer": p["negative_answer"]})
    clean_path = os.path.join(responses_dir, "clean_question_responses.json")
    with open(clean_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=4)
    print(f"Saved {clean_path}")

    for layer in layers:
        def patched_answers(p, layer=layer):
            patched = generations.get(p["pair_key"], {}).get(str(layer))
            if patched is None:
                return None
            return {
                "positive_answer": p["positive_answer"],
                "negative_answer": patched,
                "clean_negative_answer": p["negative_answer"],
            }
        patched = assemble(patched_answers)
        path = os.path.join(responses_dir, f"patched_layer{layer}_question_responses.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(patched, f, indent=4)
        print(f"Saved {path}")


def main(args):
    set_global_seed(SEED)
    os.makedirs(args.output_dir, exist_ok=True)

    pairs = dedupe_records(load_jsonl(os.path.join(args.sweep_dir, PAIRS_FILE_NAME)))
    if not pairs:
        raise FileNotFoundError(f"No stage 1 pairs found in {args.sweep_dir}. Run run_layer_sweep.py first.")
    if args.debug:
        pairs = pairs[:4]
    layers = resolve_layers(args.layers, args.top_k, args.sweep_dir)
    print(f"Pairs: {len(pairs)}")
    print(f"Generating with patches at layers: {layers}")

    dataset_by_id = {ex["ReviewID"]: ex for ex in load_dataset(args.input_path)}

    # Resume: {pair_key: {layer: patched answer}}
    generations_path = os.path.join(args.output_dir, GENERATIONS_FILE_NAME)
    generations = {}
    for row in load_jsonl(generations_path):
        generations.setdefault(row["pair_key"], {})[str(row["layer"])] = row["patched_negative_answer"]

    model, tokenizer = load_model_and_tokenizer(args.model)

    with open(generations_path, "a", encoding="utf-8") as out_file:
        for pair in tqdm(pairs, desc="Patched generation"):
            todo = [l for l in layers if str(l) not in generations.get(pair["pair_key"], {})]
            if not todo:
                continue
            num_last_tokens = pair.get("num_last_tokens", args.num_last_tokens)

            rct_inputs = format_rct_inputs(dataset_by_id[pair["review_id"]]["Inputs"])
            pos_ids = build_prompt_ids(tokenizer, pair["positive_question"], rct_inputs, model.device)
            neg_ids = build_prompt_ids(tokenizer, pair["negative_question"], rct_inputs, model.device)
            pos_cache = get_positive_cache(model, pos_ids, num_last_tokens)
            neg_positions = last_positions(neg_ids.shape[1], num_last_tokens)

            for layer in todo:
                with patch_residual(model, layer, neg_positions, pos_cache[layer]):
                    patched_resp = greedy_generate(model, tokenizer, neg_ids, args.max_new_tokens)
                answer = tokenizer.decode(patched_resp, skip_special_tokens=True).strip()
                generations.setdefault(pair["pair_key"], {})[str(layer)] = answer
                out_file.write(json.dumps({
                    "pair_key": pair["pair_key"],
                    "layer": layer,
                    "patched_negative_answer": answer,
                }) + "\n")
                out_file.flush()

            gc.collect()
            torch.cuda.empty_cache()

    build_response_files(pairs, generations, dataset_by_id, layers, os.path.join(args.output_dir, "responses"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate negative-framing responses with positive-framing last-token activations patched in")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="HuggingFace model name (must match stage 1).")
    parser.add_argument("--input_path", required=True, help="same dataset file used in stage 1.")
    parser.add_argument("--sweep_dir", required=True, help="output directory of run_layer_sweep.py.")
    parser.add_argument("--output_dir", required=True, help="directory to save patched generations.")
    parser.add_argument("--layers", nargs="+", default=["auto"], help="layer indices to patch, or 'auto' to use the top-k layers from the stage 1 summary.")
    parser.add_argument("--top_k", default=3, type=int, help="number of layers to use with --layers auto.")
    parser.add_argument("--max_new_tokens", default=512, type=int, help="max tokens for patched responses (keep equal to stage 1).")
    parser.add_argument("--num_last_tokens", default=1, type=int, help="fallback if not recorded in stage 1 outputs.")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, help="only run 4 pairs.")
    args = parser.parse_args()

    print()
    print("Arguments Provided for Patched Generation:")
    for k, v in vars(args).items():
        print(f"{k:<18} {v}")
    print()

    main(args)
