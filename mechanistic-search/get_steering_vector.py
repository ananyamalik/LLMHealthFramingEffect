"""
Role steering vector: is the `last` patching effect of "My <role> said, <claim>" -> "<claim>" one shared direction?

Same pairs as run_role_patching.py (question-only prompt by default):
    run 1 (source): "My <role> said, <claim>"
    run 2 (target): "<claim>"
Instead of writing each pair's own run 1 residual into its run 2 (the `last` setup), a single vector per layer is
added to run 2's residual stream (decoder layer output) at its last prompt token, the position where the answer starts.

Vectors (difference of means at the last prompt token, fit on the training reviews only):
    pooled    v_L = mean over training pairs of (run 1 last - run 2 last), both framings
    framing   the same, fit only on training pairs of the test pair's framing (positive claims get v_positive, ...).
              If the role pushes the model toward agreeing with the claim, the pooled vector partly cancels across
              framings and the framing vectors should recover more
    random    a fixed random direction per layer with the norm of the pooled vector (control)
Reviews are split into train/test (--test_fraction, seeded), so both framings of a review are on the same side and
no test pair contributes to the vector it is steered with.

Steering: run 2 + alpha * v_L at its last prompt token, layer L, for every layer and --alphas (alpha = 1 adds the
average run 1 - run 2 difference). Like patch_residual, the vector is only added on passes that contain that position
(the prefill, and the teacher-forced pass for full answers); decode steps see it through the cached keys/values.

Answers (--answer), as in run_role_patching.py:
    yes_no: D(prompt) = log p(Yes) - log p(No) at the last prompt position.
    full:   R1, R2 = greedy answers to run 1/run 2, D(prompt) = log p(R1 | prompt) - log p(R2 | prompt).
            Also saves run 2's greedy answer with each vector at --generate_layers and --generate_alphas.

    recovery(vector, alpha, L) = (D(run 2 steered) - D(run 2)) / (D(run 1) - D(run 2))
        0 = the vector does nothing, 1 = the steered run 2 behaves like run 1 (as in the patching recovery).

Outputs (in --output_dir/<role>/):
    vectors.pt      pooled and per-framing vectors [layers, d_model], the random control, train/test reviews
    pairs.jsonl     one line per test (review, framing): prompts, clean D, steered D per vector/alpha/layer (resumable)
    summary.json    recovery per vector, alpha and layer (all / positive / negative), the clean role effect, and per
                    layer the vector norms and the cosine between the positive and negative framing vectors
"""
import argparse
import gc
import json
import os
from contextlib import contextmanager

import numpy as np
import torch
from tqdm import tqdm

from patching_utils import (
    DEFAULT_MODEL_NAME, SEED, set_global_seed, load_model_and_tokenizer, get_decoder_layers, get_context_length,
    format_rct_inputs, load_dataset, sample_reviews, cache_residuals, greedy_generate, response_logprob,
    yes_no_logit_diff,
)
from run_role_patching import (
    FRAMINGS, ANSWERS, PROMPTS, DEFAULT_MIN_EFFECT, YES_STRINGS, NO_STRINGS, role_prefix, get_claims, encode_prompt,
    find_positions, single_token_ids, aggregate,
)

VECTORS = ["pooled", "framing", "random"]


@contextmanager
def add_at_position(model, layer_idx: int, position: int, vector: torch.Tensor):
    """Adds `vector` to the output of decoder layer `layer_idx` at `position`, on passes that contain it."""
    def hook(module, inputs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        if hidden_states.shape[1] > position:
            hidden_states[0, position, :] += vector.to(device=hidden_states.device, dtype=hidden_states.dtype)
        return output

    handle = get_decoder_layers(model)[layer_idx].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def last_residual(model, input_ids: torch.Tensor, num_layers: int) -> torch.Tensor:
    """Residual stream at the last prompt token, every layer: [layers, d_model] (float32, CPU)."""
    with cache_residuals(model, [input_ids.shape[1] - 1]) as cache:
        model.model(input_ids=input_ids, use_cache=False)
    return torch.stack([cache[l][0].float().cpu() for l in range(num_layers)])


def build_pairs(tokenizer, dataset: list[dict], role: str, args, device, max_prompt_len: int) -> list[dict]:
    """Run 1/run 2 prompt ids per (review, framing), skipping the pairs run_role_patching.py skips."""
    pairs = []
    for example in dataset:
        rct_inputs = format_rct_inputs(example["Inputs"]) if args.prompt == "articles" else []
        for framing, claim in get_claims(example).items():
            key = f"{example['ReviewID']}_{framing}"
            prefix = role_prefix(role)
            src = encode_prompt(tokenizer, args.answer, f"{prefix} {claim}", rct_inputs, device, args.prompt)
            tgt = encode_prompt(tokenizer, args.answer, claim, rct_inputs, device, args.prompt)
            if src[0].shape[1] > max_prompt_len:
                print(f"Skipping {key}: prompt too long ({src[0].shape[1]} tokens)")
                continue
            try:
                find_positions(tokenizer, src, tgt, prefix)   # checks that the claim and suffix tokens match
            except ValueError as e:
                print(f"Skipping {key}: {e}")
                continue
            pairs.append({"pair_key": key, "review_id": example["ReviewID"], "framing": framing,
                          "source_question": f"{prefix} {claim}", "target_question": claim,
                          "src_ids": src[0], "tgt_ids": tgt[0]})
    return pairs


def split_reviews(review_ids: list[str], test_fraction: float, seed: int) -> set[str]:
    rng = np.random.default_rng(seed)
    review_ids = sorted(set(review_ids))
    num_test = max(1, round(len(review_ids) * test_fraction))
    return set(rng.choice(review_ids, num_test, replace=False).tolist())


def fit_vectors(model, train: list[dict], num_layers: int, seed: int) -> dict[str, torch.Tensor]:
    """Difference-of-means vectors [layers, d_model]: pooled, per framing, and a random control with the pooled norm."""
    diffs = {framing: [] for framing in FRAMINGS}
    for pair in tqdm(train, desc="fitting"):
        diff = last_residual(model, pair["src_ids"], num_layers) - last_residual(model, pair["tgt_ids"], num_layers)
        diffs[pair["framing"]].append(diff)
    vectors = {"pooled": torch.stack(diffs[FRAMINGS[0]] + diffs[FRAMINGS[1]]).mean(0)}
    vectors |= {framing: torch.stack(d).mean(0) for framing, d in diffs.items()}
    generator = torch.Generator().manual_seed(seed)
    random = torch.randn(vectors["pooled"].shape, generator=generator)
    vectors["random"] = random / random.norm(dim=-1, keepdim=True) * vectors["pooled"].norm(dim=-1, keepdim=True)
    return vectors


def vector_for(vectors: dict[str, torch.Tensor], name: str, framing: str) -> torch.Tensor:
    return vectors[framing] if name == "framing" else vectors[name]


def run_pair(model, tokenizer, args, pair: dict, vectors: dict, layers: list[int], yes_no_ids) -> dict:
    """Clean and steered D for one test pair."""
    src_ids, tgt_ids = pair["src_ids"], pair["tgt_ids"]
    position = tgt_ids.shape[1] - 1
    record = {}
    if args.answer == "yes_no":
        yes_ids, no_ids = yes_no_ids

        def score(ids):
            return yes_no_logit_diff(model, ids, yes_ids, no_ids)[0]
    else:
        src_resp = greedy_generate(model, tokenizer, src_ids, args.max_new_tokens)
        tgt_resp = greedy_generate(model, tokenizer, tgt_ids, args.max_new_tokens)

        def score(ids):
            return response_logprob(model, ids, src_resp) - response_logprob(model, ids, tgt_resp)

        record["run1_answer"] = tokenizer.decode(src_resp, skip_special_tokens=True).strip()   # "My <role> said, <claim>"
        record["run2_answer"] = tokenizer.decode(tgt_resp, skip_special_tokens=True).strip()   # "<claim>", clean
        record["identical_answers"] = torch.equal(src_resp, tgt_resp)
    record["d_source"], record["d_target"] = score(src_ids), score(tgt_ids)

    record["d_steered"] = {}   # vector -> alpha -> layer -> D
    for name in args.vectors:
        record["d_steered"][name] = {}
        for alpha in args.alphas:
            record["d_steered"][name][str(alpha)] = {}
            for layer_idx in layers:
                vector = alpha * vector_for(vectors, name, pair["framing"])[layer_idx]
                with add_at_position(model, layer_idx, position, vector):
                    record["d_steered"][name][str(alpha)][str(layer_idx)] = score(tgt_ids)

    if args.answer == "full":
        record["steered_run2_answers"] = {}   # vector -> alpha -> layer -> run 2's greedy answer with that vector
        for name in args.vectors:
            record["steered_run2_answers"][name] = {}
            for alpha in args.generate_alphas:
                record["steered_run2_answers"][name][str(alpha)] = {}
                for layer_idx in args.generate_layers:
                    vector = alpha * vector_for(vectors, name, pair["framing"])[layer_idx]
                    with add_at_position(model, layer_idx, position, vector):
                        resp = greedy_generate(model, tokenizer, tgt_ids, args.max_new_tokens)
                    record["steered_run2_answers"][name][str(alpha)][str(layer_idx)] = tokenizer.decode(resp, skip_special_tokens=True).strip()
    return record


def summarize(pairs_path: str, summary_path: str, vectors: dict, role: str, args, min_effect: float) -> dict:
    with open(pairs_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    recovery = {}   # group -> vector -> alpha -> layer -> values
    num_with_effect = 0
    for record in records:
        effect = record["d_source"] - record["d_target"]
        if abs(effect) < min_effect:
            continue
        num_with_effect += 1
        for name, by_alpha in record["d_steered"].items():
            for alpha, by_layer in by_alpha.items():
                for layer, d_steered in by_layer.items():
                    for group in ["all", record["framing"]]:
                        value = (d_steered - record["d_target"]) / effect
                        recovery.setdefault(group, {}).setdefault(name, {}).setdefault(alpha, {}).setdefault(layer, []).append(value)

    positive, negative = vectors[FRAMINGS[0]], vectors[FRAMINGS[1]]
    summary = {
        "role": role,
        "answer": args.answer,
        "prompt": args.prompt,
        "min_effect": min_effect,
        "num_train_pairs": vectors["num_train_pairs"],
        "num_pairs": len(records),
        "num_pairs_with_effect": num_with_effect,
        # Per layer: how big the vectors are, and whether the two framings share a direction
        "vector_norm": {name: vectors[name].norm(dim=-1).tolist() for name in ["pooled"] + FRAMINGS},
        "cos_positive_negative": torch.nn.functional.cosine_similarity(positive, negative, dim=-1).tolist(),
        "clean": {
            group: {
                key: aggregate([r[key] for r in records if group in ("all", r["framing"])])
                for key in ["d_source", "d_target"]
            } | {"effect": aggregate([r["d_source"] - r["d_target"] for r in records if group in ("all", r["framing"])])}
            for group in ["all"] + FRAMINGS
        },
        "recovery": {
            group: {
                name: {
                    alpha: {l: aggregate(v) for l, v in sorted(by_layer.items(), key=lambda kv: int(kv[0]))}
                    for alpha, by_layer in by_alpha.items()
                }
                for name, by_alpha in recovery[group].items()
            }
            for group in ["all"] + FRAMINGS if group in recovery
        },
    }
    if args.answer == "full":
        summary["num_identical_answers"] = sum(r["identical_answers"] for r in records)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    return summary


def main(args):
    set_global_seed(SEED)
    min_effect = args.min_effect if args.min_effect is not None else DEFAULT_MIN_EFFECT[args.answer]

    dataset = sample_reviews(load_dataset(args.input_path), args.num_reviews, seed=args.seed)
    print(f"Number of reviews: {len(dataset)}")

    model, tokenizer = load_model_and_tokenizer(args.model)
    num_layers = len(get_decoder_layers(model))
    layers = list(range(0, num_layers, args.layer_stride))
    if args.generate_layers is None:   # default: every 10th layer (0, 10, 20, 30 for 32-layer models)
        args.generate_layers = list(range(0, num_layers, 10))
    yes_no_ids = (single_token_ids(tokenizer, YES_STRINGS), single_token_ids(tokenizer, NO_STRINGS))
    max_prompt_len = get_context_length(model) - (args.max_new_tokens if args.answer == "full" else 0)
    print(f"Steering layers: {layers}")

    for role in args.roles:
        role_dir = os.path.join(args.output_dir, role)
        os.makedirs(role_dir, exist_ok=True)
        vectors_path = os.path.join(role_dir, "vectors.pt")
        pairs_path, summary_path = os.path.join(role_dir, "pairs.jsonl"), os.path.join(role_dir, "summary.json")

        pairs = build_pairs(tokenizer, dataset, role, args, model.device, max_prompt_len)
        test_reviews = split_reviews([p["review_id"] for p in pairs], args.test_fraction, args.seed)
        train = [p for p in pairs if p["review_id"] not in test_reviews]
        test = [p for p in pairs if p["review_id"] in test_reviews]
        print(f"\n=== Role: {role} ({len(train)} train / {len(test)} test pairs) ===")

        # Reuse the saved vectors on resume, so every test pair is steered with the same ones
        if os.path.exists(vectors_path):
            vectors = torch.load(vectors_path)
            if set(vectors["test_reviews"]) != test_reviews:
                raise ValueError(f"{vectors_path} was fit on a different split; remove it or use the same --num_reviews/--seed/--test_fraction")
            print(f"Loaded {vectors_path}")
        else:
            vectors = fit_vectors(model, train, num_layers, args.seed)
            vectors |= {"train_reviews": sorted({p["review_id"] for p in train}), "test_reviews": sorted(test_reviews),
                        "num_train_pairs": len(train), "model": args.model, "answer": args.answer, "prompt": args.prompt}
            torch.save(vectors, vectors_path)
            print(f"Saved {vectors_path}")
        cos = torch.nn.functional.cosine_similarity(vectors[FRAMINGS[0]], vectors[FRAMINGS[1]], dim=-1)
        print("cos(v_positive, v_negative): " + ", ".join(f"L{l} {cos[l]:.2f}" for l in layers))

        done = set()
        if os.path.exists(pairs_path):
            with open(pairs_path, "r", encoding="utf-8") as f:
                done = {json.loads(line)["pair_key"] for line in f if line.strip()}
        print(f"{len(done)} test pairs already done")

        with open(pairs_path, "a", encoding="utf-8") as out_file:
            for pair in tqdm(test, desc=role):
                if pair["pair_key"] in done:
                    continue
                record = {key: pair[key] for key in ["pair_key", "review_id", "framing", "source_question", "target_question"]}
                record |= {"role": role, "answer": args.answer, "prompt": args.prompt}
                record |= run_pair(model, tokenizer, args, pair, vectors, layers, yes_no_ids)
                out_file.write(json.dumps(record) + "\n")
                out_file.flush()
                gc.collect()
                torch.cuda.empty_cache()

        summary = summarize(pairs_path, summary_path, vectors, role, args, min_effect)
        print(f"Test pairs: {summary['num_pairs']} (|effect| >= {min_effect}: {summary['num_pairs_with_effect']})")
        for name, by_alpha in summary["recovery"].get("all", {}).items():
            for alpha, by_layer in by_alpha.items():
                print(f"Mean recovery, {name} x{alpha}: " + ", ".join(f"L{int(l)} {r['mean']:.2f}" for l, r in by_layer.items()))
        print(f"Saved {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Difference-of-means role vector, added at run 2's last prompt token")
    parser.add_argument("--answer", required=True, choices=ANSWERS, help="yes_no (single-token readout) or full (long-form answer).")
    parser.add_argument("--prompt", default="question_only", choices=PROMPTS, help="question_only ('Question: <question>') or articles (the main pipeline's prompt with article summaries).")
    parser.add_argument("--roles", nargs="+", default=["doctor"], help="roles to run (any word works in 'My <role> said,').")
    parser.add_argument("--vectors", nargs="+", default=VECTORS, choices=VECTORS, help="vectors to steer with.")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.5, 1.0, 2.0], help="multiples of the vector to add (1 = the average run 1 - run 2 difference).")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="HuggingFace model name.")
    parser.add_argument("--input_path", required=True, help="dataset with RCT inputs and framed questions (extracted question set).")
    parser.add_argument("--output_dir", required=True, help="outputs go to <output_dir>/<role>/.")
    parser.add_argument("--num_reviews", default=100, type=int, help="seeded sample of reviews (2 pairs each), split into train/test.")
    parser.add_argument("--test_fraction", default=0.2, type=float, help="fraction of reviews held out to test the vectors.")
    parser.add_argument("--layer_stride", default=1, type=int, help="steer every k-th layer starting at 0.")
    parser.add_argument("--max_new_tokens", default=512, type=int, help="full only: max tokens for greedy answers.")
    parser.add_argument("--generate_layers", nargs="*", type=int, default=None, help="full only: layers at which run 2's steered answer is generated and saved (default: every 10th layer; pass the flag with no values for none).")
    parser.add_argument("--generate_alphas", nargs="*", type=float, default=[1.0], help="full only: alphas at which steered answers are generated.")
    parser.add_argument("--min_effect", default=None, type=float, help="pairs with |D(run 1) - D(run 2)| below this are left out of the summary (default 0.5 yes_no, 3 full).")
    parser.add_argument("--seed", default=SEED, type=int, help="seed for sampling reviews, the train/test split and the random vector.")
    args = parser.parse_args()

    print("\nArguments:")
    for k, v in vars(args).items():
        print(f"{k:<18} {v}")
    print()
    main(args)
