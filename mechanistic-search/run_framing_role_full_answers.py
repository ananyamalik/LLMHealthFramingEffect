"""
Full written answers for the key conditions of run_framing_role_mediation.py (which reads only the yes/no answer).

Prompt: prompts/question_only.jinja2 ("Question: <question>"), greedy answers. The vectors are fit the same way as in
run_framing_role_mediation.py / run_unit_steering.py, at the last prompt token of this prompt (training reviews).
All edits are made on the prompt pass only; the generated tokens see them through the cached keys/values.

--experiment steer (first --num_reviews test reviews; prompts: positive / negative claim x no role / "My <role>
said," and the neutral sentence). Per prompt, for each --gen_layers layer L and --alphas alpha:
    unsteered                        the normal answer
    push                             -alpha x framing at the last prompt token at L
    push + clamp role                the same, with the projection on the role vector held at its unsteered value at
                                     every layer after L
    push + ablate role               the same, with that projection set to 0
    ablate role                      the block alone, no push (the baseline for "push + ablate role")
    push + clamp random              control: a random direction held instead
--experiment patch: "My <source_role> said, <claim>" -> "My <target_role> said, <claim>", both framings. The target
is answered with the source's residual written in at one group of positions and one layer (--gen_patches,
<group>@<layer>; groups: role (the role word), said (' said' ','), claim (the treatment / condition tokens), last
(the last prompt token), all (everything from the role word on; should give the source's answer)).

Outputs (--output_dir; same folder as run_framing_role_mediation.py):
    directions_full.pt                        the vectors fit on this prompt (fit once and reused)
    full_<scale>/steer_answers.jsonl          per test review: questions and every answer (resumable)
    full_<scale>/steer_logprobs.jsonl         per test review: the tokens and their log-probabilities of each answer
                                              (format as run_unit_steering.py's logprobs.jsonl)
    full_<scale>/steer_answers.csv            one row per answer, next to the unsteered answer of the same prompt
    full_patch_<source>_to_<target>/patch_answers.jsonl, patch_logprobs.jsonl, patch_answers.csv
                                              the same for patching, next to the source and target answers
"""
import argparse
import csv
import gc
import json
import os
from contextlib import contextmanager

import torch
from tqdm import tqdm

from patching_utils import (
    DEFAULT_MODEL_NAME, SEED, set_global_seed, load_model_and_tokenizer, get_decoder_layers, load_dataset,
    get_stop_token_ids, cache_residuals,
)
from run_role_patching import FRAMINGS, role_prefix, get_claims, encode_prompt
from run_unit_steering import VECTORS, NO_ROLE, DEFAULT_ALPHAS, prompt_ids, fit_directions, generate
from run_framing_role_mediation import PROMPT_KEYS, question_for, intervene, stack_rows, patch_positions
from get_steering_vector import split_reviews

ANSWER = "full"
GEN_BLOCKS = [("clamp", "role"), ("ablate", "role"), ("clamp", "random")]
DEFAULT_GEN_ALPHAS = {"unit": [2.0], "residual": [0.25], "proj_std": [4.0]}
DEFAULT_PATCHES = ["role@0", "role@4", "role@8", "said@4", "said@8", "claim@8", "claim@12",
                   "last@16", "last@20", "last@24", "all@16"]


@torch.no_grad()
def last_token_outputs(model, ids: torch.Tensor) -> torch.Tensor:
    """Every decoder layer's output at the last prompt token [layers, d_model] (where `intervene` edits)."""
    capture = {}
    with intervene(model, {}, {}, capture):
        model.model(input_ids=ids, use_cache=False)
    return torch.stack([capture[l][0] for l in sorted(capture)])


def steer_rows(vectors: dict, units: dict, clean: torch.Tensor, layers: list[int], alphas: list[float], num_layers: int):
    """(condition, {"layer", "alpha", "block", "blocked"}, (add, block)) for every answer of one prompt."""
    def blocked(n, kind, L):
        return {l: (units[n][l], float(clean[l] @ units[n][l]) if kind == "clamp" else 0.0) for l in range(L + 1, num_layers)}

    rows = [("unsteered", {"layer": None, "alpha": 0.0, "block": None, "blocked": None}, ({}, {}))]
    for L in layers:
        rows.append(("ablate role", {"layer": L, "alpha": 0.0, "block": "ablate", "blocked": "role"}, ({}, blocked("role", "ablate", L))))
        for a in alphas:
            push = {L: -a * vectors["framing"][L]}
            rows.append(("push", {"layer": L, "alpha": a, "block": None, "blocked": None}, (push, {})))
            rows += [(f"push + {kind} {n}", {"layer": L, "alpha": a, "block": kind, "blocked": n}, (push, blocked(n, kind, L)))
                     for kind, n in GEN_BLOCKS]
    return rows


@torch.no_grad()
def generate_rows(model, tokenizer, ids: torch.Tensor, rows: list[tuple], args, stop_ids) -> tuple[list, list]:
    """Greedy answers of the prompt, one per (add, block) row (batched), edited on the prompt pass only."""
    texts, infos = [], []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        add, block = stack_rows(batch, model.config.hidden_size)
        with intervene(model, add, block, prompt_len=ids.shape[1]):
            t, i = generate(model, tokenizer, ids, len(batch), args, stop_ids)
        texts += t
        infos += i
    return texts, infos


@contextmanager
def patch_prompt(model, rows: list[tuple[int, list[int]] | None], src_states: dict, prompt_len: int):
    """Writes src_states[layer][positions] into decoder layer `layer`'s output for each batch row (None: no patch), on
    the prompt pass only."""
    by_layer = {}
    for i, row in enumerate(rows):
        if row is not None:
            by_layer.setdefault(row[0], []).append((i, row[1]))

    def make_hook(l):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if h.shape[1] == prompt_len:
                for i, positions in by_layer[l]:
                    h[i, positions] = src_states[l][positions].to(device=h.device, dtype=h.dtype)
            return output
        return hook

    layers = get_decoder_layers(model)
    handles = [layers[l].register_forward_hook(make_hook(l)) for l in by_layer]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def patch_groups(positions: list[tuple[int, str]]) -> dict[str, list[int]]:
    return {"role": [p for p, label in positions if label == "role"],
            "said": [p for p, label in positions if label.startswith("prefix")],
            "claim": [p for p, label in positions if label == "claim"],
            "last": [positions[-1][0]],
            "all": [p for p, _ in positions]}


def done_ids(path: str, key: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {json.loads(line)[key] for line in f if line.strip()}


def read_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_steer(model, tokenizer, fit: dict, test_set: list[dict], args, stop_ids) -> None:
    num_layers, d_model = fit["unit"]["framing"].shape
    scale = {"unit": torch.ones(num_layers), "residual": fit["residual_norm"], "proj_std": fit["proj_std"]["framing"]}[args.scale]
    vectors = {"framing": fit["unit"]["framing"] * scale[:, None]}
    random = torch.randn(num_layers, d_model, generator=torch.Generator().manual_seed(args.seed))
    units = {"role": fit["unit"]["role"], "random": random / random.norm(dim=-1, keepdim=True)}
    run_dir = os.path.join(args.output_dir, f"full_{args.scale}")
    os.makedirs(run_dir, exist_ok=True)
    torch.save({"vectors": vectors, "scale": scale, "units": units, "alphas": args.alphas, "layers": args.gen_layers,
                "scale_kind": args.scale}, os.path.join(run_dir, "vectors.pt"))
    print(f"\n[steer, full answers] -alpha x framing at layers {args.gen_layers} | alphas {args.alphas} | scale {args.scale}"
          f" | {len(test_set)} test reviews")

    answers_path, logprobs_path = os.path.join(run_dir, "steer_answers.jsonl"), os.path.join(run_dir, "steer_logprobs.jsonl")
    done = done_ids(answers_path, "review_id")
    with open(answers_path, "a", encoding="utf-8") as out_file, open(logprobs_path, "a", encoding="utf-8") as lp_file:
        for example in tqdm(test_set, desc="steer"):
            if example["ReviewID"] in done:
                continue
            claims = get_claims(example)
            record = {"review_id": example["ReviewID"], "questions": {k: question_for(claims, k) for k in PROMPT_KEYS}, "answers": []}
            logprobs = {"review_id": example["ReviewID"], "answers": []}
            for key in PROMPT_KEYS:
                ids = prompt_ids(tokenizer, record["questions"][key], NO_ROLE, model.device, ANSWER)
                rows = steer_rows(vectors, units, last_token_outputs(model, ids), args.gen_layers, args.alphas, num_layers)
                texts, infos = generate_rows(model, tokenizer, ids, [r for _, _, r in rows], args, stop_ids)
                for (condition, info, _), text, lp in zip(rows, texts, infos):
                    record["answers"].append({"prompt": key, "condition": condition, **info, "answer": text})
                    logprobs["answers"].append({"prompt": key, "condition": condition, **info, "logprobs": lp})
            out_file.write(json.dumps(record) + "\n")
            lp_file.write(json.dumps(logprobs) + "\n")
            out_file.flush()
            lp_file.flush()
            gc.collect()
            torch.cuda.empty_cache()

    rows = []
    for r in read_jsonl(answers_path):
        unsteered = {a["prompt"]: a["answer"] for a in r["answers"] if a["condition"] == "unsteered"}
        for a in r["answers"]:
            rows.append({"review_id": r["review_id"], "prompt": a["prompt"], "question": r["questions"][a["prompt"]],
                         "condition": a["condition"], "layer": a["layer"], "alpha": a["alpha"],
                         "identical_to_unsteered": a["answer"] == unsteered[a["prompt"]],
                         "unsteered_answer": unsteered[a["prompt"]], "answer": a["answer"]})
    write_csv(rows, os.path.join(run_dir, "steer_answers.csv"))
    print(f"Saved {os.path.join(run_dir, 'steer_answers.csv')} ({len(rows)} answers)")
    print("fraction of answers that differ from the unsteered answer (all prompts):")
    for condition in dict.fromkeys(r["condition"] for r in rows if r["condition"] != "unsteered"):
        by_layer = {L: [not r["identical_to_unsteered"] for r in rows if r["condition"] == condition and r["layer"] == L]
                    for L in args.gen_layers}
        print(f"  {condition:<22} " + " ".join(f"L{L} {sum(v) / len(v):.0%}" for L, v in by_layer.items() if v))


def run_patch(model, tokenizer, test_set: list[dict], args, stop_ids) -> None:
    src_role, tgt_role = args.source_role, args.target_role
    patches = [(p.split("@")[0], int(p.split("@")[1])) for p in args.gen_patches]
    layers = sorted({l for _, l in patches})
    run_dir = os.path.join(args.output_dir, f"full_patch_{src_role}_to_{tgt_role}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n[patch, full answers] '{role_prefix(src_role)}' -> '{role_prefix(tgt_role)}' | {args.gen_patches} | "
          f"{len(test_set)} test reviews")

    answers_path, logprobs_path = os.path.join(run_dir, "patch_answers.jsonl"), os.path.join(run_dir, "patch_logprobs.jsonl")
    done = done_ids(answers_path, "pair_key")
    with open(answers_path, "a", encoding="utf-8") as out_file, open(logprobs_path, "a", encoding="utf-8") as lp_file:
        for example in tqdm(test_set, desc="patch"):
            for framing, claim in get_claims(example).items():
                key = f"{example['ReviewID']}_{framing}"
                if key in done:
                    continue
                src_q, tgt_q = f"{role_prefix(src_role)} {claim}", f"{role_prefix(tgt_role)} {claim}"
                src = encode_prompt(tokenizer, ANSWER, src_q, [], model.device, "question_only")
                tgt = encode_prompt(tokenizer, ANSWER, tgt_q, [], model.device, "question_only")
                try:
                    groups = patch_groups(patch_positions(tokenizer, src, tgt, tgt_q, role_prefix(tgt_role)))
                except ValueError as e:
                    print(f"Skipping {key}: {e}")
                    continue
                seq = tgt[0].shape[1]
                with torch.no_grad(), cache_residuals(model, list(range(seq)), layer_indices=layers) as cache:
                    model.model(input_ids=src[0], use_cache=False)
                (src_text,), (src_lp,) = generate(model, tokenizer, src[0], 1, args, stop_ids)
                rows = [None] + [(l, groups[g]) for g, l in patches]
                texts, infos = [], []
                for start in range(0, len(rows), args.batch_size):
                    batch = rows[start:start + args.batch_size]
                    with patch_prompt(model, batch, cache, seq):
                        t, i = generate(model, tokenizer, tgt[0], len(batch), args, stop_ids)
                    texts += t
                    infos += i
                del cache
                conditions = ["target"] + [f"{g}@{l}" for g, l in patches]
                out_file.write(json.dumps({"pair_key": key, "review_id": example["ReviewID"], "framing": framing,
                                           "source_question": src_q, "target_question": tgt_q, "source_answer": src_text,
                                           "answers": dict(zip(conditions, texts))}) + "\n")
                lp_file.write(json.dumps({"pair_key": key, "source": src_lp, "answers": dict(zip(conditions, infos))}) + "\n")
                out_file.flush()
                lp_file.flush()
            gc.collect()
            torch.cuda.empty_cache()

    rows = []
    for r in read_jsonl(answers_path):
        for condition, text in r["answers"].items():
            if condition == "target":
                continue
            rows.append({"review_id": r["review_id"], "framing": r["framing"], "condition": condition,
                         "identical_to_target": text == r["answers"]["target"], "identical_to_source": text == r["source_answer"],
                         "source_question": r["source_question"], "target_question": r["target_question"],
                         "source_answer": r["source_answer"], "target_answer": r["answers"]["target"], "patched_answer": text})
    write_csv(rows, os.path.join(run_dir, "patch_answers.csv"))
    print(f"Saved {os.path.join(run_dir, 'patch_answers.csv')} ({len(rows)} answers)")
    print("patched answer: identical to source / identical to target")
    for condition in [f"{g}@{l}" for g, l in patches]:
        rs = [r for r in rows if r["condition"] == condition]
        if rs:
            print(f"  {condition:<10} {sum(r['identical_to_source'] for r in rs) / len(rs):.0%} / "
                  f"{sum(r['identical_to_target'] for r in rs) / len(rs):.0%}")


def main(args):
    set_global_seed(SEED)
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = load_dataset(args.input_path)
    test = split_reviews([e["ReviewID"] for e in dataset], args.test_fraction, args.seed)
    train_set = [e for e in dataset if e["ReviewID"] not in test]
    test_set = [e for e in dataset if e["ReviewID"] in test]
    if args.num_reviews > 0:
        test_set = test_set[:args.num_reviews]
    print(f"Reviews: {len(train_set)} train (extract) | {len(test_set)} test")

    model, tokenizer = load_model_and_tokenizer(args.model)
    stop_ids = list(get_stop_token_ids(model, tokenizer))
    if "steer" in args.experiment:
        path = os.path.join(args.output_dir, "directions_full.pt")
        fit = torch.load(path) if os.path.exists(path) else None
        if fit is not None and set(VECTORS) <= set(fit["unit"]):
            print(f"Loaded {path}")
        else:
            fit = fit_directions(model, tokenizer, train_set, args, ANSWER) | {"test_reviews": sorted(test), "prompt": ANSWER}
            torch.save(fit, path)
            print(f"Saved {path}")
        run_steer(model, tokenizer, fit, test_set, args, stop_ids)
    if "patch" in args.experiment:
        run_patch(model, tokenizer, test_set, args, stop_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full written answers for the framing -> role mediation conditions")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--input_path", required=True, help="dataset with the framed questions (extracted question set).")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--experiment", nargs="+", default=["steer", "patch"], choices=["steer", "patch"])
    parser.add_argument("--gen_layers", nargs="+", type=int, default=[16, 20, 24], help="steer: layers to push at.")
    parser.add_argument("--scale", default="residual", choices=list(DEFAULT_ALPHAS),
                        help="steer: unit / residual (x mean residual norm) / proj_std (x std of h.v).")
    parser.add_argument("--alphas", nargs="+", type=float, default=None, help="steer: push sizes (framing is added as -alpha; default 0.25 for residual).")
    parser.add_argument("--source_role", default="doctor", help="patch: the role whose activations are written in.")
    parser.add_argument("--target_role", default="friend", help="patch: the role prompt that receives them.")
    parser.add_argument("--gen_patches", nargs="+", default=DEFAULT_PATCHES, help="patch: <group>@<layer>, group in role / said / claim / last / all.")
    parser.add_argument("--num_reviews", default=20, type=int, help="first n test reviews (-1: all).")
    parser.add_argument("--test_fraction", default=0.2, type=float)
    parser.add_argument("--max_new_tokens", default=256, type=int)
    parser.add_argument("--top_logprobs", default=5, type=int, help="most likely tokens saved at each generated token (0: none).")
    parser.add_argument("--batch_size", default=32, type=int, help="answers generated at once.")
    parser.add_argument("--seed", default=SEED, type=int)
    args = parser.parse_args()
    args.alphas = args.alphas or DEFAULT_GEN_ALPHAS[args.scale]
    print("\nArguments:")
    for k, v in vars(args).items():
        print(f"{k:<16} {v}")
    print()
    main(args)
