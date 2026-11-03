"""
Unit-length framing and role vectors from the last prompt token, steered at the last prompt token.

Prompt: prompts/question_only.jinja2 ("Question: <question>"), full written answers (greedy).

Extract (training reviews; last prompt token; every decoder layer output):
    framing   mean over reviews of  [positive none] - [negative none]
              ("amantadine will work ..." - "amantadine won't work ...")
    <role>    mean over reviews of  ([positive <role>] - [positive none] + [negative <role>] - [negative none]) / 2
              ("My <role> said, <claim>" - "<claim>", averaged over the two framings), for doctor, friend, enemy
    role      the mean of the doctor, friend and enemy vectors
Each vector is normalized to unit length per layer (v), and added as alpha * scale * v. --scale:
    unit       scale = 1                                        (default alphas 0.5 1 2)
    residual   scale = mean |h| at the last token at that layer (default alphas 0.1 0.25 0.5)
    proj_std   scale = std over prompts of h . v at that layer  (default alphas 2 4 8)
(h = the last-token residuals of all training prompts: both framings x no role / doctor / friend / enemy.)

Experiments (held-out test reviews; the vectors are added at the last prompt token at one layer at a time):
    framing_flip   positive claim - alpha * framing; negative claim + alpha * framing
                   (does it answer like the other framing?)
    role_add       positive / negative claim (no role) + alpha * doctor / friend / enemy
                   (does it answer like "My <role> said, <claim>"?)
    interaction    neutral sentence ("Does <treatment> work for <condition>? What do you think?") +
                   alpha * role, +/- alpha * framing, and +/- alpha * framing + alpha * role
                   (does adding the role change what the framing push does?)
Only the prompt pass is steered (with --steer_generation also every generated token). Each steered answer is
compared with the unsteered answer of the same prompt, generated in the same batch.

Outputs:
    <output_dir>/directions.pt          {"raw": {name: [layers, d_model]}, "unit": {...}, "norms": {...},
                                         "residual_norm": [layers], "proj_std": {name: [layers]}, "train_reviews",
                                         "test_reviews", "model"}; fit once, reused by every --scale
    <output_dir>/<scale>/vectors.pt     the vectors as added: {"vectors": {name: [layers, d_model]} (alpha 1, i.e.
                                         scale * v), "scale": {name: [layers]}, "alphas", "layers"}
    <output_dir>/<scale>/answers.jsonl  one line per test review: its prompts, the real answers (positive, negative,
                                         neutral, "My <role> said," for each framing) and every steered answer with
                                         its experiment, prompt, vectors and alphas, layer and the unsteered answer;
                                         resumable
    <output_dir>/<scale>/answers.csv    one row per steered answer, next to the unsteered and the matching real answer
    <output_dir>/<scale>/logprobs.jsonl one line per test review: the log-probability of every generated token (and the
                                         --top_logprobs most likely tokens at each step) of each answer:
                                         {"review_id", "real": {label: info}, "steered": {prompt: [per batch:
                                         {"normal": info, "steered": [info per answer]}]}}, info = {"tokens",
                                         "token_ids", "logprobs", "top"}; answers.jsonl's "batch" says which batch an
                                         answer was generated in (its steered answers are in the same order)
"""
import argparse
import csv
import re
import gc
import json
import os
from contextlib import contextmanager

import numpy as np
import torch
from tqdm import tqdm

from patching_utils import (
    DEFAULT_MODEL_NAME, SEED, set_global_seed, load_model_and_tokenizer, get_decoder_layers, load_dataset,
    get_stop_token_ids,
)
from run_role_patching import FRAMINGS, ROLES, role_prefix, get_claims, encode_prompt
from get_steering_vector import split_reviews

VECTORS = ["framing"] + ROLES + ["role"]
NO_ROLE = "none"
PROMPTS = FRAMINGS + ["neutral"]
DEFAULT_ALPHAS = {"unit": [0.5, 1.0, 2.0], "residual": [0.1, 0.25, 0.5], "proj_std": [2.0, 4.0, 8.0]}
CLAIM = re.compile(r"^(.+?) will work for (.+)\. What do you think\?$")


def neutral_sentence(positive_claim: str) -> str:
    """'X will work for Y. What do you think?' -> 'Does X work for Y? What do you think?'"""
    treatment, condition = CLAIM.match(positive_claim).groups()
    return f"Does {treatment} work for {condition}? What do you think?"


def prompt_ids(tokenizer, claim: str, role: str, device) -> torch.Tensor:
    question = claim if role == NO_ROLE else f"{role_prefix(role)} {claim}"
    return encode_prompt(tokenizer, "full", question, [], device, "question_only")[0]


@torch.no_grad()
def last_token_states(model, ids: torch.Tensor) -> torch.Tensor:
    """Decoder layer outputs at the last prompt token [layers, d_model] (float32, cpu)."""
    hidden = model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states[1:]
    return torch.stack([h[0, -1] for h in hidden]).float().cpu()


def fit_directions(model, tokenizer, dataset: list[dict], args) -> dict:
    """Raw and unit diffmeans vectors {name: [layers, d_model]} (framing, each role, their mean), their norms, the mean residual norm at the last token
    [layers] and the std of the residuals projected on each unit vector {name: [layers]}."""
    diffs = {name: [] for name in VECTORS}
    all_states = []
    for example in tqdm(dataset, desc="extracting"):
        claims = get_claims(example)
        states = {(framing, role): last_token_states(model, prompt_ids(tokenizer, claims[framing], role, model.device))
                  for framing in FRAMINGS for role in [NO_ROLE] + ROLES}
        diffs["framing"].append(states["positive", NO_ROLE] - states["negative", NO_ROLE])
        for role in ROLES:
            diffs[role].append(sum(states[f, role] - states[f, NO_ROLE] for f in FRAMINGS) / len(FRAMINGS))
        diffs["role"].append(sum(diffs[r][-1] for r in ROLES) / len(ROLES))
        all_states += [s.half() for s in states.values()]
    raw = {name: torch.stack(v).mean(0) for name, v in diffs.items()}
    unit = {n: v / v.norm(dim=-1, keepdim=True) for n, v in raw.items()}
    h = torch.stack(all_states)   # [prompts, layers, d_model]
    return {"raw": raw, "unit": unit, "norms": {n: v.norm(dim=-1) for n, v in raw.items()},
            "residual_norm": h.float().norm(dim=-1).mean(0),
            "proj_std": {n: torch.einsum("pld,ld->pl", h.float(), v).std(0) for n, v in unit.items()},
            "train_reviews": sorted(e["ReviewID"] for e in dataset), "model": args.model}


@contextmanager
def add_per_row(model, layer_vectors: dict[int, torch.Tensor], prompt_len: int, during_generation: bool):
    """Adds layer_vectors[l][row] to decoder layer l's output for each batch row: at the last prompt token on the
    prompt pass (and at every generated token if `during_generation`)."""
    def make_hook(vectors):
        def hook(module, inputs, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            if hidden_states.shape[1] == prompt_len or during_generation:
                hidden_states[:, -1, :] += vectors.to(device=hidden_states.device, dtype=hidden_states.dtype)
            return output
        return hook

    layers = get_decoder_layers(model)
    handles = [layers[l].register_forward_hook(make_hook(v)) for l, v in layer_vectors.items()]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def generate(model, tokenizer, ids: torch.Tensor, n: int, args, stop_ids) -> tuple[list[str], list[dict]]:
    """Greedy answers of `n` copies of the prompt `ids` (under any active hooks): the texts and, per answer, its tokens
    with their log-probabilities: {"tokens", "token_ids", "logprobs", "top": [[[token, logprob], ...] per token]}."""
    output = model.generate(ids.repeat(n, 1), attention_mask=torch.ones_like(ids).repeat(n, 1),
                            max_new_tokens=args.max_new_tokens, do_sample=False, temperature=None, top_p=None,
                            eos_token_id=stop_ids, pad_token_id=tokenizer.eos_token_id,
                            return_dict_in_generate=True, output_logits=True)
    generated = output.sequences[:, ids.shape[1]:].tolist()
    log_probs = [torch.log_softmax(step.float(), dim=-1) for step in output.logits]   # per step [n, vocab]
    texts, infos = [], []
    for row, token_ids in enumerate(generated):
        stop = next((i for i, t in enumerate(token_ids) if t in stop_ids), len(token_ids))
        token_ids = token_ids[:stop]
        texts.append(tokenizer.decode(token_ids, skip_special_tokens=True).strip())
        info = {"tokens": [tokenizer.decode([t]) for t in token_ids], "token_ids": token_ids,
                "logprobs": [round(log_probs[i][row, t].item(), 4) for i, t in enumerate(token_ids)], "top": []}
        if args.top_logprobs > 0:
            for i in range(len(token_ids)):
                values, indices = log_probs[i][row].topk(args.top_logprobs)
                info["top"].append([[tok, round(v, 4)] for tok, v in
                                    zip([tokenizer.decode([t]) for t in indices.tolist()], values.tolist())])
        infos.append(info)
    return texts, infos


@torch.no_grad()
def steered_answers(model, tokenizer, ids: torch.Tensor, conds: list[tuple], vectors: dict, args, stop_ids) -> tuple[list, list]:
    """For each ({vector: alpha}, layer) in `conds`, batched (row 0 of every batch is unsteered):
    (unsteered answer of its batch, steered answer, batch index), and per batch the token log-probabilities
    {"normal": info, "steered": [info per row]}."""
    out, logprobs = [], []
    for start in range(0, len(conds), args.batch_size - 1):
        batch = [None] + conds[start:start + args.batch_size - 1]
        layer_vectors = {}
        for row, cond in enumerate(batch):
            if cond is None:
                continue
            alphas, layer = cond
            layer_vectors.setdefault(layer, torch.zeros(len(batch), vectors["framing"].shape[-1]))
            layer_vectors[layer][row] = sum(a * vectors[name][layer] for name, a in alphas.items())
        with add_per_row(model, layer_vectors, ids.shape[1], args.steer_generation):
            texts, infos = generate(model, tokenizer, ids, len(batch), args, stop_ids)
        out += [(texts[0], text, len(logprobs)) for text in texts[1:]]
        logprobs.append({"normal": infos[0], "steered": infos[1:]})
    return out, logprobs


def experiments(alphas: list[float], layers: list[int]) -> dict[str, list[tuple]]:
    """Per prompt: (experiment, {vector: alpha}, real answer it should look like, layer) for every steered answer."""
    out = {prompt: [] for prompt in PROMPTS}
    for a in alphas:
        for l in layers:
            out["positive"].append(("framing_flip", {"framing": -a}, "negative", l))
            out["negative"].append(("framing_flip", {"framing": a}, "positive", l))
            for framing in FRAMINGS:
                out[framing] += [("role_add", {role: a}, f"{role}_{framing}", l) for role in ROLES]
            out["neutral"] += [("interaction", {"role": a}, "neutral", l),
                               ("interaction", {"framing": a}, "positive", l),
                               ("interaction", {"framing": -a}, "negative", l),
                               ("interaction", {"framing": a, "role": a}, "positive", l),
                               ("interaction", {"framing": -a, "role": a}, "negative", l)]
    return out


def steer_label(alphas: dict) -> str:
    return " ".join(f"{a:+g} x {name}" for name, a in alphas.items())


def write_csv(records: list[dict], path: str) -> None:
    columns = ["review_id", "experiment", "prompt", "question", "steer", "layer", "identical_to_normal", "normal_answer",
               "steered_answer", "real_answer_label", "real_answer"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in records:
            for s in r["steered"]:
                writer.writerow({"review_id": r["review_id"], "experiment": s["experiment"], "prompt": s["prompt"],
                                 "question": r["prompts"][s["prompt"]], "steer": steer_label(s["alphas"]),
                                 "layer": s["layer"], "identical_to_normal": s["answer"] == s["normal_answer"],
                                 "normal_answer": s["normal_answer"], "steered_answer": s["answer"],
                                 "real_answer_label": s["real"], "real_answer": r["real_answers"][s["real"]]})


def main(args):
    set_global_seed(SEED)
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = load_dataset(args.input_path)
    test = split_reviews([e["ReviewID"] for e in dataset], args.test_fraction, args.seed)
    train_set = [e for e in dataset if e["ReviewID"] not in test]
    test_set = [e for e in dataset if e["ReviewID"] in test]
    if args.num_reviews > 0:
        test_set = test_set[:args.num_reviews]
    print(f"Reviews: {len(train_set)} train (extract) | {len(test_set)} test (steer)")

    model, tokenizer = load_model_and_tokenizer(args.model)
    num_layers = len(get_decoder_layers(model))
    layers = args.layers or list(range(0, num_layers, 4))
    stop_ids = list(get_stop_token_ids(model, tokenizer))

    # 1. Extract
    path = os.path.join(args.output_dir, "directions.pt")
    fit = torch.load(path) if os.path.exists(path) else None
    if fit is not None and set(VECTORS) <= set(fit["unit"]):
        print(f"Loaded {path}")
    else:
        fit = fit_directions(model, tokenizer, train_set, args) | {"test_reviews": sorted(test)}
        torch.save(fit, path)
        print(f"Saved {path}")
    print("\nlayer  residual norm  " + "  ".join(f"|{n}|" for n in VECTORS) + "   proj std: " + " ".join(VECTORS)
          + "   cos(framing, x): " + " ".join(VECTORS[1:]))
    for l in range(num_layers):
        cosines = [float(fit["unit"]["framing"][l] @ fit["unit"][n][l]) for n in VECTORS[1:]]
        print(f"L{l:<4} {fit['residual_norm'][l]:13.1f}  " + "  ".join(f"{fit['norms'][n][l]:{len(n) + 2}.2f}" for n in VECTORS)
              + "   " + " ".join(f"{fit['proj_std'][n][l]:.2f}" for n in VECTORS) + "   " + " ".join(f"{c:+.2f}" for c in cosines))

    scale = {n: {"unit": torch.ones(num_layers), "residual": fit["residual_norm"], "proj_std": fit["proj_std"][n]}[args.scale]
             for n in VECTORS}
    vectors = {n: fit["unit"][n] * scale[n][:, None] for n in VECTORS}
    run_dir = os.path.join(args.output_dir, args.scale + ("_gen" if args.steer_generation else ""))
    os.makedirs(run_dir, exist_ok=True)
    torch.save({"vectors": vectors, "scale": scale, "alphas": args.alphas, "layers": layers, "scale_kind": args.scale},
               os.path.join(run_dir, "vectors.pt"))
    conds = experiments(args.alphas, layers)
    print(f"\nSteering at layers {layers} | alphas {args.alphas} x unit vector"
          + {"unit": "", "residual": " x residual norm", "proj_std": " x projection std"}[args.scale]
          + f" | {sum(len(c) for c in conds.values())} steered answers per review")

    # 2. Steer
    answers_path = os.path.join(run_dir, "answers.jsonl")
    done = set()
    if os.path.exists(answers_path):
        with open(answers_path, "r", encoding="utf-8") as f:
            done = {json.loads(line)["review_id"] for line in f if line.strip()}
    print(f"{len(done)} reviews already done")

    def answer(ids):
        texts, infos = generate(model, tokenizer, ids, 1, args, stop_ids)
        return texts[0], infos[0]

    logprobs_path = os.path.join(run_dir, "logprobs.jsonl")
    with open(answers_path, "a", encoding="utf-8") as out_file, open(logprobs_path, "a", encoding="utf-8") as lp_file:
        for example in tqdm(test_set, desc="steering"):
            if example["ReviewID"] in done:
                continue
            claims = get_claims(example)
            prompts = claims | {"neutral": neutral_sentence(claims["positive"])}
            real = {p: answer(prompt_ids(tokenizer, prompts[p], NO_ROLE, model.device)) for p in PROMPTS}
            real |= {f"{r}_{f}": answer(prompt_ids(tokenizer, claims[f], r, model.device)) for r in ROLES for f in FRAMINGS}
            steered, steered_logprobs = [], {}
            for p in PROMPTS:
                ids = prompt_ids(tokenizer, prompts[p], NO_ROLE, model.device)
                results, steered_logprobs[p] = steered_answers(model, tokenizer, ids, [(alphas, l) for _, alphas, _, l in conds[p]],
                                                               vectors, args, stop_ids)
                steered += [{"experiment": e, "prompt": p, "alphas": alphas, "layer": l, "real": ref, "normal_answer": normal,
                             "answer": text, "batch": b} for (e, alphas, ref, l), (normal, text, b) in zip(conds[p], results)]
            out_file.write(json.dumps({"review_id": example["ReviewID"], "prompts": prompts,
                                       "real_answers": {k: text for k, (text, _) in real.items()}, "steered": steered}) + "\n")
            lp_file.write(json.dumps({"review_id": example["ReviewID"], "real": {k: info for k, (_, info) in real.items()},
                                      "steered": steered_logprobs}) + "\n")
            out_file.flush()
            lp_file.flush()
            gc.collect()
            torch.cuda.empty_cache()

    with open(answers_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    write_csv(records, os.path.join(run_dir, "answers.csv"))
    print(f"\nSaved {os.path.join(run_dir, 'answers.csv')} and {os.path.join(run_dir, 'vectors.pt')}")
    print("fraction of steered answers that differ from the unsteered answer:")
    keys = sorted({(s["experiment"], s["prompt"], steer_label(s["alphas"])) for r in records for s in r["steered"]})
    for e, p, label in keys:
        changed = [np.mean([s["answer"] != s["normal_answer"] for r in records for s in r["steered"]
                            if (s["experiment"], s["prompt"], steer_label(s["alphas"]), s["layer"]) == (e, p, label, l)])
                   for l in layers]
        print(f"  {e:<13} {p:<9} {label:<32} " + " ".join(f"L{l} {x:.0%}" for l, x in zip(layers, changed)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unit-length framing and role vectors, steered at the last prompt token")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--input_path", required=True, help="dataset with the framed questions (extracted question set).")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--layers", nargs="*", type=int, default=None, help="layers to steer (default: every 4th).")
    parser.add_argument("--scale", default="residual", choices=list(DEFAULT_ALPHAS),
                        help="unit: alpha x v; residual: alpha x mean residual norm x v; proj_std: alpha x std of h.v x v.")
    parser.add_argument("--alphas", nargs="+", type=float, default=None, help="push sizes (positive; default per --scale, see the docstring).")
    parser.add_argument("--steer_generation", action="store_true", help="also steer every generated token.")
    parser.add_argument("--num_reviews", default=20, type=int, help="first n test reviews to steer (-1: all).")
    parser.add_argument("--test_fraction", default=0.2, type=float)
    parser.add_argument("--max_new_tokens", default=256, type=int)
    parser.add_argument("--top_logprobs", default=5, type=int, help="most likely tokens saved at each generated token (0: none).")
    parser.add_argument("--batch_size", default=32, type=int, help="answers generated at once (one is the unsteered one).")
    parser.add_argument("--seed", default=SEED, type=int)
    args = parser.parse_args()
    args.alphas = args.alphas or DEFAULT_ALPHAS[args.scale]
    print("\nArguments:")
    for k, v in vars(args).items():
        print(f"{k:<16} {v}")
    print()
    main(args)
