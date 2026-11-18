"""
Does the role x framing interaction vector change the written answer? Patching test at layers 16-20.

Prompt: the question-only full-answer prompt ("Question: <claim>", prompts/question_only.jinja2), greedy answer, then
the yes/no question as a follow-up turn without any edit (as run_diffmeans_steering.py --answer full):
D = log p(Yes) - log p(No) there, i.e. what the written answer concluded (> 0: the treatment works).

Vectors: role_framing_vectors.py's fit on the training reviews (anchor `last`, the last prompt token of this prompt),
<vectors_dir>/<role>/directions.pt, per layer: inter = d_pos - d_neg, and its parts on / off the framing direction
(inter_par, inter_perp). The test reviews of the same split are used here.

Framing gap: G = D(positive) - D(negative), per review. G_role on "My <role> said, <claim>", G_none on the bare claim.
The interaction vector is, in the residual stream, G_role's state minus G_none's state (minus the role's shared
shift), so removing it from the role prompts should move G_role to G_none if it drives the answer.

Edits, at the last prompt token at one layer L (prompt pass only; --steer_generation: also every generated token).
s = -1/2 on the positive prompt, +1/2 on the negative one, so the two framings get the same role shift:
    on "My <role> said, <claim>"   (necessity)
        inter        h += s * inter               interaction removed (each framing gets the mean role shift)
        inter_par    h += s * inter_par           only its part along the framing direction
        inter_perp   h += s * inter_perp          only its part off the framing direction
        review       h += s * inter_r             this review's own interaction, from its 4 real last-token states
                                                  at L (= h(f, none) + (d_pos_r + d_neg_r) / 2: the additive
                                                  counterfactual of this very prompt)
        add_inter    h -= s * inter               interaction doubled (dose: G should move away from G_none)
        random       h += s * r                   r random, |r| = |inter| (control)
    on "<claim>"                   (sufficiency: the role's interaction without the role)
        none+inter   h -= s * inter               G_none should move toward G_role
        none+random  h -= s * r

Summary (per role, condition, layer; mean over test reviews, 95% bootstrap interval):
    gap               mean G under the edit;  gap_role, gap_none: the clean references
    recovery          (gap - gap_role) / (gap_none - gap_role)  role edits: 1 = G_role moved all the way to G_none
                      (add_inter: -1 expected)
    transfer          (gap - gap_none) / (gap_role - gap_none)  none+ edits: 1 = G_none moved all the way to G_role
    shift_pos/neg     mean change of D on the positive / negative prompt
    answer_unchanged  fraction of edited answers identical to the clean answer of the same prompt

Outputs (--output_dir/<last|last_gen>/):
    pairs.jsonl        one line per (review, prompt): clean D and answer, last-token states' check, and per
                       condition x layer the D and answer; resumable
    summary.json, interaction_patching.png
    side_by_side.csv   one row per edited answer: clean answer next to it, the bare claim's answer, D's
"""
import argparse
import csv
import gc
import json
import os
from contextlib import contextmanager

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from patching_utils import (DEFAULT_MODEL_NAME, SEED, set_global_seed, load_model_and_tokenizer, get_decoder_layers,
                            load_dataset, render_prompt, get_stop_token_ids)
from run_role_patching import FRAMINGS, ROLES, YES_STRINGS, NO_STRINGS, get_claims, single_token_ids
from run_belief_trajectory import YES_NO_FOLLOW_UP, chat_ids, yes_no_d
from run_unit_steering import NO_ROLE, prompt_ids, last_token_states

POS, NEG = FRAMINGS
ROLE_CONDS = ["inter", "inter_par", "inter_perp", "review", "add_inter", "random"]
NONE_CONDS = ["none+inter", "none+random"]
ROLE_COLORS = {"doctor": "#2a78d6", "friend": "#eb6834", "enemy": "#1baf7a"}
SIGN = {POS: -0.5, NEG: 0.5}


def load_vectors(vectors_dir: str, roles: list[str], seed: int) -> tuple[dict, list[str], str]:
    vecs, test, model = {}, None, None
    gen = torch.Generator().manual_seed(seed)
    for role in roles:
        path = os.path.join(vectors_dir, role, "directions.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing: run role_framing_vectors.py first")
        fit = torch.load(path)
        d = fit["directions"]
        r = torch.randn(d["inter"].shape, generator=gen)
        vecs[role] = {"inter": d["inter"], "inter_par": d["inter_par"], "inter_perp": d["inter_perp"],
                      "random": r / r.norm(dim=-1, keepdim=True) * d["inter"].norm(dim=-1, keepdim=True)}
        test, model = fit["test_reviews"], fit.get("model")
    return vecs, test, model


@contextmanager
def add_last_token(model, layer_vectors: dict[int, torch.Tensor], prompt_len: int, during_generation: bool):
    """Adds layer_vectors[l][row] ([batch, d]) to decoder layer l's output at the last prompt token on the prompt pass
    (and at every generated token if during_generation)."""
    def make_hook(v):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if h.shape[1] == prompt_len or during_generation:
                h[:, -1, :] += v.to(device=h.device, dtype=h.dtype)
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
def generate(model, tokenizer, ids: torch.Tensor, n: int, max_new_tokens: int, stop_ids) -> list[str]:
    """Greedy answers of n copies of the prompt ids (under any active hooks)."""
    out = model.generate(ids.repeat(n, 1), attention_mask=torch.ones_like(ids).repeat(n, 1),
                         max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                         eos_token_id=stop_ids, pad_token_id=tokenizer.eos_token_id)
    texts = []
    for row in out[:, ids.shape[1]:].tolist():
        stop = next((i for i, t in enumerate(row) if t in stop_ids), len(row))
        texts.append(tokenizer.decode(row[:stop], skip_special_tokens=True).strip())
    return texts


@torch.no_grad()
def follow_up_d(model, tokenizer, question: str, answer: str, yes_no_ids) -> float:
    first_turn = render_prompt("question_only", template_dir="../mechanistic-search/prompts", question=question, abstracts=[])
    ids = chat_ids(tokenizer, [{"role": "user", "content": first_turn}, {"role": "assistant", "content": answer},
                               {"role": "user", "content": YES_NO_FOLLOW_UP}], model.device)
    return yes_no_d(model, ids, *yes_no_ids)


@torch.no_grad()
def run_prompt(model, tokenizer, question: str, ids: torch.Tensor, edits: list[tuple[str, int, torch.Tensor]], args,
               stop_ids, yes_no_ids) -> dict:
    """Clean answer + one answer per (condition, layer, vector) edit, batched; D from the follow-up of each."""
    rows = [None] + edits
    texts = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        layer_vectors = {}
        for i, e in enumerate(batch):
            if e is None:
                continue
            _, layer, v = e
            layer_vectors.setdefault(layer, torch.zeros(len(batch), v.shape[-1]))[i] = v
        with add_last_token(model, layer_vectors, ids.shape[1], args.steer_generation):
            texts += generate(model, tokenizer, ids, len(batch), args.max_new_tokens, stop_ids)
    ds = [follow_up_d(model, tokenizer, question, t, yes_no_ids) for t in texts]
    out = {"question": question, "d_clean": ds[0], "clean_answer": texts[0], "d": {}, "answers": {}}
    for (cond, layer, _), t, d in zip(edits, texts[1:], ds[1:]):
        out["d"].setdefault(cond, {})[str(layer)] = d
        out["answers"].setdefault(cond, {})[str(layer)] = t
    return out


def run_review(model, tokenizer, example, vecs, args, stop_ids, yes_no_ids, done) -> list[dict]:
    claims = get_claims(example)
    rid = example["ReviewID"]
    ids = {(f, k): prompt_ids(tokenizer, claims[f], k, model.device) for f in FRAMINGS for k in [NO_ROLE] + args.roles}
    states = {key: last_token_states(model, v) for key, v in ids.items()}   # [layers, d] float32
    records = []
    for f in FRAMINGS:
        s = SIGN[f]
        for k in [NO_ROLE] + args.roles:
            key = f"{rid}|{f}|{k}"
            if key in done:
                continue
            edits = []
            for l in args.layers:
                if k == NO_ROLE:
                    for role in args.roles:
                        edits += [(f"{c}|{role}", l, -s * vecs[role][c.split('+')[1]][l]) for c in NONE_CONDS]
                else:
                    inter_r = (states[(POS, k)][l] - states[(POS, NO_ROLE)][l]) - (states[(NEG, k)][l] - states[(NEG, NO_ROLE)][l])
                    for c in ROLE_CONDS:
                        if c == "review":
                            v = inter_r
                        elif c == "add_inter":
                            v = -vecs[k]["inter"][l]
                        else:
                            v = vecs[k][c][l]
                        edits.append((c, l, s * v))
            question = claims[f] if k == NO_ROLE else f"My {k} said, {claims[f]}"
            rec = run_prompt(model, tokenizer, question, ids[(f, k)], edits, args, stop_ids, yes_no_ids)
            if k != NO_ROLE:   # how much of the mean interaction this review's own one points along (check)
                rec["inter_r_cos"] = {str(l): float(torch.nn.functional.cosine_similarity(
                    (states[(POS, k)][l] - states[(POS, NO_ROLE)][l]) - (states[(NEG, k)][l] - states[(NEG, NO_ROLE)][l]),
                    vecs[k]["inter"][l], dim=0)) for l in args.layers}
            records.append({"key": key, "review_id": rid, "framing": f, "role": k} | rec)
    return records


def boot_ci(x: np.ndarray, rng, n: int = 1000) -> list[float]:
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def summarize(records: list[dict], roles: list[str], layers: list[int], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    by = {(r["review_id"], r["framing"], r["role"]): r for r in records}
    reviews = sorted({r["review_id"] for r in records
                      if all((r["review_id"], f, k) in by for f in FRAMINGS for k in [NO_ROLE] + roles)})
    out = {"n_reviews": len(reviews), "roles": {}}

    def gap(k, cond=None, l=None):
        def d(rid, f):
            rec = by[(rid, f, k)]
            return rec["d_clean"] if cond is None else rec["d"][cond][str(l)]
        return np.array([d(rid, POS) - d(rid, NEG) for rid in reviews])

    def shift(k, f, cond, l):
        return np.array([by[(rid, f, k)]["d"][cond][str(l)] - by[(rid, f, k)]["d_clean"] for rid in reviews])

    def unchanged(k, cond, l):
        return float(np.mean([by[(rid, f, k)]["answers"][cond][str(l)] == by[(rid, f, k)]["clean_answer"]
                              for rid in reviews for f in FRAMINGS]))

    g_none = gap(NO_ROLE)
    for role in roles:
        g_role = gap(role)
        den = g_none.mean() - g_role.mean()
        res = {"gap_role": float(g_role.mean()), "gap_role_ci": boot_ci(g_role, rng),
               "gap_none": float(g_none.mean()), "gap_none_ci": boot_ci(g_none, rng),
               "behavioral_interaction": float(-den), "behavioral_interaction_ci": boot_ci(g_role - g_none, rng),
               "conditions": {}}
        for cond in ROLE_CONDS + [f"{c}|{role}" for c in NONE_CONDS]:
            k = NO_ROLE if cond.startswith("none+") else role
            rows = []
            for l in layers:
                g = gap(k, cond, l)
                # per-review ratio is unstable; the ratio of means, with a paired bootstrap over reviews
                if k == role:
                    num, ref = g - g_role, g_none - g_role
                else:
                    num, ref = g - g_none, g_role - g_none
                ix = rng.integers(0, len(reviews), (1000, len(reviews)))
                ratios = num[ix].mean(1) / ref[ix].mean(1)
                rows.append({"layer": l, "gap": float(g.mean()), "gap_ci": boot_ci(g, rng),
                             ("recovery" if k == role else "transfer"): float(num.mean() / ref.mean()),
                             "ratio_ci": [float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))],
                             "shift_pos": float(shift(k, POS, cond, l).mean()),
                             "shift_neg": float(shift(k, NEG, cond, l).mean()),
                             "answer_unchanged": unchanged(k, cond, l)})
            res["conditions"][cond] = rows
        out["roles"][role] = res
    return out


def plot(summary: dict, layers: list[int], title: str, path: str) -> None:
    roles = list(summary["roles"])
    fig, axes = plt.subplots(2, len(roles), figsize=(5.5 * len(roles), 9), squeeze=False)
    colors = {"inter": "#b8457a", "inter_par": "#d98bb4", "inter_perp": "#6b1f47", "review": "#0b0b0b",
              "add_inter": "#e0a030", "random": "#8f8e89"}
    for j, role in enumerate(roles):
        res = summary["roles"][role]
        ax = axes[0, j]
        for c in ROLE_CONDS:
            rows = res["conditions"][c]
            ax.plot(layers, [r["gap"] for r in rows], "o-", color=colors[c], label=c, ms=4)
            lo, hi = zip(*[r["gap_ci"] for r in rows])
            ax.fill_between(layers, lo, hi, color=colors[c], alpha=0.1)
        ax.axhline(res["gap_role"], color=ROLE_COLORS.get(role, "gray"), ls="--", label=f"clean {role} (G_role)")
        ax.axhline(res["gap_none"], color="black", ls=":", label="clean bare claim (G_none)")
        ax.set(title=f"{role}: framing gap D(pos) - D(neg), role prompts edited", xlabel="layer edited", ylabel="gap")
        ax.legend(fontsize=7)

        ax = axes[1, j]
        for c in ROLE_CONDS:
            rows = res["conditions"][c]
            ax.plot(layers, [r["recovery"] for r in rows], "o-", color=colors[c], label=f"{c} (recovery)", ms=4)
        for c, ls in [("none+inter", "-"), ("none+random", ":")]:
            rows = res["conditions"][f"{c}|{role}"]
            ax.plot(layers, [r["transfer"] for r in rows], "s" + ls, color="#1baf7a", label=f"{c} (transfer)", ms=4)
        ax.axhline(1, color="gray", lw=0.8)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set(title=f"{role}: recovery (role prompts) / transfer (bare claim)", xlabel="layer edited",
               ylim=(-1.6, 1.6))
        ax.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"Saved {path}")


def write_side_by_side(records: list[dict], path: str) -> None:
    by = {(r["review_id"], r["framing"], r["role"]): r for r in records}
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["review_id", "framing", "role", "condition", "layer", "question", "d_clean", "d_edited",
                    "identical_to_clean", "clean_answer", "edited_answer", "bare_claim_answer", "d_bare_claim"])
        for r in records:
            bare = by.get((r["review_id"], r["framing"], NO_ROLE))
            for cond, per_layer in r["answers"].items():
                for l, ans in per_layer.items():
                    w.writerow([r["review_id"], r["framing"], r["role"], cond, l, r["question"],
                                f"{r['d_clean']:.3f}", f"{r['d'][cond][l]:.3f}", ans == r["clean_answer"],
                                r["clean_answer"], ans, bare["clean_answer"] if bare else "",
                                f"{bare['d_clean']:.3f}" if bare else ""])
    print(f"Saved {path}")


def main(args):
    set_global_seed(SEED)
    vecs, test_reviews, fit_model = load_vectors(args.vectors_dir, args.roles, args.seed)
    if fit_model and fit_model != args.model:
        raise ValueError(f"vectors were fit on {fit_model}, not {args.model}")
    test = set(test_reviews)
    dataset = [e for e in load_dataset(args.input_path) if e["ReviewID"] in test]
    if args.num_reviews > 0:
        dataset = dataset[:args.num_reviews]
    print(f"Test reviews: {len(dataset)} (of {len(test)} held out) | layers {args.layers} | roles {args.roles}")

    model, tokenizer = load_model_and_tokenizer(args.model)
    stop_ids = get_stop_token_ids(model, tokenizer)
    yes_no_ids = (single_token_ids(tokenizer, YES_STRINGS), single_token_ids(tokenizer, NO_STRINGS))

    out_dir = os.path.join(args.output_dir, "last_gen" if args.steer_generation else "last")
    os.makedirs(out_dir, exist_ok=True)
    pairs_path = os.path.join(out_dir, "pairs.jsonl")
    done = set()
    if os.path.exists(pairs_path):
        with open(pairs_path, encoding="utf-8") as f:
            done = {json.loads(line)["key"] for line in f if line.strip()}
    print(f"{len(done)} prompts already done")

    with open(pairs_path, "a", encoding="utf-8") as out_file:
        for example in tqdm(dataset, desc="reviews"):
            for rec in run_review(model, tokenizer, example, vecs, args, stop_ids, yes_no_ids, done):
                out_file.write(json.dumps(rec) + "\n")
                out_file.flush()
            gc.collect()
            torch.cuda.empty_cache()

    with open(pairs_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    summary = summarize(records, args.roles, args.layers, args.seed) | {
        "model": args.model, "layers": args.layers, "steer_generation": args.steer_generation}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_side_by_side(records, os.path.join(out_dir, "side_by_side.csv"))

    print(f"\n{summary['n_reviews']} reviews")
    for role, res in summary["roles"].items():
        print(f"[{role}] G_role {res['gap_role']:+.2f}  G_none {res['gap_none']:+.2f}  "
              f"behavioral interaction {res['behavioral_interaction']:+.2f} {res['behavioral_interaction_ci']}")
        for cond, rows in res["conditions"].items():
            key = "transfer" if cond.startswith("none+") else "recovery"
            print(f"   {cond:<20} " + "  ".join(f"L{r['layer']} {r[key]:+.2f} (unch {r['answer_unchanged']:.0%})" for r in rows))
    added = "last prompt token" + (" + every generated token" if args.steer_generation else "")
    plot(summary, args.layers, f"{args.model} | interaction patching, written answer (D from yes/no follow-up) | at {added}",
         os.path.join(out_dir, "interaction_patching.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patching test of the role x framing interaction vector on full answers")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--input_path", required=True, help="dataset with the framed questions (extracted question set).")
    parser.add_argument("--vectors_dir", required=True, help="role_framing_vectors.py output: <diffmeans_dir>/last/role_framing_vectors")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--roles", nargs="+", default=ROLES)
    parser.add_argument("--layers", nargs="+", type=int, default=[16, 17, 18, 19, 20])
    parser.add_argument("--num_reviews", default=30, type=int, help="first n test reviews (-1: all 74).")
    parser.add_argument("--max_new_tokens", default=256, type=int)
    parser.add_argument("--batch_size", default=16, type=int, help="answers generated together (same prompt).")
    parser.add_argument("--steer_generation", action="store_true", help="also add the edit at every generated token (outputs in last_gen/).")
    parser.add_argument("--seed", default=SEED, type=int)
    main(parser.parse_args())
