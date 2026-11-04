"""
Is the role x framing interaction causal, in both directions? Steering at one layer at a time, with a random control.

Prompt and D as run_framing_role_mediation.py (yes/no prompt, D = log p(Yes) - log p(No) at the last prompt token).
Directions: that run's directions.pt (unit framing / doctor / friend / enemy vectors, fit on its training reviews) and
its held-out test reviews. Every vector is unit length and added at the last prompt token as alpha * scale * v, with
scale = the mean residual norm at that layer (--scale residual there), so all vectors have the same strength. random =
a fixed random unit vector per layer.

forward   steer framing (-v_framing) or random (-v_random) on the positive claim, with and without "My <role> said,".
          Per review: interaction = role shift steered - role shift unsteered, role shift = D(<role>) - D(none).
          R = mean interaction / mean real interaction, real = role shift under the negative claim - under the positive
          claim. R ~ 1: steering framing changes the role's effect as a real negative claim does; ~0: it does not.
reverse   steer a role vector (+v_<role>) or random (+v_random) on both claims without a role.
          Per review: change = framing effect steered - unsteered, framing effect = D(positive) - D(negative).
          R = mean change / mean real, real = framing effect with "My <role> said," - without. The random rows use the
          same real (per role) as the denominator.
          main_R = the steered D shift (averaged over the two claims) / the real role shift: a check that the role vector
          does add the role's main effect at this strength.
role_add  the reverse runs per claim (and with the mean role vector, "role"): does adding +v_<role> to "<claim>" move D
          as "My <role> said, <claim>" does? steer_effect = D(steered) - D(<claim>), real = D(My <role> said, <claim>)
          - D(<claim>) (for "role": averaged over the roles), R = steer_effect / real. random rows are compared with the
          mean role's real.

Outputs (--output_dir):
    records.jsonl   per test review: the clean D of every prompt and every steered D (resumable)
    summary.csv     one row per experiment x role x vector x layer x alpha
    role_add.csv    one row per claim x vector x layer x alpha
    R_alpha<a>.png  R by layer, per role, framing / role vector vs random
"""
import argparse
import gc
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from tqdm import tqdm

from patching_utils import DEFAULT_MODEL_NAME, SEED, set_global_seed, load_model_and_tokenizer, load_dataset
from run_role_patching import FRAMINGS, ROLES, YES_STRINGS, NO_STRINGS, get_claims, single_token_ids
from run_unit_steering import NO_ROLE, prompt_ids
from run_framing_role_mediation import ANSWER, question_for, run_rows, ratio, write_rows

PROMPT_KEYS = [f"{f}|{r}" for f in FRAMINGS for r in [NO_ROLE] + ROLES]
FORWARD_PROMPTS = [f"positive|{r}" for r in [NO_ROLE] + ROLES]
REVERSE_PROMPTS = [f"{f}|{NO_ROLE}" for f in FRAMINGS]
ADD_VECTORS = ROLES + ["role", "random"]


def steer_review(model, tokenizer, example: dict, vectors: dict, layers: list[int], args, yes_no_ids) -> dict:
    claims = get_claims(example)
    conds = [(L, a) for L in layers for a in args.alphas]
    plan = {key: [] for key in PROMPT_KEYS}   # key -> [(experiment, vector, L, a)]
    for key in FORWARD_PROMPTS:
        plan[key] += [("forward", n, L, a) for n in ["framing", "random"] for L, a in conds]
    for key in REVERSE_PROMPTS:
        plan[key] += [("reverse", n, L, a) for n in ADD_VECTORS for L, a in conds]
    sign = {"forward": -1.0, "reverse": 1.0}

    record = {"review_id": example["ReviewID"], "clean": {}, "steered": []}
    for key in PROMPT_KEYS:
        ids = prompt_ids(tokenizer, question_for(claims, key), NO_ROLE, model.device, ANSWER)
        rows = [({}, {})] + [({L: sign[e] * a * vectors[n][L]}, {}) for e, n, L, a in plan[key]]
        d, _ = run_rows(model, ids, rows, yes_no_ids, args)
        record["clean"][key] = d[0]
        record["steered"] += [{"experiment": e, "prompt": key, "vector": n, "layer": L, "alpha": a, "d": x}
                              for (e, n, L, a), x in zip(plan[key], d[1:])]
    return record


def summarize(records: list[dict], layers: list[int], alphas: list[float]) -> list[dict]:
    clean = lambda key: np.array([r["clean"][key] for r in records])
    steered = [{(s["experiment"], s["prompt"], s["vector"], s["layer"], s["alpha"]): s["d"] for s in r["steered"]}
               for r in records]
    S = lambda e, key, n, L, a: np.array([s[e, key, n, L, a] for s in steered])

    rows = []
    for role in ROLES:
        k_role, k_none = f"positive|{role}", f"positive|{NO_ROLE}"
        clean_shift = clean(k_role) - clean(k_none)
        real = (clean(f"negative|{role}") - clean(f"negative|{NO_ROLE}")) - clean_shift
        real_reverse = (clean(f"positive|{role}") - clean(f"negative|{role}")) - (clean(k_none) - clean(f"negative|{NO_ROLE}"))
        real_main = np.mean([clean(f"{f}|{role}") - clean(f"{f}|{NO_ROLE}") for f in FRAMINGS], 0)
        for L in layers:
            for a in alphas:
                for n in ["framing", "random"]:
                    inter = (S("forward", k_role, n, L, a) - S("forward", k_none, n, L, a)) - clean_shift
                    rows.append({"experiment": "forward", "role": role, "vector": n, "layer": L, "alpha": a,
                                 "change": inter.mean(), "change_sem": stats.sem(inter), "real": real.mean(),
                                 "R": ratio(inter.mean(), real.mean()), "R_sem": stats.sem(inter) / abs(real.mean()),
                                 "p": stats.ttest_1samp(inter, 0).pvalue, "main_R": np.nan,
                                 "steer_effect": (S("forward", k_none, n, L, a) - clean(k_none)).mean()})
                for n in [role, "random"]:
                    fe = S("reverse", f"positive|{NO_ROLE}", n, L, a) - S("reverse", f"negative|{NO_ROLE}", n, L, a)
                    change = fe - (clean(k_none) - clean(f"negative|{NO_ROLE}"))
                    main = np.mean([S("reverse", f"{f}|{NO_ROLE}", n, L, a) - clean(f"{f}|{NO_ROLE}") for f in FRAMINGS], 0)
                    rows.append({"experiment": "reverse", "role": role, "vector": "role" if n == role else n,
                                 "layer": L, "alpha": a, "change": change.mean(), "change_sem": stats.sem(change),
                                 "real": real_reverse.mean(), "R": ratio(change.mean(), real_reverse.mean()),
                                 "R_sem": stats.sem(change) / abs(real_reverse.mean()),
                                 "p": stats.ttest_1samp(change, 0).pvalue,
                                 "main_R": ratio(main.mean(), real_main.mean()), "steer_effect": main.mean()})
    return rows


def summarize_role_add(records: list[dict], layers: list[int], alphas: list[float]) -> list[dict]:
    clean = lambda key: np.array([r["clean"][key] for r in records])
    steered = [{(s["experiment"], s["prompt"], s["vector"], s["layer"], s["alpha"]): s["d"] for s in r["steered"]}
               for r in records]
    rows = []
    for f in FRAMINGS:
        k_none = f"{f}|{NO_ROLE}"
        real = {role: clean(f"{f}|{role}") - clean(k_none) for role in ROLES}
        real["role"] = real["random"] = np.mean([real[role] for role in ROLES], 0)
        for n in ADD_VECTORS:
            for L in layers:
                for a in alphas:
                    effect = np.array([s["reverse", k_none, n, L, a] for s in steered]) - clean(k_none)
                    rows.append({"claim": f, "vector": n, "layer": L, "alpha": a, "d_clean": clean(k_none).mean(),
                                 "steer_effect": effect.mean(), "steer_effect_sem": stats.sem(effect),
                                 "real": real[n].mean(), "R": ratio(effect.mean(), real[n].mean()),
                                 "p": stats.ttest_1samp(effect, 0).pvalue})
    return rows


def plot(rows: list[dict], alpha: float, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    titles = {"forward": "forward: steer framing −, role effect changes?",
              "reverse": "reverse: steer role +, framing effect changes?"}
    for ax, e in zip(axes, ["forward", "reverse"]):
        for i, role in enumerate(ROLES):
            color = f"C{i}"
            for n, ls in [("framing" if e == "forward" else "role", "-"), ("random", "--")]:
                r = [x for x in rows if (x["experiment"], x["role"], x["vector"], x["alpha"]) == (e, role, n, alpha)]
                ax.errorbar([x["layer"] for x in r], [x["R"] for x in r], yerr=[x["R_sem"] for x in r], color=color,
                            ls=ls, marker="o" if ls == "-" else None, ms=3, capsize=2,
                            label=f"{role} ({'random' if n == 'random' else n})")
        ax.axhline(0, color="black", lw=0.8)
        ax.axhline(1, color="gray", ls=":", lw=0.8)
        ax.set_title(titles[e])
        ax.set_xlabel(f"steer layer (α={alpha})")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("R = steered change / real change")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main(args):
    set_global_seed(SEED)
    os.makedirs(args.output_dir, exist_ok=True)
    fit = torch.load(args.directions_path)
    num_layers, d_model = fit["unit"]["framing"].shape
    layers = args.layers
    scale = fit["residual_norm"]
    generator = torch.Generator().manual_seed(args.seed)
    random = torch.randn(num_layers, d_model, generator=generator)
    units = {n: fit["unit"][n] for n in ["framing"] + ROLES + ["role"]} | {"random": random / random.norm(dim=-1, keepdim=True)}
    vectors = {n: v * scale[:, None] for n, v in units.items()}
    torch.save({"vectors": vectors, "scale": scale, "layers": layers, "alphas": args.alphas},
               os.path.join(args.output_dir, "vectors.pt"))

    test = set(fit["test_reviews"])
    test_set = [e for e in load_dataset(args.input_path) if e["ReviewID"] in test]
    if args.num_reviews > 0:
        test_set = test_set[:args.num_reviews]
    print(f"{len(test_set)} test reviews | layers {layers} | alphas {args.alphas}")

    path = os.path.join(args.output_dir, "records.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            done = {json.loads(line)["review_id"] for line in f if line.strip()}
    print(f"{len(done)} reviews already done")
    if len(done) < len(test_set):
        model, tokenizer = load_model_and_tokenizer(args.model)
        yes_no_ids = (single_token_ids(tokenizer, YES_STRINGS), single_token_ids(tokenizer, NO_STRINGS))
        with open(path, "a", encoding="utf-8") as out_file:
            for example in tqdm(test_set, desc="steer"):
                if example["ReviewID"] in done:
                    continue
                out_file.write(json.dumps(steer_review(model, tokenizer, example, vectors, layers, args, yes_no_ids)) + "\n")
                out_file.flush()
                gc.collect()
                torch.cuda.empty_cache()
    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    rows = summarize(records, layers, args.alphas)
    write_rows(rows, os.path.join(args.output_dir, "summary.csv"))
    add_rows = summarize_role_add(records, layers, args.alphas)
    write_rows(add_rows, os.path.join(args.output_dir, "role_add.csv"))
    for a in args.alphas:
        plot(rows, a, os.path.join(args.output_dir, f"R_alpha{a:g}.png"))
    a = max(args.alphas)
    print(f"\nR by layer (alpha {a:g}), steered vector vs random ({len(records)} reviews):")
    for row in rows:
        if row["alpha"] == a:
            print(f"   {row['experiment']:<8} {row['role']:<7} {row['vector']:<8} L{row['layer']:<3} R {row['R']:+.2f} "
                  f"± {row['R_sem']:.2f}  p {row['p']:.2g}" + (f"  main_R {row['main_R']:+.2f}" if row["experiment"] == "reverse" else ""))
    print(f"\n+alpha x role vector on the bare claim (alpha {a:g}): steer effect on D, real 'My <role> said,' shift, R:")
    for row in add_rows:
        if row["alpha"] == a:
            print(f"   {row['claim']:<9} {row['vector']:<7} L{row['layer']:<3} effect {row['steer_effect']:+6.2f}  "
                  f"real {row['real']:+6.2f}  R {row['R']:+.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Role x framing interaction from steering, both directions, with a random control")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--input_path", required=True, help="dataset with the framed questions (extracted question set).")
    parser.add_argument("--directions_path", required=True, help="run_framing_role_mediation.py's directions.pt.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(12, 21)))
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.25, 0.5], help="push sizes (x mean residual norm).")
    parser.add_argument("--num_reviews", default=-1, type=int, help="first n test reviews (-1: all).")
    parser.add_argument("--batch_size", default=64, type=int, help="forward passes run at once.")
    parser.add_argument("--seed", default=SEED, type=int)
    args = parser.parse_args()
    print("\nArguments:")
    for k, v in vars(args).items():
        print(f"{k:<16} {v}")
    print()
    main(args)
