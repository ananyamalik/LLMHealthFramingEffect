"""
Bilinear check for role x framing at the long-form prompt.

With s the framing (0 = negative, 1 = positive claim) and r the role (0 = bare claim, 1 = "My <role> said,"),
the 2x2 corners A00, A01, A10, A11 are fit exactly by the bilinear model
    A(s, r) = A00 + s f + r d_neg + s r I          (f = A01 - A00, d_neg = A10 - A00, I = d_pos - d_neg)
for any I, so the 2x2 cannot test it. What it assumes is that the response is a straight line in s with r fixed.
This script adds framings between (and beyond) the two endpoints and checks that line.

Prompts: as the diffmeans run (question-only full-answer prompt, "My <role> said, <claim>", residual = decoder layer
output at the last prompt token). Claim levels, from the review's own claim "<T> will work for <C>. What do you think?"
(pos and neg are exactly the cached diffmeans prompts):
    very_pos       <T> will definitely work for <C>.        somewhat_neg   <T> probably won't work for <C>.
    pos            <T> will work for <C>.                    neg            <T> won't work for <C>.
    somewhat_pos   <T> will probably work for <C>.           very_neg       <T> definitely won't work for <C>.
    neutral        <T> may or may not work for <C>.

Analysis (CPU), per role and layer, over --num_splits random halves of the reviews (fit half: f, d_neg, d_pos, I;
eval half: everything below), means over eval reviews:
    t        the level's framing coordinate in the bare prompt: (A(lv, none) - A00) . f^ / |f|  (neg 0, pos 1)
    d(lv)    the role effect at that level, A(lv, role) - A(lv, none)
    bilinear residual   |d(lv) - (d_neg + t I)| / |I|      the test: the role effect is linear in s
    baselines           additive  |d(lv) - d_mean| / |I|     (no interaction)
                        step      |d(lv) - (d_neg if t < 0.5 else d_pos)| / |I|
    noise floor         the bilinear residual at the endpoints (pos, neg) on the eval half: what sampling alone gives
    t_I      where on the I line the role effect sits: (d(lv) - d_neg) . I / |I|^2  (bilinear: t_I = t)
    bare_line          |A(lv, none) - (A00 + t f)| / |f|: the bare response off the straight framing line
                       (bilinear also assumes this is 0), with its own endpoint noise floor

Outputs (--output_dir): acts.pt (collected once, GPU), summary.json, bilinear_check.png

    sbatch ... run_bilinear_check.sh                     # collect + analyze
    python run_bilinear_check.py --output_dir outputs/olmo2-7B-instruct/bilinear_check --analyze_only
"""
import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from patching_utils import DEFAULT_MODEL_NAME, SEED, set_global_seed, load_dataset, sample_reviews, cache_residuals

ROLES = ["doctor", "friend", "enemy"]
NO_ROLE = "none"
CLAIM = re.compile(r"^(.+?) will work for (.+)\. What do you think\?$")
LEVELS = {
    "very_pos": "{t} will definitely work for {c}.",
    "pos": "{t} will work for {c}.",
    "somewhat_pos": "{t} will probably work for {c}.",
    "neutral": "{t} may or may not work for {c}.",
    "somewhat_neg": "{t} probably won't work for {c}.",
    "neg": "{t} won't work for {c}.",
    "very_neg": "{t} definitely won't work for {c}.",
}
ENDPOINTS = ("pos", "neg")
MIDDLE = [k for k in LEVELS if k not in ENDPOINTS]
ROLE_COLORS = {"doctor": "#2a78d6", "friend": "#eb6834", "enemy": "#1baf7a"}


def level_claims(positive_claim: str) -> dict[str, str]:
    m = CLAIM.match(positive_claim)
    if m is None:
        raise ValueError(f"claim does not match '<T> will work for <C>. What do you think?': {positive_claim}")
    t, c = m.groups()
    return {lv: tmpl.format(t=t, c=c) + " What do you think?" for lv, tmpl in LEVELS.items()}


@torch.no_grad()
def collect(args) -> dict:
    from patching_utils import load_model_and_tokenizer
    from run_role_patching import get_claims, encode_prompt, role_prefix
    set_global_seed(SEED)
    dataset = sample_reviews(load_dataset(args.input_path), args.num_reviews, seed=args.seed)
    model, tokenizer = load_model_and_tokenizer(args.model)
    rows, acts = [], []
    for example in tqdm(dataset, desc="reviews"):
        claims = get_claims(example)
        levels = level_claims(claims["positive"])
        assert levels["neg"] == claims["negative"], (levels["neg"], claims["negative"])
        for lv, claim in levels.items():
            for role in [NO_ROLE] + args.roles:
                question = claim if role == NO_ROLE else f"{role_prefix(role)} {claim}"
                ids = encode_prompt(tokenizer, "full", question, [], model.device, "question_only")[0]
                with cache_residuals(model, [ids.shape[1] - 1], args.layers) as cache:
                    model.model(input_ids=ids, use_cache=False)
                acts.append(torch.stack([cache[l][0].float().cpu() for l in args.layers]).half())
                rows.append({"review_id": example["ReviewID"], "level": lv, "role": role, "question": question})
    return {"rows": rows, "acts": torch.stack(acts), "layers": args.layers, "roles": args.roles,
            "levels": list(LEVELS), "model": args.model}


def cells(saved: dict) -> tuple[torch.Tensor, list[str]]:
    """[reviews, levels, 1 + roles, layers, d] float32, and the review ids."""
    idx = {(r["review_id"], r["level"], r["role"]): i for i, r in enumerate(saved["rows"])}
    roles = [NO_ROLE] + saved["roles"]
    reviews = sorted({r["review_id"] for r in saved["rows"]})
    order = [idx[(rid, lv, k)] for rid in reviews for lv in saved["levels"] for k in roles]
    x = saved["acts"][order].float()
    return x.view(len(reviews), len(saved["levels"]), len(roles), *x.shape[1:]), reviews


def analyze_split(x, levels, k, fit, ev):
    """One split, role index k (1..). Returns per level dict of [layers] tensors."""
    L = {lv: i for i, lv in enumerate(levels)}
    mf = x[fit].mean(0)            # [levels, roles, layers, d]
    me = x[ev].mean(0)
    a00, a01 = mf[L["neg"], 0], mf[L["pos"], 0]
    f = a01 - a00
    d_neg = mf[L["neg"], k] - a00
    d_pos = mf[L["pos"], k] - a01
    I = d_pos - d_neg
    fn, In = f.norm(dim=-1), I.norm(dim=-1)
    fh = f / fn[:, None]
    out = {}
    for lv in levels:
        base = me[L[lv], 0]
        d = me[L[lv], k] - base
        t = ((base - a00) * fh).sum(-1) / fn
        bil = d - (d_neg + t[:, None] * I)
        step = torch.where((t < 0.5)[:, None], d_neg, d_pos)
        out[lv] = {"t": t, "bilinear": bil.norm(dim=-1) / In,
                   "additive": (d - (d_neg + d_pos) / 2).norm(dim=-1) / In,
                   "step": (d - step).norm(dim=-1) / In,
                   "t_I": ((d - d_neg) * I).sum(-1) / In ** 2,
                   "bare_line": (base - (a00 + t[:, None] * f)).norm(dim=-1) / fn}
    return out


def analyze(saved: dict, num_splits: int, seed: int) -> dict:
    x, reviews = cells(saved)
    levels, layers = saved["levels"], saved["layers"]
    g = torch.Generator().manual_seed(seed)
    n = len(reviews)
    splits = [torch.randperm(n, generator=g) for _ in range(num_splits)]
    summary = {"model": saved.get("model"), "layers": layers, "levels": levels, "n_reviews": n, "roles": {}}
    for k, role in enumerate(saved["roles"], start=1):
        acc = {}
        for perm in splits:
            for lv, vals in analyze_split(x, levels, k, perm[: n // 2], perm[n // 2:]).items():
                for key, v in vals.items():
                    acc.setdefault(lv, {}).setdefault(key, []).append(v)
        res = {lv: {key: torch.stack(v).mean(0).tolist() for key, v in d.items()} for lv, d in acc.items()}
        floor = torch.stack(acc["pos"]["bilinear"] + acc["neg"]["bilinear"]).mean(0).tolist()
        bare_floor = torch.stack(acc["pos"]["bare_line"] + acc["neg"]["bare_line"]).mean(0).tolist()
        summary["roles"][role] = {"levels": res, "noise_floor": floor, "bare_noise_floor": bare_floor}
    return summary


def print_summary(summary: dict) -> None:
    layers = summary["layers"]
    head = "".join(f"{f'L{l}':>8}" for l in layers)
    for role, res in summary["roles"].items():
        print(f"\n=== {role} ({summary['n_reviews']} reviews) ===   residuals / |I|;  t = framing coordinate, t_I = on the I line")
        print(f"{'':<14}{'':<11}{head}")
        print(f"{'noise floor':<14}{'(pos, neg)':<11}" + "".join(f"{v:8.2f}" for v in res["noise_floor"]))
        for lv in summary["levels"]:
            r = res["levels"][lv]
            for key in ["t", "t_I", "bilinear", "additive", "step"]:
                print(f"{lv if key == 't' else '':<14}{key:<11}" + "".join(f"{v:8.2f}" for v in r[key]))
        print(f"{'bare line':<14}{'floor':<11}" + "".join(f"{v:8.2f}" for v in res["bare_noise_floor"]))
        for lv in MIDDLE:
            print(f"{'':<14}{lv[:10]:<11}" + "".join(f"{v:8.2f}" for v in res["levels"][lv]["bare_line"]))


def plot(summary: dict, layer: int, path: str) -> None:
    li = summary["layers"].index(layer)
    roles = list(summary["roles"])
    fig, axes = plt.subplots(2, len(roles), figsize=(5.4 * len(roles), 8.6), squeeze=False)
    for j, role in enumerate(roles):
        res = summary["roles"][role]
        lv_sorted = sorted(summary["levels"], key=lambda lv: res["levels"][lv]["t"][li])
        t = [res["levels"][lv]["t"][li] for lv in lv_sorted]
        ax = axes[0, j]
        for key, c, m in [("bilinear", ROLE_COLORS.get(role, "gray"), "o"), ("additive", "#8f8e89", "s"), ("step", "#e0a030", "^")]:
            ax.plot(t, [res["levels"][lv][key][li] for lv in lv_sorted], m + "-", color=c, label=key, ms=5)
        ax.axhline(res["noise_floor"][li], color="black", ls=":", label="noise floor (held-out endpoints)")
        for x, lv in zip(t, lv_sorted):
            ax.annotate(lv, (x, 0), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=7, rotation=30)
        ax.set(title=f"{role}, layer {layer}: |d(level) − prediction| / |I|", xlabel="framing coordinate t (neg 0, pos 1)")
        ax.legend(fontsize=7)

        ax = axes[1, j]
        t_I = [res["levels"][lv]["t_I"][li] for lv in lv_sorted]
        ax.plot(t, t_I, "o-", color=ROLE_COLORS.get(role, "gray"))
        lim = [min(t + t_I) - 0.1, max(t + t_I) + 0.1]
        ax.plot(lim, lim, color="gray", lw=0.8, label="bilinear: t_I = t")
        for x, y, lv in zip(t, t_I, lv_sorted):
            ax.annotate(lv, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
        ax.set(title=f"{role}: where the role effect sits on the I line", xlabel="framing coordinate t", ylabel="t_I")
        ax.legend(fontsize=7)
    fig.suptitle(f"{summary['model']} | bilinear check, layer {layer} | {summary['n_reviews']} reviews, split halves")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"Saved {path}")


def main(args):
    acts_path = os.path.join(args.output_dir, "acts.pt")
    os.makedirs(args.output_dir, exist_ok=True)
    if os.path.exists(acts_path):
        saved = torch.load(acts_path)
        print(f"Loaded {len(saved['rows'])} rows from {acts_path}")
    elif args.analyze_only:
        raise FileNotFoundError(f"{acts_path} missing: run without --analyze_only on a GPU first")
    else:
        saved = collect(args)
        torch.save(saved, acts_path)
        print(f"Saved {acts_path}")
    summary = analyze(saved, args.num_splits, args.seed)
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print_summary(summary)
    for layer in args.plot_layers:
        if layer in summary["layers"]:
            plot(summary, layer, os.path.join(args.output_dir, f"bilinear_check_L{layer}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bilinear check of role x framing with intermediate framings")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--input_path", default=None, help="dataset with the framed questions (needed to collect).")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--roles", nargs="+", default=ROLES)
    parser.add_argument("--layers", nargs="+", type=int, default=[16, 18, 20, 24, 31], help="layers to collect.")
    parser.add_argument("--num_reviews", default=-1, type=int, help="seeded sample of reviews (-1: all 368).")
    parser.add_argument("--num_splits", default=20, type=int)
    parser.add_argument("--plot_layers", nargs="+", type=int, default=[16, 20, 31])
    parser.add_argument("--analyze_only", action="store_true")
    parser.add_argument("--seed", default=SEED, type=int)
    main(parser.parse_args())
