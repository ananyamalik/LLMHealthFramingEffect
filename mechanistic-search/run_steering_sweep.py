"""
Steering sweep: how strong a push (beta) and where (layer x token site) it has to be added to move the answer.

Prompt: the bare claim in the yes/no question-only prompt (prompts/question_only_yes_no.jinja2), for a seeded sample
of --num_reviews held-out test reviews of get_diffmeans_directions.py (default 25, i.e. 50 prompts), both framings.
D = log p(Yes) - log p(No) at the last prompt token (> 0: the model says the treatment works).

Vectors (diffmeans directions, <diffmeans_dir>/<anchor>/directions.pt; beta = 1 adds the full average difference):
    framing           positive - negative claim, averaged over roles               (anchor yes_no)
    role_shared       mean over roles of (My <role> said, - no role)                (anchor yes_no)
    role_dme          doctor - enemy                                                (anchor yes_no)
    random_framing    a fixed random direction per layer with the norm of framing   (control)
    random_role       a fixed random direction per layer with the norm of role_shared (control)
    framing_matched, role_shared_matched   (--matched, the default) the same directions fit at the token being steered
                      (intervention / verdict / claim_end / question_end; the yes_no anchor for last), so the push is
                      the difference the model actually has at that token. Only at the single-token anchor sites.

Sites (token positions of the prompt the vector is added to, at one decoder layer's output):
    intervention    last token of the treatment       (both framings identical up to here)
    verdict         " work"                           (first token after will / won't)
    claim_end       "."  after the condition
    question_end    "?"  of "What do you think?"
    last            last prompt token                 (where the answer is read)
    question_span   every token of the question (start .. "?")
    after_question  every token after the question    (instruction line + chat template)
    question        question start .. end of prompt   (= --positions question of run_diffmeans_steering.py)
    all             every token except the first (BOS / attention sink)

Every (vector, site, layer, beta) is one forward pass; the passes of one layer are batched.

Outputs (--output_dir):
    prompts/<review>_<framing>.pt   per prompt: clean D, real-role D, steered D / yes-no mass / top-1 is yes-or-no
                                    as [vectors, sites, layers, betas] arrays (NaN = not run); resumable
    sweep.csv           one row per vector x site x layer x beta: mean shift of D, framing effect under the push,
                        framing recovery, |shift|, yes/no probability mass (a check that the model still answers)
    summary.json        grid, reference effects of the real prompts, best (site, layer) per vector and beta
    heatmaps/<vector>_<metric>.png   layer x beta per site
    best_by_site.png    per vector: the best layer's value against beta, one line per site
"""
import argparse
import csv
import gc
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch
from tqdm import tqdm

from patching_utils import (
    DEFAULT_MODEL_NAME, SEED, set_global_seed, load_model_and_tokenizer, get_decoder_layers, load_dataset, sample_reviews,
)
from run_role_patching import FRAMINGS, ROLES, YES_STRINGS, NO_STRINGS, role_prefix, get_claims, encode_prompt, single_token_ids
from run_belief_probe import anchor_chars

SITES = ["intervention", "verdict", "claim_end", "question_end", "last", "question_span", "after_question", "question", "all"]
# single-token site -> the diffmeans anchor fit at that token (the yes/no prompt's last token is the "yes_no" anchor)
MATCHED_ANCHOR = {"intervention": "intervention", "verdict": "verdict", "claim_end": "claim_end",
                  "question_end": "question_end", "last": "yes_no"}
BASE_VECTORS = {"framing": "framing", "role_shared": "shared", "role_dme": "doctor_minus_enemy"}   # name -> direction key
FRAMING_LIKE = {"framing", "framing_matched"}   # vectors whose push is scored as framing recovery
DEFAULT_BETAS = [-8, -4, -2, -1, -0.5, 0.5, 1, 2, 4, 8]
FORMAT = 2   # prompts/*.pt from before the layer-indexing fix (no "format") are recomputed


def vector_names(matched: bool) -> list[str]:
    names = list(BASE_VECTORS) + ["random_framing", "random_role"]
    return names + (["framing_matched", "role_shared_matched"] if matched else [])


def load_vectors(diffmeans_dir: str, anchor: str, matched: bool, seed: int) -> tuple[dict, list[str], str]:
    """vectors[name][site] -> [layers, d_model] (the same tensor for every site except the matched vectors)."""
    fit = torch.load(os.path.join(diffmeans_dir, anchor, "directions.pt"))
    dirs = fit["directions"]
    generator = torch.Generator().manual_seed(seed)

    def random_like(v):
        r = torch.randn(v.shape, generator=generator)
        return r / r.norm(dim=-1, keepdim=True) * v.norm(dim=-1, keepdim=True)

    base = {name: dirs[key] for name, key in BASE_VECTORS.items()}
    base["random_framing"] = random_like(dirs["framing"])
    base["random_role"] = random_like(dirs["shared"])
    vectors = {name: {site: v for site in SITES} for name, v in base.items()}
    if matched:
        for name, key in [("framing_matched", "framing"), ("role_shared_matched", "shared")]:
            vectors[name] = {}
            for site, a in MATCHED_ANCHOR.items():
                path = os.path.join(diffmeans_dir, a, "directions.pt")
                if os.path.exists(path):
                    other = torch.load(path)
                    if sorted(other["test_reviews"]) != sorted(fit["test_reviews"]):
                        raise ValueError(f"{path} has a different test split")
                    vectors[name][site] = other["directions"][key]
                else:
                    print(f"No directions for anchor {a}: {name} is not run at {site}")
    return vectors, fit["test_reviews"], fit.get("model")


def site_positions(encoding, question_start: int, claim: str, seq_len: int) -> dict[str, list[int]]:
    chars = anchor_chars(claim)
    tok = lambda c: encoding.char_to_token(question_start + c)
    q0, q_end = tok(0), tok(chars["question_end"])
    return {"intervention": [tok(chars["intervention"])], "verdict": [tok(chars["verdict"])],
            "claim_end": [tok(chars["claim_end"])], "question_end": [q_end], "last": [seq_len - 1],
            "question_span": list(range(q0, q_end + 1)), "after_question": list(range(q_end + 1, seq_len)),
            "question": list(range(q0, seq_len)), "all": list(range(1, seq_len))}


def yes_no_scores(model, hidden_last: torch.Tensor, yes_ids, no_ids) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per row: D, p(Yes) + p(No), and whether the top-1 token is a Yes/No variant."""
    log_probs = torch.log_softmax(model.lm_head(hidden_last).float(), dim=-1)
    lse_yes, lse_no = torch.logsumexp(log_probs[:, yes_ids], -1), torch.logsumexp(log_probs[:, no_ids], -1)
    mass = torch.logsumexp(log_probs[:, yes_ids + no_ids], -1).exp()
    top_ok = torch.isin(log_probs.argmax(-1), torch.tensor(yes_ids + no_ids, device=log_probs.device))
    return (lse_yes - lse_no).cpu().numpy(), mass.cpu().numpy(), top_ok.cpu().numpy()


@torch.no_grad()
def forward_batch(model, ids: torch.Tensor, layer: int | None, delta: torch.Tensor | None, yes_no_ids):
    """One forward of ids repeated per row of delta [rows, seq, d_model], added at decoder layer `layer`'s output."""
    rows = 1 if delta is None else delta.shape[0]
    handle = None
    if delta is not None:
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden += delta.to(hidden.dtype)
            return output
        handle = get_decoder_layers(model)[layer].register_forward_hook(hook)
    try:
        out = model.model(input_ids=ids.repeat(rows, 1), use_cache=False)
    finally:
        if handle is not None:
            handle.remove()
    return yes_no_scores(model, out.last_hidden_state[:, -1], *yes_no_ids)


def sweep_prompt(model, ids, positions, vectors, names, layers, betas, args, yes_no_ids) -> dict:
    shape = (len(names), len(SITES), len(layers), len(betas))
    d, mass, top_ok = (np.full(shape, np.nan, dtype=np.float32) for _ in range(3))
    seq_len, d_model = ids.shape[1], model.config.hidden_size
    for li, layer in enumerate(layers):
        # rows of this layer: (vector index, site index, beta index)
        rows = [(vi, si, bi) for vi, n in enumerate(names) for si, s in enumerate(SITES) if s in vectors[n]
                for bi in range(len(betas))]
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            delta = torch.zeros(len(batch), seq_len, d_model, device=model.device, dtype=model.dtype)
            for i, (vi, si, bi) in enumerate(batch):
                vec = vectors[names[vi]][SITES[si]][layer].to(model.device, model.dtype)
                delta[i, positions[SITES[si]]] = betas[bi] * vec
            ds, ms, ok = forward_batch(model, ids, layer, delta, yes_no_ids)
            for i, (vi, si, bi) in enumerate(batch):
                idx = (vi, si, li, bi)
                d[idx], mass[idx], top_ok[idx] = ds[i], ms[i], ok[i]
            del delta
    return {"d": d, "mass": mass, "top_ok": top_ok}


# Summary
def summarize(records: list[dict], names, layers, betas) -> tuple[list[dict], dict]:
    by_review = {}
    for r in records:
        by_review.setdefault(r["review_id"], {})[r["framing"]] = r
    pairs = [v for v in by_review.values() if set(v) == set(FRAMINGS)]
    pos, neg = [p["positive"] for p in pairs], [p["negative"] for p in pairs]
    allr = pos + neg
    stack = lambda rs, k: np.stack([r[k] for r in rs])                          # [n, V, S, L, B]
    clean = lambda rs: np.array([r["d_clean"] for r in rs])[:, None, None, None, None]
    D_pos, D_neg, D_all = stack(pos, "d"), stack(neg, "d"), stack(allr, "d")

    real_effect = float(np.mean(clean(pos) - clean(neg)))
    real_d = lambda role: np.array([r["d_roles"][role] for r in allr])
    reference = {
        "n_reviews": len(pairs), "real_framing_effect": real_effect,
        "real_framing_effect_by_role": {role: float(np.mean([p["d_roles"][role] - n["d_roles"][role] for p, n in zip(pos, neg)]))
                                        for role in ROLES},
        "mean_d_clean": float(clean(allr).mean()),
        "real_role_shift": {role: float(np.mean(real_d(role) - clean(allr).ravel())) for role in ROLES},
        "real_shared_shift": float(np.mean([real_d(role) - clean(allr).ravel() for role in ROLES])),
        "real_doctor_minus_enemy": float(np.mean(real_d("doctor") - real_d("enemy"))),
    }
    shift = D_all - clean(allr)
    fe = (D_pos - clean(pos)) - (D_neg - clean(neg))                              # change of the framing effect
    # framing recovery: +beta pushes the negative claim toward positive, -beta the positive toward negative
    sign = np.sign(np.array(betas))
    toward = np.where(sign > 0, (D_neg - clean(neg)), (D_pos - clean(pos)))      # broadcasts over the beta axis
    recovery = toward.mean(0) * sign / real_effect
    metrics = {"mean_shift": shift.mean(0), "mean_abs_shift": np.abs(shift).mean(0),
               "shift_positive": (D_pos - clean(pos)).mean(0), "shift_negative": (D_neg - clean(neg)).mean(0),
               "framing_effect_change": fe.mean(0), "framing_recovery": recovery,
               "yes_no_mass": stack(allr, "mass").mean(0), "top1_yes_no": stack(allr, "top_ok").mean(0)}

    rows = []
    for vi, n in enumerate(names):
        for si, s in enumerate(SITES):
            for li, l in enumerate(layers):
                for bi, b in enumerate(betas):
                    if np.isnan(metrics["mean_shift"][vi, si, li, bi]):
                        continue
                    rows.append({"vector": n, "site": s, "layer": l, "beta": b}
                                | {k: float(v[vi, si, li, bi]) for k, v in metrics.items()})
    return rows, {"reference": reference, "metrics": metrics}


def headline(name: str) -> str:
    """The metric that answers 'does this push do what the real contrast does?' for this vector."""
    if name in FRAMING_LIKE:
        return "framing_recovery"
    return "mean_abs_shift" if name.startswith("random") else "mean_shift"


def best_table(metrics: dict, names, layers, betas, min_mass: float) -> dict:
    """Per vector and beta: the (site, layer) with the largest headline value among pushes that keep p(Yes)+p(No) >=
    min_mass (so the 'best' push is not one that just breaks the model)."""
    out = {}
    for vi, n in enumerate(names):
        m, mass = metrics[headline(n)][vi], metrics["yes_no_mass"][vi]
        out[n] = {"metric": headline(n)}
        for bi, b in enumerate(betas):
            vals = np.where((mass[..., bi] >= min_mass) & ~np.isnan(m[..., bi]), np.abs(m[..., bi]), -np.inf)
            if not np.isfinite(vals).any():
                continue
            si, li = np.unravel_index(np.argmax(vals), vals.shape)
            out[n][str(b)] = {"site": SITES[si], "layer": layers[li], "value": float(m[si, li, bi]),
                              "yes_no_mass": float(mass[si, li, bi])}
    return out


def plot_heatmaps(metrics: dict, names, layers, betas, out_dir: str, title: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for vi, n in enumerate(names):
        for metric in [headline(n), "framing_effect_change", "yes_no_mass"]:
            data = metrics[metric][vi]                                       # [S, L, B]
            sites = [si for si in range(len(SITES)) if not np.isnan(data[si]).all()]
            cols = min(3, len(sites))
            rows = math.ceil(len(sites) / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 2.9 * rows), squeeze=False)
            finite = data[sites][np.isfinite(data[sites])]
            if metric == "yes_no_mass":
                kw = {"cmap": "viridis", "vmin": 0, "vmax": 1}
            else:
                lim = float(np.nanpercentile(np.abs(finite), 99)) if finite.size else 1.0
                kw = {"cmap": "RdBu_r", "norm": TwoSlopeNorm(0, -lim - 1e-6, lim + 1e-6)}
            for ax, si in zip(axes.flat, sites):
                im = ax.imshow(data[si].T, aspect="auto", origin="lower", **kw)
                ax.set_title(SITES[si], fontsize=9)
                ax.set_xticks(range(0, len(layers), max(1, len(layers) // 8)))
                ax.set_xticklabels([layers[i] for i in range(0, len(layers), max(1, len(layers) // 8))], fontsize=7)
                ax.set_yticks(range(len(betas)))
                ax.set_yticklabels([f"{b:g}" for b in betas], fontsize=7)
                ax.set_xlabel("layer", fontsize=8)
                ax.set_ylabel("beta", fontsize=8)
            for ax in list(axes.flat)[len(sites):]:
                ax.axis("off")
            fig.colorbar(im, ax=axes, shrink=0.8, label=metric)
            fig.suptitle(f"{title}\n{n}: {metric}", fontsize=10)
            fig.savefig(os.path.join(out_dir, f"{n}_{metric}.png"), dpi=130, bbox_inches="tight")
            plt.close(fig)


def plot_best_by_site(metrics: dict, names, layers, betas, min_mass: float, path: str, title: str, reference: dict) -> None:
    cols = min(4, len(names))
    rows = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.4 * rows), squeeze=False)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for ax, (vi, n) in zip(axes.flat, enumerate(names)):
        m, mass = metrics[headline(n)][vi], metrics["yes_no_mass"][vi]
        for si, s in enumerate(SITES):
            if np.isnan(m[si]).all():
                continue
            ok = np.where(mass[si] >= min_mass, m[si], np.nan)                  # [L, B]
            best = [ok[np.nanargmax(np.abs(ok[:, bi])), bi] if np.isfinite(ok[:, bi]).any() else np.nan
                    for bi in range(len(betas))]
            ax.plot(range(len(betas)), best, marker="o", ms=3, lw=1.5, color=colors[si], label=s)
        if headline(n) == "framing_recovery":
            ax.axhline(1, color="black", lw=0.8, ls=":")
        elif n in ("role_shared", "role_shared_matched"):
            ax.axhline(reference["real_shared_shift"], color="black", lw=0.8, ls=":")
        elif n == "role_dme":
            ax.axhline(reference["real_doctor_minus_enemy"] / 2, color="black", lw=0.8, ls=":")   # half: from the midpoint
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_xticks(range(len(betas)))
        ax.set_xticklabels([f"{b:g}" for b in betas])
        ax.set_title(f"{n} ({headline(n)})", fontsize=9)
        ax.set_xlabel("beta", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=6, frameon=False)
    fig.suptitle(f"{title}\nbest layer per site and beta (pushes with p(Yes)+p(No) >= {min_mass:g}); "
                 f"dotted = real contrast", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(args):
    set_global_seed(SEED)
    vectors, test_reviews, fit_model = load_vectors(args.diffmeans_dir, args.anchor, args.matched, args.seed)
    if fit_model and fit_model != args.model:
        raise ValueError(f"directions were fit on {fit_model}, not {args.model}")
    names = [n for n in (args.vectors or vector_names(args.matched)) if n in vectors]
    test = set(test_reviews)
    dataset = sample_reviews([e for e in load_dataset(args.input_path) if e["ReviewID"] in test], args.num_reviews, args.seed)
    betas = [float(b) for b in args.betas]
    print(f"{len(dataset)} sampled test reviews (of {len(test)} held out) | vectors {names}")

    os.makedirs(os.path.join(args.output_dir, "prompts"), exist_ok=True)
    with open(os.path.join(args.output_dir, "eval_reviews.json"), "w", encoding="utf-8") as f:
        json.dump([e["ReviewID"] for e in dataset], f, indent=2)

    model, tokenizer = load_model_and_tokenizer(args.model)
    num_layers = len(get_decoder_layers(model))
    layers = args.layers if args.layers else list(range(0, num_layers, args.layer_stride))
    yes_no_ids = (single_token_ids(tokenizer, YES_STRINGS), single_token_ids(tokenizer, NO_STRINGS))
    n_rows = sum(len(betas) for n in names for s in SITES if s in vectors[n])
    print(f"layers {layers} | betas {betas} | {n_rows} steered passes per layer, {n_rows * len(layers)} per prompt")

    records = []
    for example in tqdm(dataset, desc="reviews"):
        for framing, claim in get_claims(example).items():
            path = os.path.join(args.output_dir, "prompts", f"{example['ReviewID']}_{framing}.pt")
            if os.path.exists(path):
                saved = torch.load(path, weights_only=False)
                if saved.get("format") == FORMAT:
                    records.append(saved)
                    continue
                print(f"{path} is from an older version of this script: recomputing it")
            ids, encoding, question_start = encode_prompt(tokenizer, "yes_no", claim, [], model.device, "question_only")
            positions = site_positions(encoding, question_start, claim, ids.shape[1])
            d_clean = float(forward_batch(model, ids, None, None, yes_no_ids)[0][0])
            d_roles = {}
            for role in ROLES:
                role_ids = encode_prompt(tokenizer, "yes_no", f"{role_prefix(role)} {claim}", [], model.device, "question_only")[0]
                d_roles[role] = float(forward_batch(model, role_ids, None, None, yes_no_ids)[0][0])
            record = {"review_id": example["ReviewID"], "framing": framing, "question": claim, "d_clean": d_clean,
                      "d_roles": d_roles, "positions": positions,
                      "site_tokens": {s: tokenizer.decode(ids[0, p].tolist()) for s, p in positions.items()},
                      "vectors": names, "sites": SITES, "layers": layers, "betas": betas,
                      "format": FORMAT}
            record |= sweep_prompt(model, ids, positions, vectors, names, layers, betas, args, yes_no_ids)
            torch.save(record, path)
            records.append(record)
        gc.collect()
        torch.cuda.empty_cache()

    if records and (records[0]["vectors"], records[0]["layers"], records[0]["betas"]) != (names, layers, betas):
        raise ValueError("saved prompts in prompts/ were run with a different grid: use a new --output_dir")
    print("Site tokens (first prompt): " + json.dumps(records[0]["site_tokens"]))
    rows, out = summarize(records, names, layers, betas)
    write_csv(rows, os.path.join(args.output_dir, "sweep.csv"))
    best = best_table(out["metrics"], names, layers, betas, args.min_mass)
    summary = {"model": args.model, "anchor": args.anchor, "vectors": names, "sites": SITES, "layers": layers,
               "betas": betas, "min_mass": args.min_mass, "eval_reviews": [e["ReviewID"] for e in dataset],
               "reference": out["reference"], "best": best}
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    title = f"{args.model} | steering sweep, bare claim, yes/no D ({out['reference']['n_reviews']} reviews)"
    plot_heatmaps(out["metrics"], names, layers, betas, os.path.join(args.output_dir, "heatmaps"), title)
    plot_best_by_site(out["metrics"], names, layers, betas, args.min_mass,
                      os.path.join(args.output_dir, "best_by_site.png"), title, out["reference"])

    ref = out["reference"]
    print(f"\nreal framing effect {ref['real_framing_effect']:+.2f} | real shared role shift {ref['real_shared_shift']:+.2f} "
          f"| real doctor - enemy {ref['real_doctor_minus_enemy']:+.2f}")
    for n, table in best.items():
        print(f"{n} ({table['metric']}), best site/layer per beta (p(Yes)+p(No) >= {args.min_mass:g}):")
        for b in betas:
            if str(b) in table:
                t = table[str(b)]
                print(f"   beta {b:+5g}: {t['site']:<14} L{t['layer']:<3} {t['value']:+.2f}  (mass {t['yes_no_mass']:.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep steering strength (beta) x layer x token site on the bare claim")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="HuggingFace model name (must match the directions).")
    parser.add_argument("--input_path", required=True, help="dataset with the framed questions (extracted question set).")
    parser.add_argument("--diffmeans_dir", required=True, help="output dir of get_diffmeans_directions.py.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--anchor", default="yes_no", help="anchor of the (non-matched) directions.")
    parser.add_argument("--vectors", nargs="+", default=None, help=f"subset of {vector_names(True)} (default: all).")
    parser.add_argument("--no_matched", dest="matched", action="store_false", help="skip the site-matched directions.")
    parser.add_argument("--betas", nargs="+", type=float, default=DEFAULT_BETAS)
    parser.add_argument("--layers", nargs="*", type=int, default=None, help="layers to steer (default: every --layer_stride).")
    parser.add_argument("--layer_stride", default=1, type=int)
    parser.add_argument("--num_reviews", default=25, type=int, help="seeded sample of held-out test reviews (x 2 framings; -1: all).")
    parser.add_argument("--batch_size", default=128, type=int, help="steered forward passes run at once.")
    parser.add_argument("--min_mass", default=0.5, type=float, help="best-push tables/plots ignore pushes with p(Yes)+p(No) below this.")
    parser.add_argument("--seed", default=SEED, type=int, help="seed of the review sample and the random directions.")
    args = parser.parse_args()
    print("\nArguments:")
    for k, v in vars(args).items():
        print(f"{k:<16} {v}")
    print()
    main(args)
