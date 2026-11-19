"""
Where is the verdict read from? A probe for D at the last prompt token, compared with the framing and interaction
vectors. CPU only, on results that already exist.

Data, one row per (review, framing, condition), condition = bare claim / My doctor / friend / enemy said:
    x   the last-token residual of the question-only prompt at every layer (<diffmeans_dir>/activations.pt, anchor
        `last`; the same prompts the patching and full-answer steering runs used)
    D   --target long:   the written answer's verdict, D = log p(Yes) - log p(No) of the yes/no follow-up after the
                         greedy answer. From the clean answers of run_interaction_patching.py (--patching_pairs,
                         preferred) and run_diffmeans_steering.py --answer full (d_clean, d_roles; --steering_glob).
                         ~25 test reviews.
        --target yes_no: the single-token yes/no verdict of the same claim (rows["d"] in activations.pt), all reviews.

Probe: ridge regression of D on x per layer (centered, unstandardized, so the weight vector w lives in the residual
space the vectors do; penalty by generalized cross-validation), out-of-fold predictions with folds grouped by review
(--target long: leave one review out). Three fits:
    pooled   every row. A 4096-d probe can learn the 8 cell means of D, so it reproduces the gap table almost by
             construction: an upper bound
    within   x and D centered within each (condition, framing) cell before fitting: w only sees variation across
             reviews. Cell differences in its predictions (pred = w . (x - mean x) + mean D) come through w alone,
             so this is the test of whether the review-level verdict direction also explains the role differences
    bare     bare-claim rows only, applied to every row: does the bare claim's verdict direction carry the roles?

Per fit and layer:
    r, r_within, r_cell    held-out correlation of predicted and actual D: all rows / after removing cell means /
                           of the 8 cell means
    gaps                   predicted framing gap D(pos) - D(neg) per condition (held-out predictions), next to the
                           actual one (+4.79 / -0.67 / -1.92 / +1.71 for doctor / friend / enemy / bare in the patching
                           run), and the predicted interaction = gap(role) - gap(bare)
    cells                  predicted and actual mean D per (condition, framing); proj: mean x . w^ per cell
    geometry (w fit on all rows)
        cos(w, f), cos(w, I_role)
        span_fI[role]      |P w| / |w|, P = projection on span(f, I_role); span_all: span(f, I_doctor, I_friend, I_enemy)
                           (random direction: about sqrt(k / d_model), 0.02-0.03)
        w.f, w.I_role      the change in D the probe predicts for moving by f / by I_role. f = A01 - A00, so w.f
                           should match the bare framing gap, and w.I_role the behavioral interaction, if the vectors
                           are what the verdict is read from
Reading: low overlap and w.I far from the behavioral interaction -> the verdict is read from something else (e.g. an
"agree with the speaker" direction); high overlap and w.I close to it -> the vectors are behaviorally relevant and the
patching null points to routing or a weak edit.

Outputs (--output_dir): summary_<target>.json, probe_<target>.png
    python probe_verdict_direction.py --target long yes_no
    python probe_verdict_direction.py --self_test        # synthetic data with a planted readout: checks the pipeline
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

FRAMINGS = ["positive", "negative"]
CONDS = ["none", "doctor", "friend", "enemy"]
ROLES = CONDS[1:]
FITS = ["pooled", "within", "bare"]
COLORS = {"none": "#0b0b0b", "doctor": "#2a78d6", "friend": "#eb6834", "enemy": "#1baf7a",
          "pooled": "#8f8e89", "within": "#b8457a", "bare": "#0b0b0b"}


# ----------------------------------------------------------------------------------------------------- data
def load_long_d(patching_pairs: str | None, steering_glob: str | None) -> dict:
    """{(review, framing, cond): D} of the written answers; the patching run's value is preferred."""
    out = {}
    if steering_glob:
        for path in sorted(glob.glob(steering_glob)):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    if "d_roles" not in r:
                        continue
                    for cond, d in [("none", r["d_clean"])] + list(r["d_roles"].items()):
                        out.setdefault((r["review_id"], r["framing"], cond), d)
    if patching_pairs and os.path.exists(patching_pairs):
        with open(patching_pairs, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                out[(r["review_id"], r["framing"], r["role"])] = r["d_clean"]
    return out


def build(saved: dict, target: str, long_d: dict, layers: list[int]):
    """X [rows, layers, d] float32, D [rows], meta rows (review, framing, cond), only complete reviews."""
    rows = saved["rows"]
    idx = {(r["review_id"], r["framing"], r["role"]): i for i, r in enumerate(rows)}
    if target == "long":
        dval = long_d
    else:
        dval = {(r["review_id"], r["framing"], r["role"]): r["d"] for r in rows}
    reviews = sorted({k[0] for k in dval if all((k[0], f, c) in dval and (k[0], f, c) in idx
                                                for f in FRAMINGS for c in CONDS)})
    keys = [(rid, f, c) for rid in reviews for f in FRAMINGS for c in CONDS]
    x = saved["acts"]["last"][[idx[k] for k in keys]][:, layers].float()
    return x, torch.tensor([dval[k] for k in keys], dtype=torch.float64), keys


# ----------------------------------------------------------------------------------------------------- ridge
def ridge_fit(x: torch.Tensor, y: torch.Tensor, alphas: np.ndarray):
    """Dual ridge, centered; penalty by GCV. Returns (w [d], x_mean, y_mean, alpha)."""
    xm, ym = x.mean(0), y.mean()
    xc, yc = x - xm, y - ym
    lam, u = torch.linalg.eigh(xc @ xc.T)
    lam = lam.clamp_min(0)
    uy = u.T @ yc
    scale = float(lam.mean())
    best = None
    n = len(y)
    for a in alphas * scale:
        h = lam / (lam + a)
        resid = yc - u @ (h * uy)
        gcv = float(n * (resid ** 2).sum() / (n - h.sum()) ** 2)
        if best is None or gcv < best[0]:
            best = (gcv, a)
    a = best[1]
    w = xc.T @ (u @ (uy / (lam + a)))
    return w, xm, ym, float(a / scale)


def fit_variant(x, y, cells, mask_fit, kind, alphas):
    """Fit on rows mask_fit. within: center x and y by cell (on the fitting rows) first."""
    xf, yf = x[mask_fit], y[mask_fit]
    if kind == "within":
        cf = cells[mask_fit]
        xf, yf = xf.clone(), yf.clone()
        for c in np.unique(cf):
            m = torch.from_numpy(cf == c)
            xf[m] -= xf[m].mean(0)
            yf[m] -= yf[m].mean()
        w, _, _, a = ridge_fit(xf, yf, alphas)
        return w, x[mask_fit].mean(0), y[mask_fit].mean(), a   # intercept from the uncentered fitting rows
    return ridge_fit(xf, yf, alphas)


def oof_predict(x, y, groups, cells, conds, kind, alphas, n_folds):
    uniq = np.unique(groups)
    folds = np.array_split(np.random.default_rng(0).permutation(uniq), min(n_folds, len(uniq)))
    pred = torch.zeros_like(y)
    for test_groups in folds:
        te = np.isin(groups, test_groups)
        fit_rows = ~te & (conds == "none") if kind == "bare" else ~te
        w, xm, ym, _ = fit_variant(x, y, cells, fit_rows, kind, alphas)
        pred[torch.from_numpy(te)] = (x[torch.from_numpy(te)] - xm) @ w + ym
    return pred


# ----------------------------------------------------------------------------------------------------- metrics
def corr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def cell_table(values, keys) -> dict:
    t = {}
    for v, (_, f, c) in zip(values, keys):
        t.setdefault(f"{c}|{f}", []).append(float(v))
    return {k: float(np.mean(v)) for k, v in t.items()}


def gaps(values, keys) -> dict:
    t = cell_table(values, keys)
    g = {c: t[f"{c}|positive"] - t[f"{c}|negative"] for c in CONDS}
    return g | {f"interaction_{r}": g[r] - g["none"] for r in ROLES}


def unit(v):
    return v / v.norm().clamp_min(1e-12)


def span_frac(w, vecs) -> float:
    q, _ = torch.linalg.qr(torch.stack(vecs).T)
    return float((q.T @ w).norm() / w.norm())


def analyze_layer(x, y, keys, vec_l, alphas, n_folds):
    x, y = x.double(), y.double()
    groups = np.array([k[0] for k in keys])
    conds = np.array([k[2] for k in keys])
    cells = np.array([f"{k[2]}|{k[1]}" for k in keys])
    actual = gaps(y.numpy(), keys)
    out = {"actual_gaps": actual, "actual_cells": cell_table(y.numpy(), keys), "fits": {}}
    cell_mean_y = cell_table(y.numpy(), keys)
    for kind in FITS:
        pred = oof_predict(x, y, groups, cells, conds, kind, alphas, n_folds).numpy()
        cm_pred = cell_table(pred, keys)
        resid_p = pred - np.array([cm_pred[c] for c in cells])
        resid_y = y.numpy() - np.array([cell_mean_y[c] for c in cells])
        fit_rows = (conds == "none") if kind == "bare" else np.ones(len(y), bool)
        w, _, _, alpha = fit_variant(x, y, cells, fit_rows, kind, alphas)
        wu = unit(w.float())
        f, I = vec_l["f"], vec_l["inter"]
        geo = {"cos_f": float(wu @ unit(f)),
               **{f"cos_I_{r}": float(wu @ unit(I[r])) for r in ROLES},
               **{f"span_fI_{r}": span_frac(wu, [f, I[r]]) for r in ROLES},
               "span_all": span_frac(wu, [f] + [I[r] for r in ROLES]),
               "w_dot_f": float(w.float() @ f), **{f"w_dot_I_{r}": float(w.float() @ I[r]) for r in ROLES}}
        proj = cell_table((x.float() @ wu).numpy(), keys)
        out["fits"][kind] = {"r": corr(pred, y.numpy()), "r_within": corr(resid_p, resid_y),
                             "r_cell": corr([cm_pred[c] for c in sorted(cm_pred)], [cell_mean_y[c] for c in sorted(cm_pred)]),
                             "gaps": gaps(pred, keys), "cells": cm_pred, "proj": proj, "alpha": alpha, **geo}
    return out


# ----------------------------------------------------------------------------------------------------- report
def print_layer(target, layer, s, n_reviews):
    a = s["actual_gaps"]
    print(f"\n[{target}] layer {layer} ({n_reviews} reviews) | actual gap: "
          + "  ".join(f"{c} {a[c]:+.2f}" for c in CONDS)
          + " | interaction: " + "  ".join(f"{r} {a[f'interaction_{r}']:+.2f}" for r in ROLES))
    for kind, q in s["fits"].items():
        g = q["gaps"]
        print(f"  {kind:<7} r {q['r']:+.2f} r_within {q['r_within']:+.2f} r_cell {q['r_cell']:+.2f} | pred gap "
              + " ".join(f"{c} {g[c]:+.2f}" for c in CONDS)
              + f" | cos f {q['cos_f']:+.2f} I " + "/".join(f"{q[f'cos_I_{r}']:+.2f}" for r in ROLES)
              + f" | span(f,I) " + "/".join(f"{q[f'span_fI_{r}']:.2f}" for r in ROLES) + f" all {q['span_all']:.2f}"
              + f" | w.f {q['w_dot_f']:+.2f} w.I " + "/".join(f"{q[f'w_dot_I_{r}']:+.2f}" for r in ROLES))


def plot(target, layers, summaries, n_reviews, d_model, path, bar_layer):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    L = layers
    ax = axes[0, 0]
    for kind in FITS:
        q = [s["fits"][kind] for s in summaries]
        ax.plot(L, [v["r"] for v in q], "-", color=COLORS[kind], label=f"{kind}: r")
        ax.plot(L, [v["r_within"] for v in q], "--", color=COLORS[kind], label=f"{kind}: r within cell")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set(title="Held-out fit of D", xlabel="layer", ylim=(-0.3, 1.02))
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    for kind, ls in [("pooled", ":"), ("within", "-"), ("bare", "--")]:
        q = [s["fits"][kind] for s in summaries]
        ax.plot(L, [v["cos_f"] for v in q], ls, color="#0b0b0b", label=f"{kind}: cos(w, f)")
        for r in ROLES:
            ax.plot(L, [v[f"cos_I_{r}"] for v in q], ls, color=COLORS[r], label=f"{kind}: cos(w, I_{r})" if kind == "within" else None)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set(title="Probe direction vs f and I (dotted pooled, solid within, dashed bare)", xlabel="layer", ylim=(-1, 1))
    ax.legend(fontsize=6.5)

    ax = axes[0, 2]
    for kind, ls in [("pooled", ":"), ("within", "-"), ("bare", "--")]:
        q = [s["fits"][kind] for s in summaries]
        ax.plot(L, [v["span_all"] for v in q], ls, color="#b8457a", label=f"{kind}: span(f, I_doc, I_fri, I_ene)")
    ax.axhline(np.sqrt(4 / d_model), color="gray", ls=":", label="random direction")
    ax.set(title="|projection of w on span(f, I's)| / |w|", xlabel="layer", ylim=(0, 1))
    ax.legend(fontsize=7)

    for ax, kind in zip([axes[1, 0], axes[1, 1]], ["within", "pooled"]):
        actual = [s["actual_gaps"] for s in summaries]
        q = [s["fits"][kind] for s in summaries]
        ax.plot(L, [v["w_dot_f"] for v in q], "-", color="#0b0b0b", label="w·f (pred. bare framing gap)")
        ax.axhline(actual[0]["none"], color="#0b0b0b", ls=":", lw=1, label="actual bare gap")
        for r in ROLES:
            ax.plot(L, [v[f"w_dot_I_{r}"] for v in q], "-", color=COLORS[r], label=f"w·I_{r}")
            ax.axhline(actual[0][f"interaction_{r}"], color=COLORS[r], ls=":", lw=1)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set(title=f"{kind}: D change the probe predicts along f / I (dotted: actual)", xlabel="layer")
        ax.legend(fontsize=7)

    ax = axes[1, 2]
    li = layers.index(bar_layer)
    s = summaries[li]
    xs = np.arange(len(CONDS))
    width = 0.2
    ax.bar(xs - 1.5 * width, [s["actual_gaps"][c] for c in CONDS], width, color="#c9c8c3", label="actual")
    for i, kind in enumerate(FITS):
        ax.bar(xs + (i - 0.5) * width, [s["fits"][kind]["gaps"][c] for c in CONDS], width, color=COLORS[kind], label=kind)
    ax.set_xticks(xs, ["bare", "doctor", "friend", "enemy"])
    ax.axhline(0, color="gray", lw=0.8)
    ax.set(title=f"Framing gap D(pos) − D(neg), held-out predictions, layer {bar_layer}")
    ax.legend(fontsize=7)
    fig.suptitle(f"Verdict probe at the last prompt token | target: {target} | {n_reviews} reviews, folds grouped by review")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"Saved {path}")


# ----------------------------------------------------------------------------------------------------- main
def load_vectors(vectors_dir, layers):
    vec = {}
    for r in ROLES:
        d = torch.load(os.path.join(vectors_dir, r, "directions.pt"))["directions"]
        vec["f"] = d["f"][layers].float()
        vec.setdefault("inter", {})[r] = d["inter"][layers].float()
    return [{"f": vec["f"][i], "inter": {r: vec["inter"][r][i] for r in ROLES}} for i in range(len(layers))]


def run(saved, target, long_d, vectors, layers, args):
    x, y, keys = build(saved, target, long_d, layers)
    n_reviews = len({k[0] for k in keys})
    n_folds = n_reviews if target == "long" else args.n_folds
    print(f"\n=== target {target}: {len(keys)} rows, {n_reviews} reviews, {n_folds} folds ===")
    summaries = []
    for i, layer in enumerate(layers):
        s = analyze_layer(x[:, i], y, keys, vectors[i], np.array(args.alphas), n_folds)
        summaries.append(s)
        if layer in args.print_layers:
            print_layer(target, layer, s, n_reviews)
    return summaries, n_reviews, x.shape[-1]


def self_test():
    """Planted readout: D = w* . x. The within / pooled fits must recover w*, and the gap table must follow from it."""
    g = torch.Generator().manual_seed(0)
    d, n_rev = 48, 40
    f, w_star = torch.randn(d, generator=g), torch.randn(d, generator=g)
    I = {r: torch.randn(d, generator=g) for r in ROLES}
    role_shift = {r: torch.randn(d, generator=g) for r in ROLES}
    rows, acts, dv = [], [], {}
    for i in range(n_rev):
        content = torch.randn(d, generator=g) * 2
        for fr in FRAMINGS:
            s = 1.0 if fr == "positive" else 0.0
            for c in CONDS:
                x = content + s * f + (0 if c == "none" else role_shift[c] + s * I[c]) + 0.05 * torch.randn(d, generator=g)
                rows.append({"review_id": f"R{i}", "framing": fr, "role": c})
                acts.append(x)
                dv[(f"R{i}", fr, c)] = float(w_star @ x)
    saved = {"rows": rows, "acts": {"last": torch.stack(acts)[:, None]}}
    x, y, keys = build(saved, "long", dv, [0])
    s = analyze_layer(x[:, 0], y, keys, {"f": f, "inter": I}, np.logspace(-6, 2, 9), n_rev)
    w_true = {"w_dot_f": float(w_star @ f), **{f"w_dot_I_{r}": float(w_star @ I[r]) for r in ROLES}}
    for kind in ["pooled", "within"]:
        q = s["fits"][kind]
        assert q["r"] > 0.99, (kind, q["r"])
        for key, v in w_true.items():
            assert abs(q[key] - v) < 0.05 * abs(v) + 0.1, (kind, key, q[key], v)
        for c in CONDS:
            assert abs(q["gaps"][c] - s["actual_gaps"][c]) < 0.05 * abs(s["actual_gaps"][c]) + 0.2, (kind, c)
    # span: w* in span(f, I's) -> 1; a random w far from it
    assert span_frac(unit(f + I["doctor"]), [f] + [I[r] for r in ROLES]) > 0.999
    assert span_frac(unit(torch.randn(d, generator=g)), [f] + [I[r] for r in ROLES]) < 0.8

    # Opposite case: the readout u is orthogonal to f and every I -> low overlap, w.f and w.I ~ 0
    q, _ = torch.linalg.qr(torch.stack([f] + [I[r] for r in ROLES]).T)
    u = w_star - q @ (q.T @ w_star)
    dv_u = {k: float(u @ saved["acts"]["last"][i, 0]) for i, k in
            enumerate((r["review_id"], r["framing"], r["role"]) for r in rows)}
    x, y, keys = build(saved, "long", dv_u, [0])
    s = analyze_layer(x[:, 0], y, keys, {"f": f, "inter": I}, np.logspace(-6, 2, 9), n_rev)
    q_w = s["fits"]["within"]
    assert q_w["span_all"] < 0.05, q_w["span_all"]
    assert abs(q_w["w_dot_f"]) < 0.1 and all(abs(q_w[f"w_dot_I_{r}"]) < 0.1 for r in ROLES), q_w
    print("self test passed: planted readout recovered (r, w.f, w.I, gap table) by pooled and within fits; "
          "an orthogonal readout gives span ~0 and w.f, w.I ~0")


def main(args):
    if args.self_test:
        self_test()
        return
    saved = torch.load(args.acts_path, mmap=True)
    layers = args.layers or list(range(saved["acts"]["last"].shape[1]))
    vectors = load_vectors(args.vectors_dir, layers)
    long_d = load_long_d(args.patching_pairs, args.steering_glob)
    os.makedirs(args.output_dir, exist_ok=True)
    for target in args.target:
        summaries, n_reviews, d_model = run(saved, target, long_d, vectors, layers, args)
        plot(target, layers, summaries, n_reviews, d_model, os.path.join(args.output_dir, f"probe_{target}.png"),
             args.bar_layer if args.bar_layer in layers else layers[-1])
        path = os.path.join(args.output_dir, f"summary_{target}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"target": target, "model": saved.get("model"), "n_reviews": n_reviews, "layers": layers,
                       "per_layer": summaries}, f, indent=2)
        print(f"Saved {path}")


if __name__ == "__main__":
    base = "outputs/olmo2-7B-instruct"
    parser = argparse.ArgumentParser(description="Probe for the verdict D vs the framing and interaction vectors")
    parser.add_argument("--acts_path", default=f"{base}/diffmeans/activations.pt")
    parser.add_argument("--vectors_dir", default=f"{base}/diffmeans/last/role_framing_vectors")
    parser.add_argument("--patching_pairs", default="outputs/last/pairs.jsonl",
                        help="clean answers of run_interaction_patching.py (wherever you copied its pairs.jsonl)")
    parser.add_argument("--steering_glob", default=f"{base}/diffmeans_steering*/full/*/pairs.jsonl")
    parser.add_argument("--output_dir", default=f"{base}/verdict_probe")
    parser.add_argument("--target", nargs="+", default=["long", "yes_no"], choices=["long", "yes_no"])
    parser.add_argument("--layers", nargs="+", type=int, default=None, help="default: all")
    parser.add_argument("--print_layers", nargs="+", type=int, default=[8, 16, 18, 20, 24, 31])
    parser.add_argument("--bar_layer", default=20, type=int)
    parser.add_argument("--n_folds", default=10, type=int, help="yes_no target: grouped folds")
    parser.add_argument("--alphas", nargs="+", type=float, default=list(np.logspace(-4, 3, 15)),
                        help="ridge penalties, as multiples of the mean kernel eigenvalue (chosen by GCV)")
    parser.add_argument("--self_test", action="store_true")
    main(parser.parse_args())
