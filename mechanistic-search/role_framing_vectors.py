"""
Framing, role-given-framing and their interaction, side by side, for each role. CPU only: reuses
get_diffmeans_directions.py's activations.pt and train/test split (default anchor `last`: the last token of the
question-only prompt, where the full answer starts).

Cells (A_<role><framing>, mean residual over the training reviews, per layer; role = doctor, friend, enemy):
    A00 = "X won't work ..."                  A01 = "X will work ..."
    A10 = "My <role> said, X won't work ..."  A11 = "My <role> said, X will work ..."

Vectors (each also formed per review; the vectors are the means over reviews):
    f       = A01 - A00                 framing, without the role
    d_pos   = A11 - A01                 the role under positive framing
    d_neg   = A10 - A00                 the role under negative framing
    inter   = d_pos - d_neg = (A11 - A10) - (A01 - A00)
              = framing with the role - framing without it; 0 if role and framing add up

1. Decomposition of the interaction (fh = f / |f|):
    inter_par  = (inter . fh) fh      the role amplifies / dampens the framing axis
    inter_perp = inter - inter_par    framing-specific role effect that is not on f
    inter_over_dpos  |inter| / |d_pos|
    gain             inter . fh / |f|   (> 0: the role makes the framing gap larger; -0.3: 30% smaller along f)
    perp_frac        |inter_perp| / |inter|
    perp_null95      95th pct of |mean_r s_r inter_perp_r| over random signs s_r (swapping role <-> none within a
                     review flips the sign of its inter_r): is the off-axis part more than review noise?
    (+ 95% bootstrap intervals over reviews for gain and perp_frac; inter_null95 likewise for the whole inter)

2. Subspaces of d_pos and d_neg. U_pos, U_neg = top k right singular vectors of the centered per-review vectors
   (how the role effect varies across reviews), for k in --ks:
    cosines          principal angle cosines svdvals(U_pos U_neg^T)
    overlap          mean squared cosine (1 = same subspace, k / d_model for random ones)
    split-half reference, the train reviews split in two halves A and B (--num_splits random splits):
        ceiling      overlap(U_pos^A, U_pos^B) and overlap(U_neg^A, U_neg^B), averaged: the overlap one subspace
                     gets with itself at this sample size
        cross_half   overlap(U_pos^A, U_neg^B) and overlap(U_neg^A, U_pos^B), averaged: the same sample size, across
        ratio        cross_half / ceiling (~1: d_pos and d_neg vary in the same subspace; << 1: different ones)

Also per layer: norms, cosines between the four vectors (95% bootstrap for cos(d_pos, d_neg), cos(inter, f)),
held-out reliability cos(train fit, test fit), and held-out AUCs of d_pos / d_neg for role vs none within each framing.

Outputs, in <diffmeans_dir>/<anchor>/role_framing_vectors/:
    <role>/directions.pt     {"directions": {"f", "d_pos", "d_neg", "inter", "inter_par", "inter_perp", "d_mean":
                              [layers, d_model]}, "subspaces": {"pos", "neg": [layers, max k, d_model]},
                              "train_reviews", "test_reviews", "role", "anchor", "model"}
    <role>/summary.json
    <role>/role_framing_vectors.png
    comparison.png           the roles side by side

    python role_framing_vectors.py --diffmeans_dir outputs/olmo2-7B-instruct/diffmeans --in_memory
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

POS, NEG, NONE = "positive", "negative", "none"
NAMES = ["f", "d_pos", "d_neg", "inter"]
COLORS = {"f": "#0b0b0b", "d_pos": "#2a78d6", "d_neg": "#eb6834", "inter": "#b8457a", "d_mean": "#8f8e89",
          "inter_par": "#d98bb4", "inter_perp": "#6b1f47"}
ROLE_COLORS = {"doctor": "#2a78d6", "friend": "#eb6834", "enemy": "#1baf7a"}


def auc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label], score[~label]
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(unit(a) @ unit(b))


def per_review(x_l: torch.Tensor, idx: dict, reviews: list[str], role: str) -> dict[str, torch.Tensor]:
    """[reviews, d] per-review vectors."""
    A = {(k, f): x_l[[idx[(r, f, k)] for r in reviews]] for k in [NONE, role] for f in [POS, NEG]}
    f = A[(NONE, POS)] - A[(NONE, NEG)]
    d_pos = A[(role, POS)] - A[(NONE, POS)]
    d_neg = A[(role, NEG)] - A[(NONE, NEG)]
    return {"f": f, "d_pos": d_pos, "d_neg": d_neg, "inter": d_pos - d_neg, "d_mean": (d_pos + d_neg) / 2}


def decompose(inter: torch.Tensor, f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    fh = unit(f)
    par = (inter @ fh)[..., None] * fh
    return par, inter - par


def top_subspace(v: torch.Tensor, k: int) -> torch.Tensor:
    """[k, d] top right singular vectors of the centered rows of v."""
    return torch.linalg.svd(v - v.mean(0), full_matrices=False).Vh[:k]


def overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((torch.linalg.svdvals(a @ b.T) ** 2).mean())


def subspaces(dp: torch.Tensor, dn: torch.Tensor, ks: list[int], rng, num_splits: int):
    kmax = max(ks)
    up, un = top_subspace(dp, kmax), top_subspace(dn, kmax)
    out = {}
    for k in ks:
        c = torch.linalg.svdvals(up[:k] @ un[:k].T)
        out[str(k)] = {"cosines": c.tolist(), "overlap": float((c ** 2).mean()), "random": k / dp.shape[1],
                       "ceiling": [], "cross_half": []}
    n = dp.shape[0]
    for _ in range(num_splits):
        perm = rng.permutation(n)
        a, b = perm[: n // 2], perm[n // 2:]
        s = {key: top_subspace(v[ix], kmax) for key, v, ix in
             [("pa", dp, a), ("pb", dp, b), ("na", dn, a), ("nb", dn, b)]}
        for k in ks:
            o = out[str(k)]
            o["ceiling"].append((overlap(s["pa"][:k], s["pb"][:k]) + overlap(s["na"][:k], s["nb"][:k])) / 2)
            o["cross_half"].append((overlap(s["pa"][:k], s["nb"][:k]) + overlap(s["na"][:k], s["pb"][:k])) / 2)
    for k in ks:
        o = out[str(k)]
        o["ceiling"], o["cross_half"] = float(np.mean(o["ceiling"])), float(np.mean(o["cross_half"]))
        o["ratio"] = o["cross_half"] / o["ceiling"]
    return out, up, un


def analyze_layer(x_l, idx, train, test, role, rng, args):
    tr, te = per_review(x_l, idx, train, role), per_review(x_l, idx, test, role)
    m = {k: v.mean(0) for k, v in tr.items()}
    mt = {k: v.mean(0) for k, v in te.items()}
    n = len(train)
    m["inter_par"], m["inter_perp"] = decompose(m["inter"], m["f"])
    mt["inter_par"], mt["inter_perp"] = decompose(mt["inter"], mt["f"])

    # Sign-flip nulls: per-review inter_r, and its part off the (train) framing direction
    signs = torch.from_numpy(rng.choice([-1.0, 1.0], (args.num_perms, n))).float()
    perp_r = decompose(tr["inter"], m["f"])[1]
    null95 = float(np.percentile(((signs @ tr["inter"]) / n).norm(dim=1).numpy(), 95))
    perp_null95 = float(np.percentile(((signs @ perp_r) / n).norm(dim=1).numpy(), 95))

    def stats(mm):
        par, perp = decompose(mm["inter"], mm["f"])
        return {"gain": float(mm["inter"] @ unit(mm["f"]) / mm["f"].norm()),
                "perp_frac": float(perp.norm() / mm["inter"].norm()),
                "cos_dpos_dneg": cos(mm["d_pos"], mm["d_neg"]), "cos_inter_f": cos(mm["inter"], mm["f"])}

    boot = [stats({k: tr[k][b].mean(0) for k in NAMES}) for b in rng.integers(0, n, (args.num_boot, n))]
    ci = {k: [float(np.percentile([b[k] for b in boot], 2.5)), float(np.percentile([b[k] for b in boot], 97.5))]
          for k in boot[0]}

    fu = unit(m["f"])
    s = {"norm": {k: float(m[k].norm()) for k in m},
         "inter_null95": null95, "perp_null95": perp_null95,
         "inter_over_dpos": float(m["inter"].norm() / m["d_pos"].norm()),
         "inter_over_f": float(m["inter"].norm() / m["f"].norm()),
         "gain": float(m["inter"] @ fu / m["f"].norm()),
         "perp_frac": float(m["inter_perp"].norm() / m["inter"].norm()),
         "ci": ci,
         "cos": {f"{a}|{b}": cos(m[a], m[b]) for i, a in enumerate(NAMES) for b in NAMES[i + 1:]}
                | {"inter_perp|d_pos": cos(m["inter_perp"], m["d_pos"]), "inter_perp|d_neg": cos(m["inter_perp"], m["d_neg"])},
         "along_f": {k: float(m[k] @ fu) for k in ["d_pos", "d_neg", "inter"]},
         "reliability": {k: cos(m[k], mt[k]) for k in NAMES + ["d_mean", "inter_perp"]}}

    def rows(k, f):
        return x_l[[idx[(r, f, k)] for r in test]]

    lab = np.r_[np.ones(len(test), bool), np.zeros(len(test), bool)]
    transfer = {}
    for vec in ["d_pos", "d_neg"]:
        for f in [POS, NEG]:
            transfer[f"{vec}|{f}"] = auc((torch.cat([rows(role, f), rows(NONE, f)]) @ unit(m[vec])).numpy(), lab)
    s["transfer_auc"] = transfer

    s["subspace"], up, un = subspaces(tr["d_pos"], tr["d_neg"], args.ks, rng, args.num_splits)
    return s, m, up, un


def plot_role(summaries: list[dict], role: str, ks: list[int], title: str, path: str) -> None:
    L = np.arange(len(summaries))
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))

    ax = axes[0, 0]
    for k in NAMES + ["inter_perp"]:
        ax.plot(L, [s["norm"][k] for s in summaries], color=COLORS[k], label=k)
    ax.plot(L, [s["inter_null95"] for s in summaries], color=COLORS["inter"], ls=":", label="inter null 95%")
    ax.plot(L, [s["perp_null95"] for s in summaries], color=COLORS["inter_perp"], ls=":", label="inter_perp null 95%")
    ax.set(title="Norms", xlabel="layer", yscale="log")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    for key, c in [("d_pos|d_neg", "#2a78d6"), ("f|d_pos", "#6aa3e6"), ("f|d_neg", "#eb6834"), ("f|inter", "#b8457a"),
                   ("inter_perp|d_pos", "#1baf7a"), ("inter_perp|d_neg", "#8f8e89")]:
        ax.plot(L, [s["cos"][key] for s in summaries], color=c, label=f"cos({key.replace('|', ', ')})")
    for key, c in [("cos_dpos_dneg", "#2a78d6"), ("cos_inter_f", "#b8457a")]:
        lo, hi = zip(*[s["ci"][key] for s in summaries])
        ax.fill_between(L, lo, hi, color=c, alpha=0.15)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set(title="Cosines (bands: 95% bootstrap)", xlabel="layer", ylim=(-1, 1))
    ax.legend(fontsize=7)

    ax = axes[0, 2]
    ax.plot(L, [s["inter_over_dpos"] for s in summaries], color=COLORS["d_pos"], label="|inter| / |d_pos|")
    ax.plot(L, [s["perp_frac"] for s in summaries], color=COLORS["inter_perp"], label="|inter_perp| / |inter|")
    ax.plot(L, [s["gain"] for s in summaries], color=COLORS["inter"], label="gain = inter·f̂ / |f|")
    for key, c in [("gain", COLORS["inter"]), ("perp_frac", COLORS["inter_perp"])]:
        lo, hi = zip(*[s["ci"][key] for s in summaries])
        ax.fill_between(L, lo, hi, color=c, alpha=0.15)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set(title="1. Interaction decomposition", xlabel="layer")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for k in ["d_pos", "d_neg", "inter"]:
        ax.plot(L, [s["along_f"][k] for s in summaries], color=COLORS[k], label=f"{k} · f̂")
    ax.plot(L, [s["norm"]["f"] for s in summaries], color=COLORS["f"], ls="--", label="|f|")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set(title="Component along the framing direction", xlabel="layer")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    kc = plt.cm.viridis(np.linspace(0, 0.85, len(ks)))
    for i, k in enumerate(ks):
        sub = [s["subspace"][str(k)] for s in summaries]
        ax.plot(L, [o["overlap"] for o in sub], color=kc[i], label=f"k={k} overlap(U_pos, U_neg)")
        ax.plot(L, [o["ceiling"] for o in sub], color=kc[i], ls=":", lw=1)
        ax.plot(L, [o["cross_half"] for o in sub], color=kc[i], ls="--", lw=1)
    ax.set(title="2. Subspaces: overlap (dashed: half vs half across, dotted: within)", xlabel="layer", ylim=(0, 1.02))
    ax.legend(fontsize=7)

    ax = axes[1, 2]
    for i, k in enumerate(ks):
        ax.plot(L, [s["subspace"][str(k)]["ratio"] for s in summaries], color=kc[i], label=f"k={k}")
    ax.axhline(1, color="gray", lw=0.8)
    ax.set(title="2. Subspaces: cross_half / ceiling (1 = same subspace)", xlabel="layer", ylim=(0, 1.1))
    ax.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"Saved {path}")


def plot_comparison(by_role: dict[str, list[dict]], k: int, layer: int, title: str, path: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    panels = [("gain", "gain = inter·f̂ / |f|  (< 0: role shrinks the framing gap)", "ci"),
              ("perp_frac", "|inter_perp| / |inter|  (share off the framing axis)", "ci"),
              ("inter_over_dpos", "|inter| / |d_pos|", None),
              ("cos_dpos_dneg", "cos(d_pos, d_neg)", "ci")]
    for ax, (key, name, band) in zip(axes.flat, panels):
        for role, summaries in by_role.items():
            L = np.arange(len(summaries))
            c = ROLE_COLORS.get(role, "gray")
            vals = [s["cos"]["d_pos|d_neg"] if key == "cos_dpos_dneg" else s[key] for s in summaries]
            ax.plot(L, vals, color=c, label=role)
            if band:
                lo, hi = zip(*[s["ci"][key] for s in summaries])
                ax.fill_between(L, lo, hi, color=c, alpha=0.15)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set(title=name, xlabel="layer")
        ax.legend(fontsize=8)

    ax = axes[1, 1]
    for role, summaries in by_role.items():
        L = np.arange(len(summaries))
        ax.plot(L, [s["subspace"][str(k)]["ratio"] for s in summaries], color=ROLE_COLORS.get(role, "gray"), label=role)
    ax.axhline(1, color="gray", lw=0.8)
    ax.set(title=f"Subspaces (k={k}): cross_half / ceiling", xlabel="layer", ylim=(0, 1.1))
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    for role, summaries in by_role.items():
        c = summaries[layer]["subspace"][str(k)]["cosines"]
        ax.plot(np.arange(1, len(c) + 1), c, "o-", color=ROLE_COLORS.get(role, "gray"), label=role)
    ax.set(title=f"Principal-angle cosines, U_pos vs U_neg, layer {layer} (k={k})", xlabel="index", ylim=(0, 1.02))
    ax.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"Saved {path}")


def main(args):
    saved = torch.load(args.acts_path, mmap=True)
    rows = saved["rows"]
    idx = {(r["review_id"], r["framing"], r["role"]): i for i, r in enumerate(rows)}
    roles = args.roles or saved["roles"]
    fit = torch.load(os.path.join(args.diffmeans_dir, args.anchor, "directions.pt"))
    train, test = sorted(fit["train_reviews"]), sorted(fit["test_reviews"])
    x = saved["acts"][args.anchor]
    if args.in_memory:
        x = x.clone()
    rng = np.random.default_rng(args.seed)
    ks = sorted(args.ks)
    base_dir = os.path.join(args.diffmeans_dir, args.anchor, "role_framing_vectors")
    model = saved.get("model")

    by_role = {}
    for role in roles:
        summaries, vecs, ups, uns = [], {}, [], []
        for l in range(x.shape[1]):
            s, m, up, un = analyze_layer(x[:, l].float(), idx, train, test, role, rng, args)
            summaries.append(s)
            ups.append(up)
            uns.append(un)
            for k, v in m.items():
                vecs.setdefault(k, []).append(v)
            if l % args.print_stride == 0 or l == x.shape[1] - 1:
                sub = s["subspace"][str(args.report_k)]
                print(f"[{role}] L{l:<2} |I|/|d+| {s['inter_over_dpos']:.2f} gain {s['gain']:+.2f} "
                      f"perp_frac {s['perp_frac']:.2f} (|I_perp| {s['norm']['inter_perp']:.2f}, null {s['perp_null95']:.2f}) "
                      f"| cos(d+,d-) {s['cos']['d_pos|d_neg']:+.2f} | k={args.report_k}: overlap {sub['overlap']:.2f} "
                      f"cross_half {sub['cross_half']:.2f} ceiling {sub['ceiling']:.2f} ratio {sub['ratio']:.2f} "
                      f"top cos {sub['cosines'][0]:.2f}..{sub['cosines'][-1]:.2f}")
        by_role[role] = summaries
        out_dir = os.path.join(base_dir, role)
        os.makedirs(out_dir, exist_ok=True)
        meta = {"role": role, "anchor": args.anchor, "model": model, "train_reviews": train, "test_reviews": test}
        torch.save({"directions": {k: torch.stack(v) for k, v in vecs.items()},
                    "subspaces": {"pos": torch.stack(ups), "neg": torch.stack(uns)}} | meta,
                   os.path.join(out_dir, "directions.pt"))
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(meta | {"ks": ks, "n_train_reviews": len(train), "n_test_reviews": len(test), "layers": summaries},
                      f, indent=2)
        plot_role(summaries, role, ks, f"{model} | anchor {args.anchor} | f, d|pos, d|neg, interaction ({role})",
                  os.path.join(out_dir, "role_framing_vectors.png"))
    if len(by_role) > 1:
        plot_comparison(by_role, args.report_k, args.report_layer if args.report_layer >= 0 else x.shape[1] - 1,
                        f"{model} | anchor {args.anchor} | roles side by side", os.path.join(base_dir, "comparison.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Framing, role|positive, role|negative and their interaction")
    parser.add_argument("--diffmeans_dir", required=True, help="output dir of get_diffmeans_directions.py")
    parser.add_argument("--acts_path", default=None, help="default: <diffmeans_dir>/activations.pt")
    parser.add_argument("--anchor", default="last", help="last = question-only prompt, where the full answer starts")
    parser.add_argument("--roles", nargs="+", default=None, help="default: every role in the activations")
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10, 20], help="subspace sizes")
    parser.add_argument("--report_k", default=10, type=int, help="k printed and in comparison.png (must be in --ks)")
    parser.add_argument("--report_layer", default=-1, type=int, help="layer of the principal-cosine panel (-1: last)")
    parser.add_argument("--num_splits", default=10, type=int, help="random half splits for the subspace reference")
    parser.add_argument("--num_perms", default=2000, type=int)
    parser.add_argument("--num_boot", default=300, type=int)
    parser.add_argument("--in_memory", action="store_true", help="load the anchor's activations into RAM first")
    parser.add_argument("--print_stride", default=4, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    args.acts_path = args.acts_path or os.path.join(args.diffmeans_dir, "activations.pt")
    assert args.report_k in args.ks, "--report_k must be one of --ks"
    main(args)
