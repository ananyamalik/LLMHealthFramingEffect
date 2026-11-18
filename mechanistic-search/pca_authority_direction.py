
"""
Principal components of the authority (doctor - enemy) direction, and the authority direction with the framing
subspace projected out. CPU only: reuses get_diffmeans_directions.py's activations.pt and train/test split.

Per anchor and layer (train reviews of <diffmeans_dir>/<anchor>/directions.pt; every metric marked "test" is on the
held-out reviews):

Authority PCA. One authority difference per (review, framing): a = x[doctor] - x[enemy]. The claim tokens are the same
in both rows, so the content of the review cancels; what is left is the role and how the role interacts with the
claim. dme (the steering vector role_dme) is the mean of these differences.
    mean_energy        |mean a|^2 / mean |a|^2: share of the authority signal that is the single mean direction
    pcs[j]             centered PCA of a (variation of authority across reviews and framings), per PC:
        evr                 explained variance ratio
        cos_framing         cosine with the framing direction
        framing_subspace    norm of the PC inside the framing subspace (K = max --ks): 0 = orthogonal, 1 = inside
        framing_auc         test AUC of the PC score for positive vs negative claim (0.5: the PC does not depend on
                            framing; far from 0.5: a framing-dependent part of authority, i.e. role x framing)
        behavior_r          test correlation of the PC score with the behavioral D(doctor) - D(enemy)
    interaction        mean a over positive claims - over negative claims (= framing_doctor - framing_enemy):
                       cosine with framing and with dme

Framing subspace F_K (K = --ks): orthonormal basis of [framing direction, top K-1 centered PCs of the per-review framing
differences x[positive, role] - x[negative, role], every role incl. none]. framing_r2_test: share of the held-out
framing differences' energy inside F_K (how much of framing the subspace holds).

Ablated authority: dme_perp_K = dme - F_K F_K^T dme, per K:
    kept_norm          |dme_perp_K| / |dme|  (random K-dim subspace as control: random_kept_norm)
    auc_authority      test AUC of the projection on dme_perp_K, doctor vs enemy rows (should stay ~1)
    framing_leak       test AUC of the same projection, positive vs negative, over all rows (0.5 = no framing)
    framing_leak_roles the same within the doctor and enemy rows only
(dme itself is reported as K = 0.)

Outputs, per anchor in <diffmeans_dir>/<anchor>/authority_pca/:
    summary.json
    authority_pca.png
    directions_framing_ablated.pt   {"directions": {"doctor_minus_enemy_perp_k<K>": [layers, d_model], ...},
                                     "authority_pcs": [layers, n_pcs, d_model], "framing_basis_k<K>": [layers, K, d_model],
                                     "test_reviews", "train_reviews", "model"}  (run_steering_sweep.py --perp_k K reads it)

    python pca_authority_direction.py --diffmeans_dir outputs/olmo2-7B-instruct/diffmeans --anchors yes_no last
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

FRAMINGS = ["positive", "negative"]
NO_ROLE = "none"


def auc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm().clamp_min(1e-12)


def index_rows(rows: list[dict]) -> dict[tuple[str, str, str], int]:
    return {(r["review_id"], r["framing"], r["role"]): i for i, r in enumerate(rows)}


def framing_basis(framing: torch.Tensor, diffs: torch.Tensor, k: int) -> torch.Tensor:
    """[k, d] orthonormal: the framing direction, then the top k-1 centered PCs of the framing differences."""
    if k == 1:
        return unit(framing)[None]
    centered = diffs - diffs.mean(0, keepdim=True)
    _, _, vt = torch.linalg.svd(centered, full_matrices=False)
    q, _ = torch.linalg.qr(torch.cat([unit(framing)[None], vt[:k - 1]]).T)
    return q.T


def analyze_layer(x_l, idx, train, test, roles, dirs_l, ks, n_pcs, d_behavior, generator):
    """x_l: [rows, d] float32 at one layer. Returns (layer summary, ablated vectors, authority PCs, bases)."""
    dme, framing = dirs_l["doctor_minus_enemy"], dirs_l["framing"]

    def auth(reviews):
        keys = [(r, f) for r in sorted(reviews) for f in FRAMINGS]
        a = x_l[[idx[(r, f, "doctor")] for r, f in keys]] - x_l[[idx[(r, f, "enemy")] for r, f in keys]]
        lab = np.array([f == FRAMINGS[0] for _, f in keys])
        beh = np.array([d_behavior[idx[(r, f, "doctor")]] - d_behavior[idx[(r, f, "enemy")]] for r, f in keys])
        return a, lab, beh

    def fdiffs(reviews):
        return torch.stack([x_l[idx[(r, FRAMINGS[0], k)]] - x_l[idx[(r, FRAMINGS[1], k)]]
                            for r in sorted(reviews) for k in [NO_ROLE] + roles])

    a_tr, lab_tr, _ = auth(train)
    a_te, lab_te, beh_te = auth(test)
    f_tr, f_te = fdiffs(train), fdiffs(test)

    # Authority PCA
    mean_a = a_tr.mean(0)
    mean_energy = float(mean_a.norm() ** 2 / (a_tr.norm(dim=1) ** 2).mean())
    _, s, vt = torch.linalg.svd(a_tr - mean_a, full_matrices=False)
    evr = (s ** 2 / (s ** 2).sum())[:n_pcs]
    pcs = vt[:n_pcs]
    bases = {k: framing_basis(framing, f_tr, k) for k in ks}
    big = bases[max(ks)]
    fu, du = unit(framing), unit(dme)
    pc_rows = []
    for j in range(n_pcs):
        score = (a_te @ pcs[j]).numpy()
        pc_rows.append({"evr": float(evr[j]), "cos_framing": float(pcs[j] @ fu), "cos_dme": float(pcs[j] @ du),
                        "framing_subspace": float((big @ pcs[j]).norm()),
                        "framing_auc": auc(score, lab_te),
                        "behavior_r": float(np.corrcoef(score, beh_te)[0, 1])})
    interaction = a_tr[lab_tr].mean(0) - a_tr[~lab_tr].mean(0)

    # Ablated authority vectors, scored on the held-out rows
    te_rows = [idx[(r, f, k)] for r in sorted(test) for f in FRAMINGS for k in [NO_ROLE] + roles]
    te_framing = np.array([f == FRAMINGS[0] for r in sorted(test) for f in FRAMINGS for k in [NO_ROLE] + roles])
    te_role = np.array([k for r in sorted(test) for f in FRAMINGS for k in [NO_ROLE] + roles])
    xt = x_l[te_rows]
    de = (te_role == "doctor") | (te_role == "enemy")

    def score_vector(v):
        p = (xt @ unit(v)).numpy()
        return {"auc_authority": auc(p[de], te_role[de] == "doctor"),
                "framing_leak": auc(p, te_framing), "framing_leak_roles": auc(p[de], te_framing[de])}

    ablation = {"0": {"kept_norm": 1.0, "random_kept_norm": 1.0, "framing_r2_test": 0.0} | score_vector(dme)}
    perp = {}
    for k, b in bases.items():
        v = dme - b.T @ (b @ dme)
        rq, _ = torch.linalg.qr(torch.randn(dme.shape[0], k, generator=generator))
        rand_kept = float((dme - rq @ (rq.T @ dme)).norm() / dme.norm())
        ablation[str(k)] = {"kept_norm": float(v.norm() / dme.norm()), "random_kept_norm": rand_kept,
                            "framing_r2_test": float((f_te @ b.T).norm() ** 2 / f_te.norm() ** 2)} | score_vector(v)
        perp[k] = v
    summary = {"norm_dme": float(dme.norm()), "mean_energy": mean_energy, "pcs": pc_rows,
               "interaction_norm": float(interaction.norm()), "interaction_cos_framing": float(unit(interaction) @ fu),
               "interaction_cos_dme": float(unit(interaction) @ du), "ablation": ablation}
    return summary, perp, pcs, bases


def plot(summaries: list[dict], ks: list[int], n_show: int, title: str, path: str) -> None:
    layers = np.arange(len(summaries))
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, n_show))

    def per_pc(key):
        return np.array([[s["pcs"][j][key] for j in range(n_show)] for s in summaries])   # [layers, n_show]

    ax = axes[0, 0]
    ax.plot(layers, [s["mean_energy"] for s in summaries], color="black", lw=2, label="mean direction (dme)")
    for j in range(n_show):
        ax.plot(layers, per_pc("evr")[:, j], color=cmap[j], label=f"PC{j + 1} evr (centered)")
    ax.set(title="Authority differences: energy in dme, EVR of the PCs", xlabel="layer", ylim=(0, 1))
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    v = per_pc("framing_auc")
    for j in range(n_show):
        ax.plot(layers, np.abs(v[:, j] - 0.5) + 0.5, color=cmap[j], label=f"PC{j + 1}")
    ax.axhline(0.5, color="gray", lw=0.8)
    ax.set(title="Authority PCs: test AUC for positive vs negative (folded)", xlabel="layer", ylim=(0.45, 1.02))
    ax.legend(fontsize=7)

    ax = axes[0, 2]
    v = per_pc("framing_subspace")
    for j in range(n_show):
        ax.plot(layers, v[:, j], color=cmap[j], label=f"PC{j + 1}")
    ax.plot(layers, [abs(s["interaction_cos_framing"]) for s in summaries], color="#d62728", ls="--",
            label="|cos(interaction, framing)|")
    ax.set(title=f"Authority PCs: norm inside framing subspace (K={max(ks)})", xlabel="layer", ylim=(0, 1))
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    v = per_pc("behavior_r")
    for j in range(n_show):
        ax.plot(layers, v[:, j], color=cmap[j], label=f"PC{j + 1}")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set(title="Authority PCs: test corr with D(doctor) - D(enemy)", xlabel="layer")
    ax.legend(fontsize=7)

    kc = plt.cm.plasma(np.linspace(0, 0.85, len(ks) + 1))
    ax = axes[1, 1]
    for i, k in enumerate(["0"] + [str(k) for k in ks]):
        ax.plot(layers, [s["ablation"][k]["kept_norm"] for s in summaries], color=kc[i], label=f"K={k}")
    ax.plot(layers, [s["ablation"][str(max(ks))]["random_kept_norm"] for s in summaries], color="gray", ls=":",
            label=f"random K={max(ks)}")
    ax.set(title="|dme_perp_K| / |dme|", xlabel="layer", ylim=(0, 1.02))
    ax.legend(fontsize=7)

    ax = axes[1, 2]
    for i, k in enumerate(["0"] + [str(k) for k in ks]):
        ax.plot(layers, [s["ablation"][k]["framing_leak"] for s in summaries], color=kc[i], label=f"framing leak K={k}")
        ax.plot(layers, [s["ablation"][k]["auc_authority"] for s in summaries], color=kc[i], ls="--", lw=0.9)
    ax.axhline(0.5, color="gray", lw=0.8)
    ax.set(title="dme_perp_K, test: framing leak (solid), doctor vs enemy AUC (dashed)", xlabel="layer", ylim=(0.3, 1.02))
    ax.legend(fontsize=7)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"Saved {path}")


def main(args):
    saved = torch.load(args.acts_path, mmap=True)
    rows, roles = saved["rows"], saved["roles"]
    idx = index_rows(rows)
    d_behavior = np.array([r["d"] for r in rows], dtype=np.float64)
    ks = sorted(args.ks)
    for anchor in args.anchors:
        fit = torch.load(os.path.join(args.diffmeans_dir, anchor, "directions.pt"))
        train, test = set(fit["train_reviews"]), set(fit["test_reviews"])
        x = saved["acts"][anchor].float() if args.in_memory else saved["acts"][anchor]
        n_layers = x.shape[1]
        generator = torch.Generator().manual_seed(args.seed)
        summaries, perp, pcs, bases = [], {k: [] for k in ks}, [], {k: [] for k in ks}
        for l in range(n_layers):
            x_l = x[:, l].float()
            dirs_l = {n: fit["directions"][n][l].float() for n in ["doctor_minus_enemy", "framing"]}
            s, p, pc, b = analyze_layer(x_l, idx, train, test, roles, dirs_l, ks, args.n_pcs, d_behavior, generator)
            summaries.append(s)
            pcs.append(pc)
            for k in ks:
                perp[k].append(p[k])
                bases[k].append(b[k])
            if l % args.print_stride == 0 or l == n_layers - 1:
                a = s["ablation"]
                top = " ".join(f"PC{j + 1}(evr {q['evr']:.2f} fAUC {q['framing_auc']:.2f} cosF {q['cos_framing']:+.2f})"
                               for j, q in enumerate(s["pcs"][:3]))
                print(f"[{anchor}] L{l:<2} mean_energy {s['mean_energy']:.2f} | {top} | "
                      + " ".join(f"K{k}: kept {a[k]['kept_norm']:.2f} leak {a[k]['framing_leak']:.2f} auth {a[k]['auc_authority']:.2f}"
                                 for k in ["0"] + [str(k) for k in ks]))
        out_dir = os.path.join(args.diffmeans_dir, anchor, "authority_pca")
        os.makedirs(out_dir, exist_ok=True)
        torch.save({"directions": {f"doctor_minus_enemy_perp_k{k}": torch.stack(perp[k]) for k in ks},
                    "authority_pcs": torch.stack(pcs)}
                   | {f"framing_basis_k{k}": torch.stack(bases[k]) for k in ks}
                   | {"test_reviews": fit["test_reviews"], "train_reviews": fit["train_reviews"],
                      "model": fit.get("model"), "anchor": anchor, "ks": ks},
                   os.path.join(out_dir, "directions_framing_ablated.pt"))
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump({"anchor": anchor, "model": fit.get("model"), "ks": ks, "n_train_reviews": len(train),
                       "n_test_reviews": len(test), "layers": summaries}, f, indent=2)
        plot(summaries, ks, min(args.n_show, args.n_pcs), f"{fit.get('model')} | anchor {anchor} | authority = doctor - enemy",
             os.path.join(out_dir, "authority_pca.png"))
        del x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCA of the authority (doctor - enemy) direction and framing ablation")
    parser.add_argument("--diffmeans_dir", required=True, help="output dir of get_diffmeans_directions.py")
    parser.add_argument("--acts_path", default=None, help="default: <diffmeans_dir>/activations.pt")
    parser.add_argument("--anchors", nargs="+", default=["yes_no"])
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8, 16], help="framing subspace sizes to ablate")
    parser.add_argument("--n_pcs", default=10, type=int)
    parser.add_argument("--n_show", default=5, type=int, help="PCs drawn in the plot")
    parser.add_argument("--in_memory", action="store_true", help="load the anchor's activations into RAM first")
    parser.add_argument("--print_stride", default=4, type=int)
    parser.add_argument("--seed", default=0, type=int, help="seed of the random-subspace control")
    args = parser.parse_args()
    args.acts_path = args.acts_path or os.path.join(args.diffmeans_dir, "activations.pt")
    main(args)
