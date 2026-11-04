"""
Where does the role x framing interaction vector of run_interaction_vector.py sit next to the framing and role vectors?

Per review r, layer l (h = last prompt token, prompts/question_only_yes_no.jinja2, from <interaction_dir>/states.pt):
    F_r        h(positive, none) - h(negative, none)                                    framing shift
    R_r        (h(positive, role) - h(positive, none) + h(negative, role) - h(negative, none)) / 2   role shift
    I_r        h(negative, role) - h(negative, none) - h(positive, role) + h(positive, none)
               = -(framing shift with the role - framing shift without it); 0 if framing and role add up
F, R, I are their means over reviews.

Per role and layer:
    cosines    cos(I, F) and cos(I, R), with a 95% bootstrap interval over reviews. cos(I, F) > 0: the role shrinks the
               framing shift (< 0: it grows it)
    shrink     I . F / |F|^2: the fraction of the framing shift the role takes away along F (1 = all of it)
    size       |I| / |F| and |I| / |R|, and |I| under the sign-flip null of run_interaction_vector.py (95th pct)
Subspaces (per layer; pooled over roles for R and I):
    each set's top --k right singular vectors of its per-review vectors (not centred, so the mean is in the span).
    overlap(A, B) = ||Q_A^T Q_B||_F^2 / k, the mean squared cosine of the principal angles (1 = same subspace,
    k / d_model for random ones). ceiling = overlap of a set with itself, fit on two halves of the reviews

Figures (in <interaction_dir>):
    interaction_geometry_by_layer.png   the measures above by layer
    interaction_plane.png               at --layers: each review's I_r on the framing (x) x role-perpendicular (y)
                                        plane, per role; arrows = F, R and I (means)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

FRAMINGS = ["positive", "negative"]
ROLES = ["doctor", "friend", "enemy"]
NO_ROLE = "none"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"framing": "#0b0b0b", "role": "#8f8e89", "doctor": "#2a78d6", "friend": "#eb6834", "enemy": "#1baf7a"}
PAIR_COLORS = {("framing", "role"): "#8f8e89", ("framing", "interaction"): "#0b0b0b", ("role", "interaction"): "#b8457a"}


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=8)


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True)


def top_subspace(x: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k right singular vectors of x [n, d_model] as columns [d_model, k]."""
    return torch.linalg.svd(x, full_matrices=False).Vh[:k].T


def overlap(qa: torch.Tensor, qb: torch.Tensor) -> float:
    return float((qa.T @ qb).pow(2).sum() / qa.shape[1])


def boot_cos(a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor) -> tuple[float, float, float]:
    """cos(mean a, mean b) over reviews and its 95% bootstrap interval; a, b [reviews, d_model], weights [boots, reviews]
    (each resample's count of every review / number of reviews, so weights @ a is the resample's mean)."""
    point = float(unit(a.mean(0)) @ unit(b.mean(0)))
    boots = (unit(weights @ a) * unit(weights @ b)).sum(-1)
    lo, hi = torch.quantile(boots, torch.tensor([0.025, 0.975])).tolist()
    return point, lo, hi


def main(args):
    cache = torch.load(os.path.join(args.interaction_dir, "states.pt"), map_location="cpu", weights_only=False, mmap=True)
    h = cache["states"]                    # [reviews, prompts, layers, d_model] float16
    idx = {p: i for i, p in enumerate(cache["prompts"])}
    num_reviews, _, num_layers, d_model = h.shape
    rng = torch.Generator().manual_seed(args.seed)
    boot_idx = torch.randint(num_reviews, (args.num_boot, num_reviews), generator=rng)
    boot_w = torch.zeros(args.num_boot, num_reviews).scatter_add_(1, boot_idx, torch.ones_like(boot_idx, dtype=torch.float)) / num_reviews
    signs = torch.randint(0, 2, (args.num_perms, num_reviews), generator=rng).float() * 2 - 1
    half = torch.randperm(num_reviews, generator=rng)
    halves = (half[:num_reviews // 2], half[num_reviews // 2:])

    res = {role: {key: np.zeros((num_layers, 3)) for key in ["cos_f", "cos_r"]} for role in ROLES}
    for role in ROLES:
        for key in ["shrink", "size_f", "size_r", "null_f"]:
            res[role][key] = np.zeros(num_layers)
    pairs = list(PAIR_COLORS)
    over = {p: np.zeros(num_layers) for p in pairs}
    ceiling = {name: np.zeros(num_layers) for name in ["framing", "role", "interaction"]}
    plane = {}
    for l in range(num_layers):
        print(f"layer {l}", flush=True)
        x = h[:, :, l].float()
        H = lambda f, r: x[:, idx[f"{f}|{r}"]]
        F = H("positive", NO_ROLE) - H("negative", NO_ROLE)
        Rs, Is = {}, {}
        for role in ROLES:
            Rs[role] = sum(H(f, role) - H(f, NO_ROLE) for f in FRAMINGS) / 2
            Is[role] = H("negative", role) - H("negative", NO_ROLE) - H("positive", role) + H("positive", NO_ROLE)
            f_mean, r_mean, i_mean = F.mean(0), Rs[role].mean(0), Is[role].mean(0)
            res[role]["cos_f"][l] = boot_cos(Is[role], F, boot_w)
            res[role]["cos_r"][l] = boot_cos(Is[role], Rs[role], boot_w)
            res[role]["shrink"][l] = float(i_mean @ f_mean / (f_mean @ f_mean))
            res[role]["size_f"][l] = float(i_mean.norm() / f_mean.norm())
            res[role]["size_r"][l] = float(i_mean.norm() / r_mean.norm())
            res[role]["null_f"][l] = float((signs @ Is[role] / num_reviews).norm(dim=-1).quantile(0.95) / f_mean.norm())

        sets = {"framing": F, "role": torch.cat([Rs[r] for r in ROLES]), "interaction": torch.cat([Is[r] for r in ROLES])}
        q = {name: top_subspace(v, args.k) for name, v in sets.items()}
        for a, b in pairs:
            over[a, b][l] = overlap(q[a], q[b])
        for name in sets:
            n_rows = sets[name].shape[0] // num_reviews
            rows = [torch.cat([hh + j * num_reviews for j in range(n_rows)]) for hh in halves]
            ceiling[name][l] = overlap(top_subspace(sets[name][rows[0]], args.k), top_subspace(sets[name][rows[1]], args.k))

        if l in args.layers:
            f_hat = unit(F.mean(0))
            role_all = unit(sum(Rs[r].mean(0) for r in ROLES))
            r_perp = unit(role_all - (role_all @ f_hat) * f_hat)
            basis = torch.stack([f_hat, r_perp], 1)
            plane[l] = {"F": (F.mean(0) @ basis).numpy(), "R": {r: (Rs[r].mean(0) @ basis).numpy() for r in ROLES},
                        "I": {r: (Is[r] @ basis).numpy() for r in ROLES}}

    print(f"layer  " + "  ".join(f"{r:>7}: cos(I,F)  cos(I,R)  shrink  |I|/|F|" for r in ROLES))
    for l in range(num_layers):
        print(f"L{l:<4} " + "  ".join(f"{'':>7}  {res[r]['cos_f'][l, 0]:+8.2f}  {res[r]['cos_r'][l, 0]:+8.2f}  "
                                      f"{res[r]['shrink'][l]:+6.2f}  {res[r]['size_f'][l]:7.2f}" for r in ROLES))
    print(f"\nsubspace overlap (k={args.k}; random {args.k / d_model:.4f})")
    print("layer  " + "  ".join(f"{a[:4]}~{b[:5]}" for a, b in pairs) + "   ceilings: " + " ".join(ceiling))
    for l in range(num_layers):
        print(f"L{l:<4} " + "  ".join(f"{over[p][l]:10.2f}" for p in pairs) + "   "
              + " ".join(f"{ceiling[n][l]:.2f}" for n in ceiling))

    model = os.path.basename(os.path.dirname(os.path.normpath(args.interaction_dir)))
    title = f"{model} | yes/no prompt | last prompt token | {num_reviews} reviews"
    layers = np.arange(num_layers)

    # Figure 1: by layer
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
    band = 2 / np.sqrt(d_model)
    for ax, key, ylabel, t in [
        (axes[0], "cos_f", "cos(I, framing)", "Interaction vs framing\n+ = the role shrinks the framing shift"),
        (axes[1], "cos_r", "cos(I, role)", "Interaction vs that role's own shift"),
    ]:
        ax.axhspan(-band, band, color=GRID, alpha=0.9, lw=0)
        ax.axhline(0, color=INK_2, lw=0.8)
        for r in ROLES:
            ax.fill_between(layers, res[r][key][:, 1], res[r][key][:, 2], color=COLORS[r], alpha=0.18, lw=0)
            ax.plot(layers, res[r][key][:, 0], color=COLORS[r], lw=2, label=r)
        ax.set(ylim=(-1, 1), xlabel="layer", ylabel=ylabel)
        ax.set_title(t + "   (band = 95% bootstrap)", fontsize=10, color=INK, loc="left")
        ax.legend(fontsize=8, frameon=False)
        style(ax)

    ax = axes[2]
    for r in ROLES:
        ax.plot(layers, res[r]["size_f"], color=COLORS[r], lw=2, label=f"{r}: |I| / |framing|")
        ax.plot(layers, res[r]["shrink"], color=COLORS[r], lw=1.4, ls="--", label=f"{r}: share of framing removed")
        ax.plot(layers, res[r]["null_f"], color=COLORS[r], lw=0.8, ls=":")
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set(xlabel="layer", ylabel="fraction of |framing shift|")
    ax.set_title("Size of the interaction\n(dotted = 95th pct of the sign-flip null)", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False)
    style(ax)

    ax = axes[3]
    ax.axhline(args.k / d_model, color=INK_2, lw=0.8, ls="--")
    for p, color in PAIR_COLORS.items():
        ax.plot(layers, over[p], color=color, lw=2, marker="o", ms=3, label=f"{p[0]} ~ {p[1]}")
    for name, ls in [("framing", ":"), ("interaction", "-.")]:
        ax.plot(layers, ceiling[name], color=INK_2, lw=1, ls=ls, label=f"{name} ~ itself (split half)")
    ax.set(ylim=(0, 1), xlabel="layer", ylabel=f"mean cos² of principal angles (k={args.k})")
    ax.set_title("Subspace overlap across reviews\n(dashed = random subspaces)", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False)
    style(ax)
    fig.suptitle(title, fontsize=11, color=INK)
    fig.tight_layout()
    path = os.path.join(args.interaction_dir, "interaction_geometry_by_layer.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

    # Figure 2: per-review interaction vectors on the framing x role plane
    shown = [l for l in args.layers if l in plane]
    fig, axes = plt.subplots(1, len(shown), figsize=(4.6 * len(shown), 4.8), squeeze=False)
    for ax, l in zip(axes[0], shown):
        p = plane[l]
        for r in ROLES:
            ax.scatter(p["I"][r][:, 0], p["I"][r][:, 1], s=8, alpha=0.25, color=COLORS[r], lw=0)
        arrows = np.stack([p["F"]] + [p["R"][r] for r in ROLES])
        dots = np.concatenate([p["I"][r] for r in ROLES])
        lim = max(np.quantile(np.abs(dots), 0.98), np.abs(arrows).max()) * 1.15
        ax.annotate("", p["F"], (0, 0), arrowprops=dict(arrowstyle="-|>", color=COLORS["framing"], lw=2.2))
        ax.plot([], [], color=COLORS["framing"], lw=2.2, label="framing F")
        for r in ROLES:
            ax.annotate("", p["R"][r], (0, 0), arrowprops=dict(arrowstyle="-|>", color=COLORS[r], lw=1.2, ls="--"))
            ax.annotate("", p["I"][r].mean(0), (0, 0), arrowprops=dict(arrowstyle="-|>", color=COLORS[r], lw=2.4))
            ax.plot([], [], color=COLORS[r], lw=2.4, label=f"I {r} (dashed: R {r})")
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal")
        ax.axhline(0, color=INK_2, lw=0.6)
        ax.axvline(0, color=INK_2, lw=0.6)
        ax.set_xlabel("framing  (negative → positive)", fontsize=8, color=INK_2)
        ax.set_ylabel("role, perpendicular to framing", fontsize=8, color=INK_2)
        ax.set_title(f"Layer {l}", fontsize=10, color=INK, loc="left")
        ax.legend(fontsize=7, frameon=False, loc="lower left")
        style(ax)
    fig.suptitle(title + " | dots = each review's I", fontsize=10, color=INK)
    fig.tight_layout()
    path = os.path.join(args.interaction_dir, "interaction_plane.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Geometry of the role x framing interaction vector")
    parser.add_argument("--interaction_dir", required=True, help="output dir of run_interaction_vector.py (has states.pt).")
    parser.add_argument("--layers", nargs="+", type=int, default=[4, 12, 20, 28], help="layers for the plane figure.")
    parser.add_argument("--k", type=int, default=5, help="subspace dimension.")
    parser.add_argument("--num_boot", type=int, default=500)
    parser.add_argument("--num_perms", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
