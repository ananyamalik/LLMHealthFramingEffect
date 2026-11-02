"""
Are the framing and role directions of get_diffmeans_directions.py orthogonal, and does PCA find them on its own?

Directions (from <diffmeans_dir>/<anchor>/directions.pt, fit on the training reviews):
    framing   positive - negative claim, averaged over roles
    role      the main axis of the role-specific vectors (first principal component of doctor/friend/enemy_specific,
              sign: doctor end positive). In practice almost the same as doctor_minus_enemy

Rows are the residuals in <diffmeans_dir>/activations.pt. Each review's own mean (over its framings and roles) is
subtracted first, so what is left is only the framing and role variation, not which treatment the review is about.
PCA is fit on the training reviews' rows; points in the scatters are the held-out test reviews.

Figures (in <diffmeans_dir>/<anchor>/):
    geometry_by_layer.png        per layer: cos(framing, role) vs two random directions; |cos| of the PCA's top 2
                                 directions with framing and with role; variance explained by PC1 and PC2
    geometry_scatter.png         at --layers: test rows on the framing x role plane (top), and on PC1 x PC2 (bottom),
                                 colour = role, marker = framing
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

NO_ROLE = "none"
FRAMINGS = ["positive", "negative"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DIRECTION_COLORS = {"framing": "#2a78d6", "role": "#eb6834"}
ROLE_COLORS = {"none": "#8f8e89", "doctor": "#2a78d6", "friend": "#eb6834", "enemy": "#1baf7a"}
FRAMING_MARKERS = {"positive": "o", "negative": "^"}


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


def role_axis(directions: dict, roles: list[str]) -> torch.Tensor:
    """First principal component of the role-specific vectors per layer [layers, d_model], doctor end positive."""
    u = torch.stack([directions[f"{r}_specific"] for r in roles], 1)   # [layers, roles, d_model]
    axis = torch.linalg.svd(u, full_matrices=False).Vh[:, 0]           # [layers, d_model]
    sign = torch.sign((axis * directions[f"{roles[0]}_minus_{roles[-1]}"]).sum(-1, keepdim=True))
    return axis * sign


def center_by_review(x: torch.Tensor, review_ids: np.ndarray) -> torch.Tensor:
    """Subtracts each review's mean row (over its framings and roles)."""
    x = x.clone()
    for rid in np.unique(review_ids):
        m = torch.from_numpy(review_ids == rid)
        x[m] -= x[m].mean(0)
    return x


def pca(x_fit: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k principal directions [k, d_model] and their fraction of the variance."""
    x_fit = x_fit - x_fit.mean(0)
    _, s, vh = torch.linalg.svd(x_fit, full_matrices=False)
    return vh[:k], (s[:k] ** 2 / (s ** 2).sum())


def main(args):
    saved = torch.load(os.path.join(args.diffmeans_dir, "activations.pt"), mmap=True)
    fit = torch.load(os.path.join(args.diffmeans_dir, args.anchor, "directions.pt"))
    rows, roles, directions = saved["rows"], fit["roles"], fit["directions"]
    acts = saved["acts"][args.anchor]   # [rows, layers, d_model] float16
    num_layers, d_model = acts.shape[1], acts.shape[2]
    train, test = set(fit["train_reviews"]), set(fit["test_reviews"])
    keep = [i for i, r in enumerate(rows) if r["review_id"] in train | test]
    review = np.array([rows[i]["review_id"] for i in keep])
    framing_lab = np.array([rows[i]["framing"] for i in keep])
    role_lab = np.array([rows[i]["role"] for i in keep])
    is_test = np.isin(review, sorted(test))

    framing_dir = unit(directions["framing"])
    role_dir = unit(role_axis(directions, roles))

    # Per layer: orthogonality and whether unsupervised PCA finds the two directions
    cos_fr = (framing_dir * role_dir).sum(-1).numpy()
    pc_cos = {name: np.zeros((num_layers, 2)) for name in DIRECTION_COLORS}
    pc_var = np.zeros((num_layers, 2))
    scatter = {}
    for l in range(num_layers):
        x = center_by_review(acts[keep, l].float(), review)
        pcs, var = pca(x[torch.from_numpy(~is_test)], 2)
        pc_var[l] = var.numpy()
        for name, d in [("framing", framing_dir[l]), ("role", role_dir[l])]:
            pc_cos[name][l] = (pcs @ d).abs().numpy()
        if l in args.layers:
            xt = x[torch.from_numpy(is_test)]
            # framing x role plane: x = framing, y = the part of role perpendicular to framing (a true 2D plane)
            f = framing_dir[l]
            r_perp = unit(role_dir[l] - (role_dir[l] @ f) * f)
            scatter[l] = {"plane": torch.stack([xt @ f, xt @ r_perp], 1).numpy(),
                          "pca": (xt @ pcs.T).numpy(), "angle": np.degrees(np.arccos(np.clip(cos_fr[l], -1, 1))),
                          "var": var.numpy()}

    print("layer  cos(framing, role)  |cos| framing~PC1  role~PC1  framing~PC2  role~PC2   var PC1  PC2")
    for l in range(num_layers):
        print(f"L{l:<4} {cos_fr[l]:+18.2f} {pc_cos['framing'][l, 0]:17.2f} {pc_cos['role'][l, 0]:9.2f} "
              f"{pc_cos['framing'][l, 1]:12.2f} {pc_cos['role'][l, 1]:9.2f} {pc_var[l, 0]:9.2f} {pc_var[l, 1]:4.2f}")

    model = fit.get("model") or ""
    layers = np.arange(num_layers)

    # Figure 1: by layer
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    band = 2 / np.sqrt(d_model)   # ~95% of cosines between two random directions in d_model dims
    ax.axhspan(-band, band, color=GRID, alpha=0.9, lw=0)
    ax.text(num_layers - 1, band, "two random directions", ha="right", va="bottom", fontsize=8, color=INK_2)
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.plot(layers, cos_fr, color=INK, lw=2, marker="o", ms=4)
    ax.set(ylim=(-1, 1), xlabel="layer", ylabel="cosine")
    ax.set_title("Framing vs role: 0 = perpendicular (independent)", fontsize=10, color=INK, loc="left")
    style(ax)

    ax = axes[1]
    for name, color in DIRECTION_COLORS.items():
        ax.plot(layers, pc_cos[name][:, 0], color=color, lw=2, label=f"{name} vs PC1")
        ax.plot(layers, pc_cos[name][:, 1], color=color, lw=2, ls="--", label=f"{name} vs PC2")
    ax.set(ylim=(0, 1.02), xlabel="layer", ylabel="|cosine|")
    ax.set_title("Does PCA find them? 1 = PCA's direction is the same one", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False)
    style(ax)

    ax = axes[2]
    ax.plot(layers, pc_var[:, 0], color=INK, lw=2, label="PC1")
    ax.plot(layers, pc_var[:, 1], color=INK_2, lw=2, ls="--", label="PC2")
    ax.set(ylim=(0, 1), xlabel="layer", ylabel="fraction of variance")
    ax.set_title("Variance explained (within review)", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False)
    style(ax)
    fig.suptitle(f"{model} | anchor: {args.anchor}", fontsize=11, color=INK)
    fig.tight_layout()
    path = os.path.join(args.diffmeans_dir, args.anchor, "geometry_by_layer.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

    # Figure 2: scatters at chosen layers
    shown = [l for l in args.layers if l in scatter]
    fig, axes = plt.subplots(2, len(shown), figsize=(4.2 * len(shown), 8.4), squeeze=False)
    test_role, test_framing = role_lab[is_test], framing_lab[is_test]
    for j, l in enumerate(shown):
        s = scatter[l]
        for i, (key, xlabel, ylabel, title) in enumerate([
            ("plane", "framing  (− negative · positive +)", "role  (− enemy · doctor +)",
             f"Layer {l}: framing × role plane\nangle between them {s['angle']:.0f}°"),
            ("pca", f"PC1 ({100 * s['var'][0]:.0f}% of variance)", f"PC2 ({100 * s['var'][1]:.0f}%)",
             f"Layer {l}: PCA, no labels used"),
        ]):
            ax = axes[i, j]
            pts = s[key]
            for role in [NO_ROLE] + roles:
                for framing in FRAMINGS:
                    m = (test_role == role) & (test_framing == framing)
                    ax.scatter(pts[m, 0], pts[m, 1], s=14, alpha=0.35, color=ROLE_COLORS[role],
                               marker=FRAMING_MARKERS[framing], lw=0)
                    ax.scatter(*pts[m].mean(0), s=110, color=ROLE_COLORS[role], marker=FRAMING_MARKERS[framing],
                               edgecolor="white", lw=2, zorder=3)
                c = pts[test_role == role].mean(0)   # direct label at the role's centre
                ax.annotate(role, c, xytext=(6, 6), textcoords="offset points", fontsize=8, color=INK, zorder=4)
            ax.set_xlabel(xlabel, fontsize=8, color=INK_2)
            ax.set_ylabel(ylabel, fontsize=8, color=INK_2)
            ax.set_title(title, fontsize=9, color=INK, loc="left")
            style(ax)
    handles = [plt.Line2D([], [], ls="", marker="o", ms=8, color=ROLE_COLORS[r], label=r) for r in [NO_ROLE] + roles]
    handles += [plt.Line2D([], [], ls="", marker=FRAMING_MARKERS[f], ms=8, color=INK_2, label=f"{f} claim") for f in FRAMINGS]
    fig.suptitle(f"{model} | anchor: {args.anchor} | held-out reviews, each review's mean removed | big marker = group mean",
                 fontsize=10, color=INK, y=0.995)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.97), ncol=len(handles), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = os.path.join(args.diffmeans_dir, args.anchor, "geometry_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orthogonality and PCA of the diffmeans framing and role directions")
    parser.add_argument("--diffmeans_dir", required=True, help="output dir of get_diffmeans_directions.py (has activations.pt).")
    parser.add_argument("--anchor", default="last", help="which anchor's directions and activations to use.")
    parser.add_argument("--layers", nargs="+", type=int, default=[8, 16, 24, 31], help="layers to show as scatters.")
    main(parser.parse_args())
