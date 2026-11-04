"""
How do the framing and role vectors of run_unit_steering.py / run_framing_role_mediation.py sit in the residual stream?

Vectors (from a directions.pt; last prompt token, per layer, see run_unit_steering.py):
    framing              [positive none] - [negative none]
    doctor/friend/enemy  ("My <role> said, <claim>" - "<claim>"), averaged over the two framings
    role                 the mean of doctor, friend and enemy (the shared "someone said it" part)
    <role>_specific      <role> - role (what is left of each role once the shared part is removed)

Per layer:
    cosines        between every pair of vectors; two random directions in d_model dims give |cos| < 2 / sqrt(d_model)
    in role span   the fraction of framing's (unit) length squared inside span(doctor, friend, enemy), ||P f||^2, and
                   the principal angle arccos ||P f|| (90 = framing is perpendicular to every mix of the role vectors).
                   A random direction puts 3 / d_model of itself there
    size           |vector| / mean |h| at the last token: how big each shift is next to the residual

Figures (next to the directions.pt, prefix = its file name):
    <prefix>_geometry_by_layer.png   the four measures above by layer
    <prefix>_cosines.png             the cosine matrix at --layers
    <prefix>_plane.png               at --layers: every vector drawn in the plane of framing (x) and the part of role
                                     perpendicular to it (y), in units of mean |h|; the share of each vector's length
                                     in that plane is in its label (the rest points out of the plane)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROLES = ["doctor", "friend", "enemy"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"framing": "#0b0b0b", "role": "#8f8e89", "doctor": "#2a78d6", "friend": "#eb6834", "enemy": "#1baf7a"}


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


def cos(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    return (unit(a) * unit(b)).sum(-1).numpy()


def span_fraction(v: torch.Tensor, basis: torch.Tensor) -> np.ndarray:
    """||P v||^2 / ||v||^2 per layer for v [layers, d_model] and the span of basis [layers, k, d_model]."""
    q = torch.linalg.qr(basis.transpose(1, 2)).Q            # [layers, d_model, k], orthonormal columns
    proj = torch.einsum("ldk,ld->lk", q, unit(v))
    return (proj ** 2).sum(-1).numpy()


def main(args):
    fit = torch.load(args.directions, map_location="cpu", weights_only=False)
    raw = {n: v.float() for n, v in fit["raw"].items()}
    for r in ROLES:
        raw[f"{r}_specific"] = raw[r] - raw["role"]
    residual_norm = fit["residual_norm"].float()
    num_layers, d_model = raw["framing"].shape
    layers = np.arange(num_layers)
    out_prefix = os.path.splitext(args.directions)[0]
    title = f"{fit.get('model', '')} | prompt: {fit.get('prompt', '?')} | last prompt token"

    in_span = span_fraction(raw["framing"], torch.stack([raw[r] for r in ROLES], 1))
    in_specific = span_fraction(raw["framing"], torch.stack([raw[f"{r}_specific"] for r in ROLES[:2]], 1))
    print(f"{args.directions}\nlayer  " + "  ".join(f"cos(framing,{n})" for n in ROLES + ["role"])
          + "  framing in role span  angle   |framing|/|h|  |role|/|h|")
    for l in layers:
        cs = "  ".join(f"{cos(raw['framing'][l], raw[n][l]):+{len(n) + 13}.2f}" for n in ROLES + ["role"])
        print(f"L{l:<4} {cs}  {in_span[l]:20.2f}  {np.degrees(np.arccos(np.sqrt(in_span[l]))):5.0f}  "
              f"{(raw['framing'][l].norm() / residual_norm[l]):13.3f}  {(raw['role'][l].norm() / residual_norm[l]):10.3f}")

    # Figure 1: by layer
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
    band = 2 / np.sqrt(d_model)

    ax = axes[0]
    ax.axhspan(-band, band, color=GRID, alpha=0.9, lw=0)
    ax.axhline(0, color=INK_2, lw=0.8)
    for n in ROLES + ["role"]:
        ax.plot(layers, cos(raw["framing"], raw[n]), color=COLORS[n], lw=2, label=n)
    for r in ROLES:
        ax.plot(layers, cos(raw["framing"], raw[f"{r}_specific"]), color=COLORS[r], lw=1.2, ls=":")
    ax.set(ylim=(-1, 1), xlabel="layer", ylabel="cosine")
    ax.set_title("cos(framing, ·)   dotted = role-specific part\ngrey band = two random directions",
                 fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False)
    style(ax)

    ax = axes[1]
    ax.axhspan(-band, band, color=GRID, alpha=0.9, lw=0)
    ax.axhline(0, color=INK_2, lw=0.8)
    for a, b in [("doctor", "friend"), ("doctor", "enemy"), ("friend", "enemy")]:
        ax.plot(layers, cos(raw[a], raw[b]), color=COLORS[a], lw=2, label=f"{a} · {b}",
                ls={"friend": "-", "enemy": "--"}[b] if a == "doctor" else "-.")
    for a, b in [("doctor", "friend"), ("doctor", "enemy"), ("friend", "enemy")]:
        ax.plot(layers, cos(raw[f"{a}_specific"], raw[f"{b}_specific"]), color=INK_2, lw=1, ls=":")
    ax.set(ylim=(-1, 1), xlabel="layer", ylabel="cosine")
    ax.set_title("Roles with each other\n(dotted = role-specific parts)", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False)
    style(ax)

    ax = axes[2]
    ax.axhline(3 / d_model, color=INK_2, lw=0.8, ls="--")
    ax.plot(layers, in_span, color=INK, lw=2, marker="o", ms=4, label="span(doctor, friend, enemy)")
    ax.plot(layers, in_specific, color=COLORS["doctor"], lw=1.5, label="span of role-specific parts")
    ax.plot(layers, cos(raw["framing"], raw["role"]) ** 2, color=COLORS["role"], lw=1.5, label="shared role vector")
    ax.set(ylim=(0, 1), xlabel="layer", ylabel="fraction of |framing|²")
    ax2 = ax.secondary_yaxis("right", functions=(lambda x: np.degrees(np.arccos(np.sqrt(np.clip(x, 0, 1)))),
                                                 lambda a: np.cos(np.radians(a)) ** 2))
    ax2.set_ylabel("principal angle (°)", fontsize=8, color=INK_2)
    ax2.set_yticks([90, 75, 60, 45, 30, 0])
    ax2.tick_params(colors=INK_2, labelsize=8)
    ax.set_title("How much of framing lies in the role subspace\n(dashed = a random direction)",
                 fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False)
    style(ax)

    ax = axes[3]
    for n in ["framing"] + ROLES + ["role"]:
        ax.plot(layers, (raw[n].norm(dim=-1) / residual_norm).numpy(), color=COLORS[n], lw=2, label=n)
    ax.set(ylim=(0, None), xlabel="layer", ylabel="|vector| / mean |h|")
    ax.set_title("Size of each shift next to the residual", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False)
    style(ax)
    fig.suptitle(title, fontsize=11, color=INK)
    fig.tight_layout()
    path = f"{out_prefix}_geometry_by_layer.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

    # Figure 2: cosine matrices
    names = ["framing"] + ROLES + ["role"] + [f"{r}_specific" for r in ROLES]
    shown = [l for l in args.layers if l < num_layers]
    fig, axes = plt.subplots(1, len(shown), figsize=(4.4 * len(shown), 4.4), squeeze=False)
    for ax, l in zip(axes[0], shown):
        u = unit(torch.stack([raw[n][l] for n in names]))
        m = (u @ u.T).numpy()
        ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(m[i, j]) > 0.6 else INK)
        labels = [n.replace("_specific", " spec") for n in names]
        ax.set_xticks(range(len(names)), labels, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(names)), labels, fontsize=8)
        ax.set_title(f"Layer {l}", fontsize=10, color=INK, loc="left")
    fig.suptitle(title + " | cosine", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = f"{out_prefix}_cosines.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

    # Figure 3: the framing x role plane
    fig, axes = plt.subplots(1, len(shown), figsize=(4.4 * len(shown), 4.6), squeeze=False)
    for ax, l in zip(axes[0], shown):
        f = unit(raw["framing"][l])
        r_perp = unit(raw["role"][l] - (raw["role"][l] @ f) * f)
        lim = 0
        for n in ["framing"] + ROLES + ["role"]:
            v = raw[n][l] / residual_norm[l]
            x, y = (v @ f).item(), (v @ r_perp).item()
            share = (x ** 2 + y ** 2) / (v @ v).item()
            ax.annotate("", (x, y), (0, 0), arrowprops=dict(arrowstyle="-|>", color=COLORS[n], lw=2))
            ax.plot([], [], color=COLORS[n], lw=2, label=f"{n} ({100 * share:.0f}%)")
            lim = max(lim, abs(x), abs(y))
        ax.set(xlim=(-1.2 * lim, 1.25 * lim), ylim=(-1.2 * lim, 1.25 * lim), aspect="equal")
        ax.axhline(0, color=INK_2, lw=0.6)
        ax.axvline(0, color=INK_2, lw=0.6)
        angle = np.degrees(np.arccos(np.clip(cos(raw["framing"][l], raw["role"][l]), -1, 1)))
        ax.set_xlabel("framing  (negative → positive)", fontsize=8, color=INK_2)
        ax.set_ylabel("role, perpendicular to framing", fontsize=8, color=INK_2)
        ax.set_title(f"Layer {l}: angle(framing, role) {angle:.0f}°", fontsize=10, color=INK, loc="left")
        ax.legend(fontsize=7.5, frameon=False, loc="lower left")
        style(ax)
    fig.suptitle(title + " | units of mean |h| | % = share of the vector's length² in this plane",
                 fontsize=10, color=INK)
    fig.tight_layout()
    path = f"{out_prefix}_plane.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Angles, overlap and size of the framing and role vectors")
    parser.add_argument("--directions", required=True, help="a directions.pt of run_unit_steering.py / run_framing_role_mediation.py.")
    parser.add_argument("--layers", nargs="+", type=int, default=[4, 12, 20, 28], help="layers for the matrix and plane figures.")
    main(parser.parse_args())
