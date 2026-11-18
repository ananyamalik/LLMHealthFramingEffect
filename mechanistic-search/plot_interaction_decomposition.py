"""
The role x framing interaction vector of each role, decomposed along the framing axis (role_framing_vectors.py).

Per role and layer, in units of |f| (f = framing without a role, drawn with length 1 along x):
    x = inter . f^ / |f|        the gain: < 0, the role shrinks the framing gap along f
    y = |inter_perp| / |f|      the size of the part off the framing axis
Each role's off-axis part points in its own direction in the residual stream (not a shared y-axis), so y is a
length only; the arrows show how much of each interaction is along vs off the framing axis.

    python plot_interaction_decomposition.py --vectors_dir outputs/olmo2-7B-instruct/diffmeans/last/role_framing_vectors
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROLE_COLORS = {"doctor": "#2a78d6", "friend": "#eb6834", "enemy": "#1baf7a"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#c9c8c3"


def load(vectors_dir: str, roles: list[str]) -> tuple[dict, str]:
    out, model = {}, None
    for role in roles:
        with open(os.path.join(vectors_dir, role, "summary.json"), encoding="utf-8") as f:
            s = json.load(f)
        out[role], model = s["layers"], s.get("model")
    return out, model


def panel(ax, by_role: dict, layer: int, roles: list[str], xlim: tuple, ylim: tuple) -> None:
    pts = {}
    for role in roles:
        s = by_role[role][layer]
        pts[role] = (s["gain"], s["norm"]["inter_perp"] / s["norm"]["f"], s["ci"]["gain"])
    x0, x1 = xlim
    y_bar0 = -0.06 * (ylim[1] - ylim[0])
    step = 0.055 * (ylim[1] - ylim[0])

    # framing axis and f (length 1)
    ax.annotate("", xy=(x1, 0), xytext=(x0, 0), arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=1))
    ax.annotate("", xy=(1, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.2))
    ax.text(1, 0.02 * ylim[1], "f  (length 1)", ha="right", va="bottom", fontsize=9, color=INK)
    ax.axvline(0, color=GRID, ls=":", lw=1)
    ax.text(0.02, ylim[1] * 0.98, "off the framing axis ↑", ha="left", va="top", fontsize=8.5, color=INK_2)

    # labels above the arrow tips, pushed apart vertically so they do not overlap
    gap = 0.085 * ylim[1]
    label_y, last = {}, -1e9
    for role in sorted(roles, key=lambda r: pts[r][1]):
        label_y[role] = max(pts[role][1] + 0.03 * ylim[1], last + gap)
        last = label_y[role]
    for i, role in enumerate(roles):
        x, y, ci = pts[role]
        c = ROLE_COLORS.get(role, "gray")
        ax.annotate("", xy=(x, y), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=2))
        ax.plot([x, x], [0, y], color=c, ls="--", lw=0.8, alpha=0.6)
        if label_y[role] - y > 0.05 * ylim[1]:
            ax.plot([x, x], [y, label_y[role]], color=c, lw=0.6, alpha=0.5)
        ax.text(x, label_y[role], f"{role} I", ha="center", va="bottom", fontsize=10, color=c, fontweight="bold")
        # along-f component as a bar under the axis, with its bootstrap interval
        yb = y_bar0 - i * step
        ax.plot([x, 0], [yb, yb], color=c, lw=3, solid_capstyle="butt")
        ax.plot(ci, [yb, yb], color=c, lw=0.8, marker="|", ms=6)
        ax.text(0.04, yb, f"{role}: along f {x:+.2f}, off-axis {y:.2f}", ha="left", va="center", fontsize=8.5, color=c)

    ax.set(xlim=xlim, ylim=(y_bar0 - len(roles) * step, ylim[1]), title=f"layer {layer}")
    ax.set_xlabel("along f, in units of |f|   (< 0: the role shrinks the framing gap)", fontsize=9, color=INK_2)
    ax.set_ylabel("|off-axis part| / |f|", fontsize=9, color=INK_2)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.set_aspect("equal")


def main(args):
    by_role, model = load(args.vectors_dir, args.roles)
    gains = [by_role[r][l]["gain"] for r in args.roles for l in args.layers]
    perps = [by_role[r][l]["norm"]["inter_perp"] / by_role[r][l]["norm"]["f"] for r in args.roles for l in args.layers]
    xlim = (min(min(gains) - 0.15, -1.0), 1.75)
    ylim = (0, max(perps) * 1.45)
    w = 5.6
    h = w * (ylim[1] * 1.25) / (xlim[1] - xlim[0]) + 1.6   # equal aspect: height follows the data
    fig, axes = plt.subplots(1, len(args.layers), figsize=(w * len(args.layers), h), squeeze=False)
    for ax, layer in zip(axes[0], args.layers):
        panel(ax, by_role, layer, args.roles, xlim, ylim)
    fig.suptitle(f"Interaction vector I = d|pos − d|neg, decomposed along the framing axis f (helpful − not helpful)\n"
                 f"{model} | last prompt token | each role's off-axis part has its own direction: y is its length only",
                 fontsize=10.5)
    fig.tight_layout()
    path = args.out or os.path.join(args.vectors_dir, "interaction_decomposition.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    for layer in args.layers:
        print(f"L{layer}: " + "  ".join(f"{r} along {by_role[r][layer]['gain']:+.2f} off "
                                        f"{by_role[r][layer]['norm']['inter_perp'] / by_role[r][layer]['norm']['f']:.2f}"
                                        for r in args.roles))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interaction vector decomposed along the framing axis")
    parser.add_argument("--vectors_dir", required=True, help="role_framing_vectors.py output dir")
    parser.add_argument("--roles", nargs="+", default=["doctor", "friend", "enemy"])
    parser.add_argument("--layers", nargs="+", type=int, default=[16, 20, 31])
    parser.add_argument("--out", default=None, help="default: <vectors_dir>/interaction_decomposition.png")
    main(parser.parse_args())
