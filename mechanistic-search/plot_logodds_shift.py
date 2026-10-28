"""
Role-vs-no-source shift in the yes/no answer, plotted like Fig. 1 of Guo et al. (arXiv:2610.32055): the change in
the target answer's log-odds against its probability without the source. Reads the existing yes_no pairs.jsonl.

    target           the answer the claim asks for: "Yes" for positive claims ("will work"), "No" for negative ones
    L(prompt)        target log-odds over {Yes, No}: D for positive claims, -D for negative (D = log p(Yes) - log p(No))
    delta            L(run 1, "My <role> said, <claim>") - L(run 2, "<claim>"); > 0 means the role pulls toward the claim
    x                p(target | run 2) = sigmoid(L(run 2)), renormalized over {Yes, No}

Figures (in --outputs_dir), all on the same delta scale:
    logodds_shift_yes_no_<role>.png     per role, one panel per (model, prompt): points per (review, framing), a smoothed
                                        curve over all pairs, and the marginal histogram of delta stacked by framing
    logodds_shift_yes_no_roles.png      one panel per (model, prompt): delta per pair for each role, split by framing,
                                        with the mean and 95% CI
The printed table compares each role with the first one (--roles, default doctor) on the same pairs (paired Wilcoxon).

    python plot_logodds_shift.py [--roles doctor friend enemy] [--model_tags ...] [--prompts question_only articles]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t, wilcoxon

FRAMINGS = ["positive", "negative"]   # same as run_role_patching.py (not imported: that pulls in torch/transformers)
PROMPTS = ["question_only", "articles"]
FRAMING_COLORS = {"positive": "#2a78d6", "negative": "#eb6834"}
FRAMING_LABELS = {"positive": "positive claim (target: Yes)", "negative": "negative claim (target: No)"}
PROMPT_TITLES = {"question_only": "question only", "articles": "with article summaries"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_shifts(path: str) -> dict[str, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    sign = np.array([1.0 if r["framing"] == "positive" else -1.0 for r in records])
    l_source = sign * np.array([r["d_source"] for r in records])
    l_target = sign * np.array([r["d_target"] for r in records])
    return {"pair_key": np.array([r["pair_key"] for r in records]), "framing": np.array([r["framing"] for r in records]),
            "x": 1 / (1 + np.exp(-l_target)), "p_source": 1 / (1 + np.exp(-l_source)), "delta": l_source - l_target}


def smooth(x: np.ndarray, y: np.ndarray, frac: float, grid: np.ndarray) -> np.ndarray:
    """
    Tricube-weighted local mean over the nearest frac * n points. Local-constant rather than the paper's local-linear
    LOWESS: with ~30 pairs piled up near p = 0 and p = 1, the local-linear fit extrapolates wildly across the gap.
    """
    k = max(int(np.ceil(frac * len(x))), 3)
    fitted = []
    for g in grid:
        dist = np.abs(x - g)
        h = np.sort(dist)[k - 1] + 1e-12
        w = np.clip(1 - (dist / h) ** 3, 0, None) ** 3
        fitted.append(np.sum(w * y) / np.sum(w))
    return np.array(fitted)


def style(ax) -> None:
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_role(args, role: str, panels: list, ylim: tuple) -> None:
    bins = np.linspace(*ylim, 25)
    n_cols = len(args.prompts)
    n_rows = int(np.ceil(len(panels) / n_cols))
    fig = plt.figure(figsize=(5.6 * n_cols, 4.2 * n_rows + 0.6), facecolor="white")
    outer = fig.add_gridspec(n_rows, n_cols, wspace=0.28, hspace=0.45)
    for i, (model_tag, prompt, s) in enumerate(panels):
        inner = outer[i // n_cols, i % n_cols].subgridspec(1, 2, width_ratios=[4, 1.2], wspace=0.08)
        ax, ax_hist = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
        for framing in FRAMINGS:
            m = s["framing"] == framing
            ax.scatter(s["x"][m], s["delta"][m], s=42, color=FRAMING_COLORS[framing], edgecolor="white",
                       linewidth=1.5, alpha=0.9, zorder=3, label=FRAMING_LABELS[framing])
        ax_hist.hist([s["delta"][s["framing"] == f] for f in FRAMINGS], bins=bins, orientation="horizontal",
                     stacked=True, color=[FRAMING_COLORS[f] for f in FRAMINGS], edgecolor="white", linewidth=1)
        grid = np.linspace(s["x"].min(), s["x"].max(), 100)
        ax.plot(grid, smooth(s["x"], s["delta"], args.frac, grid), color=TEXT, linewidth=2, zorder=4,
                label="smoothed mean (all pairs)")
        ax.axhline(0, color=MUTED, linewidth=1)
        mean = s["delta"].mean()
        ax_hist.axhline(mean, color=TEXT, linestyle=":", linewidth=1.5)
        ax_hist.text(0.97, 0.98, f"mean {mean:+.2f}", transform=ax_hist.transAxes, va="top", ha="right", fontsize=8,
                     color=TEXT)

        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(ylim)
        ax_hist.set_ylim(ylim)
        ax.set_xlabel("p(target) without the source", color=TEXT, fontsize=10)
        ax.set_ylabel(f"Δ target log-odds ({role} − none)", color=TEXT, fontsize=10)
        ax.set_title(f"{model_tag} · {PROMPT_TITLES[prompt]} (n={len(s['delta'])})", color=TEXT, fontsize=11, loc="left")
        style(ax)
        style(ax_hist)
        ax_hist.set_yticklabels([])
        ax_hist.set_xlabel("pairs", color=MUTED, fontsize=9)

    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Yes/No answer: shift in target log-odds from adding “My {role} said,”", color=TEXT,
                 fontsize=13, y=1.04)
    out_path = os.path.join(args.outputs_dir, f"logodds_shift_yes_no_{role}.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def mean_ci(values: np.ndarray) -> tuple[float, float]:
    """Mean and half-width of its 95% t interval."""
    if len(values) < 2:
        return float(np.mean(values)), 0.0
    return float(values.mean()), float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))


def plot_roles(args, shifts: dict, ylim: tuple) -> None:
    """One panel per (model, prompt): each role's per-pair delta, split by framing, with mean and 95% CI."""
    keys = [(m, p) for m in args.model_tags for p in args.prompts if any((m, p, r) in shifts for r in args.roles)]
    n_cols = len(args.prompts)
    n_rows = int(np.ceil(len(keys) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.0 * n_rows + 0.6), squeeze=False,
                             sharey=True, facecolor="white")
    rng = np.random.default_rng(0)
    offsets = {"positive": -0.17, "negative": 0.17}
    for i, (model_tag, prompt) in enumerate(keys):
        ax = axes[i // n_cols][i % n_cols]
        for j, role in enumerate(args.roles):
            s = shifts.get((model_tag, prompt, role))
            if s is None:
                continue
            for framing in FRAMINGS:
                d = s["delta"][s["framing"] == framing]
                x = j + offsets[framing]
                ax.scatter(x + rng.uniform(-0.07, 0.07, len(d)), d, s=22, color=FRAMING_COLORS[framing],
                           edgecolor="white", linewidth=1, alpha=0.55, zorder=3,
                           label=FRAMING_LABELS[framing] if j == 0 else None)
                mean, ci = mean_ci(d)
                ax.errorbar(x, mean, yerr=ci, fmt="o", markersize=8, color=FRAMING_COLORS[framing], ecolor=TEXT,
                            elinewidth=1.5, capsize=4, markeredgecolor=TEXT, markeredgewidth=1.2, zorder=5)
            mean, _ = mean_ci(s["delta"])
            ax.text(j, ylim[1], f"all {mean:+.2f}", ha="center", va="top", fontsize=9, color=TEXT)
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_xticks(range(len(args.roles)))
        ax.set_xticklabels([f"My {r} said," for r in args.roles], color=TEXT, fontsize=10)
        ax.set_xlim(-0.6, len(args.roles) - 0.4)
        ax.set_ylim(ylim)
        ax.set_title(f"{model_tag} · {PROMPT_TITLES[prompt]}", color=TEXT, fontsize=11, loc="left")
        if i % n_cols == 0:
            ax.set_ylabel("Δ target log-odds (role − none)", color=TEXT, fontsize=10)
        style(ax)
        ax.grid(False, axis="x")
    for ax in axes.flat[len(keys):]:
        ax.set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], marker="o", color=TEXT, markerfacecolor="white", linestyle="-", markersize=7))
    labels.append("mean ± 95% CI")
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Yes/No answer: shift in target log-odds by source role (> 0 = toward the claim)", color=TEXT,
                 fontsize=13, y=1.05)
    fig.tight_layout()
    out_path = os.path.join(args.outputs_dir, "logodds_shift_yes_no_roles.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def print_stats(args, shifts: dict) -> None:
    """Per role: mean delta and p(target) change, flips, Wilcoxon vs. 0 and paired vs. the first role."""
    reference = args.roles[0]
    print(f"\n{'model':<20}{'prompt':<15}{'role':<8}{'framing':<10}{'n':>3}{'mean Δ':>9}{'p(Δ=0)':>9}{'Δp':>8}"
          f"{'flips +/-':>11}{'Δ - ' + reference:>12}{'p(paired)':>11}")
    for model_tag in args.model_tags:
        for prompt in args.prompts:
            ref = shifts.get((model_tag, prompt, reference))
            for role in args.roles:
                s = shifts.get((model_tag, prompt, role))
                if s is None:
                    continue
                for group in ["all"] + FRAMINGS:
                    m = np.ones(len(s["delta"]), bool) if group == "all" else s["framing"] == group
                    d = s["delta"][m]
                    toward = int(np.sum((s["x"][m] < 0.5) & (s["p_source"][m] > 0.5)))
                    away = int(np.sum((s["x"][m] > 0.5) & (s["p_source"][m] < 0.5)))
                    line = (f"{model_tag:<20}{prompt:<15}{role:<8}{group:<10}{len(d):>3}{d.mean():>+9.2f}"
                            f"{wilcoxon(d).pvalue:>9.3f}{np.mean(s['p_source'][m] - s['x'][m]):>+8.3f}"
                            f"{f'{toward}/{away}':>11}")
                    if ref is not None and role != reference:
                        ref_delta = dict(zip(ref["pair_key"], ref["delta"]))
                        shared = [k in ref_delta for k in s["pair_key"][m]]
                        diff = d[shared] - np.array([ref_delta[k] for k in s["pair_key"][m][shared]])
                        line += f"{diff.mean():>+12.2f}{wilcoxon(diff).pvalue:>11.3f}"
                    print(line)


def main(args):
    shifts = {}   # (model, prompt, role) -> load_shifts
    for model_tag in args.model_tags:
        for prompt in args.prompts:
            run_dir = "role_patching" if prompt == "articles" else f"role_patching_{prompt}"
            for role in args.roles:
                path = os.path.join(args.outputs_dir, model_tag, run_dir, "yes_no", role, "pairs.jsonl")
                if os.path.exists(path):
                    shifts[(model_tag, prompt, role)] = load_shifts(path)
                else:
                    print(f"Missing {path}")
    if not shifts:
        return

    all_delta = np.concatenate([s["delta"] for s in shifts.values()])
    pad = 0.08 * (all_delta.max() - all_delta.min())
    ylim = (all_delta.min() - pad, all_delta.max() + pad)
    for role in args.roles:
        panels = [(m, p, shifts[(m, p, role)]) for m in args.model_tags for p in args.prompts if (m, p, role) in shifts]
        if panels:
            plot_role(args, role, panels, ylim)
    plot_roles(args, shifts, ylim)
    print_stats(args, shifts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the role's shift in target log-odds (yes/no answers)")
    parser.add_argument("--roles", nargs="+", default=["doctor", "friend", "enemy"],
                        help="roles to plot; the first is the reference for the paired comparison.")
    parser.add_argument("--model_tags", nargs="+", default=["llama3-8B-instruct", "olmo2-7B-instruct"])
    parser.add_argument("--prompts", nargs="+", default=PROMPTS, choices=PROMPTS)
    parser.add_argument("--outputs_dir", default="./outputs")
    parser.add_argument("--frac", default=0.4, type=float, help="smoothing neighborhood fraction.")
    main(parser.parse_args())
