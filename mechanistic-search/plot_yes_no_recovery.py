"""
Yes/no role-patching recovery by layer, all models and prompts in one figure: one row per (model, prompt), one column
per setup (last, comma), one line per role (mean +/- SEM band). Reads <model>/<run_dir>/yes_no/<role>/summary.json.
Does not import run_role_patching.py, so it runs without torch/transformers.

    python plot_yes_no_recovery.py [--group all|positive|negative] [--roles ...] [--model_tags ...]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROMPTS = ["question_only", "articles"]   # same as run_role_patching.py
SETUPS = ["last", "comma"]
SETUP_TITLES = {"last": "last token → last token", "comma": "',' of 'My <role> said,' → start of claim"}
PROMPT_TITLES = {"question_only": "question only", "articles": "with article summaries"}
ROLE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # categorical slots in fixed order
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax) -> None:
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main(args):
    rows = []
    for model_tag in args.model_tags:
        for prompt in args.prompts:
            run_dir = "role_patching" if prompt == "articles" else f"role_patching_{prompt}"
            summaries = {}
            for role in args.roles:
                path = os.path.join(args.outputs_dir, model_tag, run_dir, "yes_no", role, "summary.json")
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        summaries[role] = json.load(f)
                else:
                    print(f"Missing {path}")
            if summaries:
                rows.append((model_tag, prompt, summaries))
    if not rows:
        return

    fig, axes = plt.subplots(len(rows), len(SETUPS), figsize=(6.2 * len(SETUPS), 3.1 * len(rows) + 0.6),
                             sharex=True, sharey=True, squeeze=False, facecolor="white")
    for r, (model_tag, prompt, summaries) in enumerate(rows):
        for c, setup in enumerate(SETUPS):
            ax = axes[r][c]
            ax.axhspan(0, 1, color=GRID, alpha=0.25, zorder=0)
            ax.axhline(0, color=MUTED, linewidth=1)
            ax.axhline(1, color=MUTED, linewidth=1, linestyle="--")
            for i, role in enumerate(args.roles):
                by_layer = summaries.get(role, {}).get("recovery", {}).get(args.group, {}).get(setup)
                if not by_layer:
                    continue
                layers = sorted(int(l) for l in by_layer)
                mean = [by_layer[str(l)]["mean"] for l in layers]
                sem = [by_layer[str(l)]["sem"] for l in layers]
                n = by_layer[str(layers[0])]["n"]
                color = ROLE_COLORS[i]
                ax.fill_between(layers, [m - s for m, s in zip(mean, sem)], [m + s for m, s in zip(mean, sem)],
                                color=color, alpha=0.18, linewidth=0)
                ax.plot(layers, mean, color=color, linewidth=2, label=f"My {role} said, (n={n})")
            ax.legend(fontsize=8, frameon=False, loc="upper left")
            if r == 0:
                ax.set_title(SETUP_TITLES[setup], color=TEXT, fontsize=11)
            if c == 0:
                ax.set_ylabel(f"{model_tag}\n{PROMPT_TITLES[prompt]}\n\nrecovery", color=TEXT, fontsize=10)
            if r == len(rows) - 1:
                ax.set_xlabel("layer patched", color=TEXT, fontsize=10)
            style(ax)

    fig.suptitle(f"Yes/No answer: recovery of the role effect when patching 'My <role> said, <claim>' → '<claim>' "
                 f"({args.group} claims; mean ± SEM; 0 = no effect, 1 = full)", color=TEXT, fontsize=12, y=1.02)
    fig.tight_layout()
    out_path = os.path.join(args.outputs_dir, f"recovery_yes_no_{args.group}.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot yes/no role-patching recovery by layer (all models and prompts)")
    parser.add_argument("--roles", nargs="+", default=["doctor", "friend", "enemy"])
    parser.add_argument("--model_tags", nargs="+", default=["llama3-8B-instruct", "olmo2-7B-instruct"])
    parser.add_argument("--prompts", nargs="+", default=PROMPTS, choices=PROMPTS)
    parser.add_argument("--outputs_dir", default="./outputs")
    parser.add_argument("--group", default="all", choices=["all", "positive", "negative"])
    main(parser.parse_args())
