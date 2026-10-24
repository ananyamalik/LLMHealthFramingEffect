"""
Plots the role-patching summaries, one figure per model: one row per answer type (yes_no, full), one column per setup
(last, comma), one line per role. Missing models / answer types / roles are skipped, so it can be run while jobs are
still going.

    python plot_role_patching.py [--model_tags llama3-8B-instruct olmo3-7B-instruct] [--group all|positive|negative]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_role_patching import ROLES, SETUPS, TEMPLATES

SETUP_TITLES = {"last": "last token -> last token", "comma": "',' of 'My <role> said,' -> start of claim"}
ANSWER_TITLES = {"yes_no": "Yes/No answer", "full": "Full answer"}


def plot_model(model_tag: str, input_dir: str, group: str) -> None:
    summaries = {}
    for answer in TEMPLATES:
        for role in ROLES:
            path = os.path.join(input_dir, answer, role, "summary.json")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    summaries[(answer, role)] = json.load(f)
    answers = [a for a in TEMPLATES if any(k[0] == a for k in summaries)]
    if not answers:
        print(f"No summaries found in {input_dir}")
        return

    fig, axes = plt.subplots(len(answers), len(SETUPS), figsize=(6.5 * len(SETUPS), 4.2 * len(answers)),
                             sharey=True, squeeze=False)
    for row, answer in enumerate(answers):
        for col, setup in enumerate(SETUPS):
            ax = axes[row][col]
            for role in ROLES:
                summary = summaries.get((answer, role))
                by_layer = (summary or {}).get("recovery", {}).get(group, {}).get(setup)
                if not by_layer:
                    continue
                layers = [int(l) for l in by_layer]
                ax.errorbar(layers, [by_layer[str(l)]["mean"] for l in layers],
                            yerr=[by_layer[str(l)]["sem"] for l in layers], marker="o", markersize=3, capsize=2,
                            label=f"{role} (n={summary['num_pairs_with_effect']})")
            ax.axhline(0, color="grey", linewidth=0.8)
            ax.axhline(1, color="grey", linewidth=0.8, linestyle="--")
            ax.set_title(f"{ANSWER_TITLES[answer]}: {SETUP_TITLES[setup]}", fontsize=10)
            ax.set_xlabel("Layer patched")
            ax.legend(fontsize=8)
        axes[row][0].set_ylabel("Mean recovery +/- SEM")
    fig.suptitle(f"{model_tag}: 'My <role> said, <claim>' -> '<claim>' ({group} claims)")
    fig.tight_layout()
    path = os.path.join(input_dir, f"role_patching_{group}.png")
    fig.savefig(path, dpi=200)
    print(f"Saved {path}")

    print(f"\n{model_tag}: clean role effect, mean D(run 1) - D(run 2) (yes_no: > 0 = more 'Yes'):")
    for (answer, role), summary in summaries.items():
        parts = [f"{g} {summary['clean'][g]['effect']['mean']:+.2f}" for g in ["positive", "negative"]
                 if summary["clean"][g]["effect"]["n"]]
        print(f"  {answer:<7} {role:<8} " + ", ".join(parts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot role-patching recovery by layer")
    parser.add_argument("--model_tags", nargs="+", default=["llama3-8B-instruct", "olmo3-7B-instruct"])
    parser.add_argument("--outputs_dir", default="./outputs", help="summaries are read from <outputs_dir>/<model_tag>/role_patching/.")
    parser.add_argument("--group", default="all", choices=["all", "positive", "negative"])
    args = parser.parse_args()
    for model_tag in args.model_tags:
        plot_model(model_tag, os.path.join(args.outputs_dir, model_tag, "role_patching"), args.group)
