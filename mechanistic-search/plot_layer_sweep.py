"""Plot mean recovery and first-token restoration by layer from layer_sweep_summary.json."""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_layer_sweep import SUMMARY_FILE_NAME


def main(sweep_dir: str) -> None:
    with open(os.path.join(sweep_dir, SUMMARY_FILE_NAME), "r", encoding="utf-8") as f:
        summary = json.load(f)

    layers = [int(l) for l in summary["by_layer"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    for ax, metric, title in [
        (axes[0], "recovery", "Framing effect recovered\n(teacher-forced log-likelihood ratio)"),
        (axes[1], "first_token_restoration", "First-token distribution restored\n(1 - KL_patched / KL_clean)"),
    ]:
        means = [summary["by_layer"][str(l)][metric]["mean"] for l in layers]
        sems = [summary["by_layer"][str(l)][metric]["sem"] for l in layers]
        ax.errorbar(layers, means, yerr=sems, marker="o", capsize=3)
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.axhline(1, color="grey", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Layer patched (last prompt token)")
        ax.set_title(title)
    axes[0].set_ylabel("Mean +/- SEM")
    fig.suptitle(f"Positive -> negative last-token patching (n={summary['num_pairs_with_effect']} pairs)")
    fig.tight_layout()
    path = os.path.join(sweep_dir, "layer_sweep.png")
    fig.savefig(path, dpi=200)
    print(f"Saved {path}")

    # Per question type
    fig, ax = plt.subplots(figsize=(8, 5))
    for q_type, by_layer in summary["by_question_type"].items():
        q_layers = [int(l) for l in by_layer]
        ax.plot(q_layers, [by_layer[str(l)]["mean"] for l in q_layers], marker=".", label=q_type)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.axhline(1, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Layer patched (last prompt token)")
    ax.set_ylabel("Mean recovery")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = os.path.join(sweep_dir, "layer_sweep_by_question_type.png")
    fig.savefig(path, dpi=200)
    print(f"Saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot layer sweep results")
    parser.add_argument("--sweep_dir", required=True, help="output directory of run_layer_sweep.py.")
    main(parser.parse_args().sweep_dir)
