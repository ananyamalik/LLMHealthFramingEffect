"""
Difference-of-means directions for the two axes of the question-only prompt: framing (positive / negative claim) and
role (none / doctor / friend / enemy in "My <role> said, <claim>").

Data: residuals [rows, layers, d_model] at the anchors of run_belief_probe.py (and the yes/no prompt's last token), one
row per (review, framing, role), in the question-only prompts. If --acts_path does not exist yet, the model is loaded
and they are collected here (run_belief_probe.collect: every review x both framings x bare claim + --roles) and saved
there, in run_belief_probe.py's activations.pt format; later runs (or a belief probe activations.pt passed as
--acts_path) skip the model. Only reviews with every (framing, role) row are used, and every direction is an average
of cell means (cell = framing x role), so neither axis leaks into the other through unequal cell sizes.

Directions per anchor and layer (mu[f, r] = mean residual of cell (f, r), over the training reviews):
    framing             mean over roles (incl. none) of mu[positive, r] - mu[negative, r]
    framing_<role>      the same within one role (are they one direction?)
    <role>_vs_none      r_k = mean over framings of mu[f, k] - mu[f, none]
    shared              s = mean_k r_k: what every role has in common vs the bare claim ("has a role": the longer
                        prompt, the "My _ said," template, a generic persona)
    <role>_specific     u_k = r_k - s = mean_f mu[f, k] - mean over roles of mu[f, role]. Only role rows enter it, and
                        the role prompts have the same length, so length and template cancel
    <a>_minus_<b>       mean over framings of mu[f, a] - mu[f, b], between roles (= u_a - u_b)
Since sum_k u_k = 0, mean_k |r_k|^2 = |s|^2 + mean_k |u_k|^2: shared_share = |s|^2 / mean_k |r_k|^2 is the part of
the role-vs-none effect that is common to all roles.

Checks per layer (summary.json): vector norms; cosines between the framing vectors of different roles, between the
r_k, and between framing and each role direction (~0 if the axes are distinct); and on held-out reviews
(--test_fraction, both framings and every role of a review on the same side), the AUC of the projection onto each
direction for its own contrast and, as a leak check, for the other axis' contrasts (0.5 = no separation):
    framing        positive vs negative                      leak: shared, <a>_minus_<b>
    shared         role rows vs bare rows                    leak: framing
    <role>_specific that role vs the other roles             leak: framing
    <a>_minus_<b>  a vs b                                    leak: framing
Prompt lengths per role are saved too (how much longer the role prompts are than the bare claim).

Outputs (in --output_dir):
    activations.pt  (default --acts_path) the collected residuals and row metadata, ~0.8 GB per anchor for all reviews
Per anchor, in --output_dir/<anchor>/:
    directions.pt   {name: [layers, d_model] float32} fit on the training reviews, train/test reviews, roles
    summary.json    the checks above per layer
    diffmeans.png   norms, cosines, shared_share and held-out AUCs by layer
"""
import argparse
import itertools
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from patching_utils import DEFAULT_MODEL_NAME, SEED, set_global_seed, load_model_and_tokenizer, load_dataset, sample_reviews
from run_belief_probe import NO_ROLE, ANCHORS, YES_NO, FRAMING_COLORS, ROLE_COLORS, MUTED, auc, collect
from run_role_patching import FRAMINGS, ROLES
from get_steering_vector import split_reviews


def complete_reviews(rows: list[dict], roles: list[str]) -> list[str]:
    """Reviews with a row for every (framing, role), so every cell has the same reviews."""
    have = {}
    for r in rows:
        have.setdefault(r["review_id"], set()).add((r["framing"], r["role"]))
    need = {(f, r) for f in FRAMINGS for r in [NO_ROLE] + roles}
    return sorted(rid for rid, cells in have.items() if need <= cells)


def cell_means(x: torch.Tensor, rows: list[dict], reviews: set[str], roles: list[str]) -> dict[tuple[str, str], torch.Tensor]:
    """mu[(framing, role)] = mean residual [layers, d_model] over `reviews`."""
    mu = {}
    for f in FRAMINGS:
        for role in [NO_ROLE] + roles:
            idx = [i for i, r in enumerate(rows) if r["review_id"] in reviews and r["framing"] == f and r["role"] == role]
            mu[(f, role)] = x[idx].float().mean(0)
    return mu


def fit_directions(mu: dict, roles: list[str]) -> dict[str, torch.Tensor]:
    def over_framings(a, b):
        return torch.stack([mu[(f, a)] - mu[(f, b)] for f in FRAMINGS]).mean(0)

    d = {f"framing_{role}": mu[(FRAMINGS[0], role)] - mu[(FRAMINGS[1], role)] for role in [NO_ROLE] + roles}
    d["framing"] = torch.stack([d[f"framing_{role}"] for role in [NO_ROLE] + roles]).mean(0)
    for role in roles:
        d[f"{role}_vs_none"] = over_framings(role, NO_ROLE)
    d["shared"] = torch.stack([d[f"{role}_vs_none"] for role in roles]).mean(0)
    for role in roles:
        d[f"{role}_specific"] = d[f"{role}_vs_none"] - d["shared"]
    for a, b in itertools.combinations(roles, 2):
        d[f"{a}_minus_{b}"] = over_framings(a, b)
    return d


def cos(a: torch.Tensor, b: torch.Tensor) -> list[float]:
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).tolist()


def heldout_aucs(x: torch.Tensor, rows: list[dict], test: set[str], directions: dict, roles: list[str]) -> dict:
    """Per direction and contrast: AUC by layer of the projection of the test rows onto the direction."""
    idx = [i for i, r in enumerate(rows) if r["review_id"] in test]
    framing = np.array([rows[i]["framing"] for i in idx])
    role = np.array([rows[i]["role"] for i in idx])
    xt = x[idx].float()   # [test rows, layers, d_model]

    # contrast -> (rows to compare, label True for the first group)
    contrasts = {"framing": (np.ones(len(idx), bool), framing == FRAMINGS[0]),
                 "shared": (np.ones(len(idx), bool), role != NO_ROLE)}
    for k in roles:
        contrasts[f"{k}_specific"] = (role != NO_ROLE, role == k)
    for a, b in itertools.combinations(roles, 2):
        contrasts[f"{a}_minus_{b}"] = ((role == a) | (role == b), role == a)
    role_contrasts = [c for c in contrasts if c != "framing"]
    checks = {"framing": ["framing"] + role_contrasts} | {c: [c, "framing"] for c in role_contrasts}

    out = {}
    for name, tested in checks.items():
        unit = directions[name] / directions[name].norm(dim=-1, keepdim=True)
        proj = torch.einsum("nld,ld->nl", xt, unit).numpy()   # [test rows, layers]
        out[name] = {}
        for c in tested:
            keep, label = contrasts[c]
            out[name][c] = [auc(proj[keep, l], label[keep]) for l in range(proj.shape[1])]
    return out


def summarize(directions: dict, aucs, roles: list[str], lengths: dict) -> dict:
    names = ["framing", "shared"] + [f"{k}_vs_none" for k in roles] + [f"{k}_specific" for k in roles] \
        + [f"{a}_minus_{b}" for a, b in itertools.combinations(roles, 2)]
    r_sq = torch.stack([directions[f"{k}_vs_none"].norm(dim=-1) ** 2 for k in roles]).mean(0)
    summary = {
        "roles": roles,
        "prompt_length": lengths,
        "norm": {n: directions[n].norm(dim=-1).tolist() for n in names},
        # Is framing one direction across roles?
        "cos_framing_between_roles": {f"{a}|{b}": cos(directions[f"framing_{a}"], directions[f"framing_{b}"])
                                      for a, b in itertools.combinations([NO_ROLE] + roles, 2)},
        # How much do the roles share vs the bare claim?
        "cos_vs_none_between_roles": {f"{a}|{b}": cos(directions[f"{a}_vs_none"], directions[f"{b}_vs_none"])
                                      for a, b in itertools.combinations(roles, 2)},
        "shared_share": (directions["shared"].norm(dim=-1) ** 2 / r_sq).tolist(),
        # Are framing and role distinct axes?
        "cos_framing_vs": {n: cos(directions["framing"], directions[n]) for n in names if n != "framing"},
    }
    if aucs is not None:
        summary["heldout_auc"] = aucs
    return summary


def plot(summary: dict, title: str, path: str) -> None:
    roles = summary["roles"]
    layers = np.arange(len(summary["shared_share"]))
    role_color = {k: ROLE_COLORS[i % len(ROLE_COLORS)] for i, k in enumerate(roles)}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))

    ax = axes[0, 0]
    ax.plot(layers, summary["norm"]["framing"], color=FRAMING_COLORS["positive"], label="framing")
    ax.plot(layers, summary["norm"]["shared"], color=MUTED, label="shared (has a role)")
    for k in roles:
        ax.plot(layers, summary["norm"][f"{k}_specific"], color=role_color[k], label=f"{k}_specific")
    ax.set(title="Direction norms", xlabel="layer", ylabel="L2 norm")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(layers, summary["cos_framing_vs"]["shared"], color=MUTED, label="shared")
    for k in roles:
        ax.plot(layers, summary["cos_framing_vs"][f"{k}_specific"], color=role_color[k], label=f"{k}_specific")
    for a, b in itertools.combinations(roles, 2):
        ax.plot(layers, summary["cos_framing_vs"][f"{a}_minus_{b}"], ls="--", label=f"{a}_minus_{b}")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set(title="cos(framing, role direction): ~0 if the axes are distinct", xlabel="layer", ylabel="cosine")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(layers, summary["shared_share"], color="black", lw=2, label="shared_share = |s|² / mean |r_k|²")
    for pair, values in summary["cos_vs_none_between_roles"].items():
        ax.plot(layers, values, ls="--", label=f"cos(r_{pair.replace('|', ', r_')})")
    for pair, values in summary["cos_framing_between_roles"].items():
        ax.plot(layers, values, ls=":", lw=0.9, label=f"cos(framing_{pair.replace('|', ', framing_')})")
    ax.set(title="Shared part of role vs none; framing consistency across roles", xlabel="layer", ylim=(-0.05, 1.05))
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1, 1]
    if "heldout_auc" in summary:
        a = summary["heldout_auc"]
        ax.plot(layers, a["framing"]["framing"], color=FRAMING_COLORS["positive"], lw=2, label="framing: pos vs neg")
        ax.plot(layers, a["shared"]["shared"], color=MUTED, lw=2, label="shared: role vs none")
        for k in roles:
            ax.plot(layers, a[f"{k}_specific"][f"{k}_specific"], color=role_color[k], label=f"{k}_specific: {k} vs other roles")
        for x, y in itertools.combinations(roles, 2):
            ax.plot(layers, a[f"{x}_minus_{y}"][f"{x}_minus_{y}"], ls="--", label=f"{x}_minus_{y}: {x} vs {y}")
        # Leak: the framing direction on the role contrasts, and the role directions on framing
        leaks = [a["framing"][c] for c in a["framing"] if c != "framing"] + [a[n]["framing"] for n in a if n != "framing"]
        leak = np.max(np.abs(np.array(leaks) - 0.5), axis=0) + 0.5
        ax.plot(layers, leak, color="#d62728", ls=":", label="leak: worst cross-axis AUC (folded to >= 0.5)")
        ax.axhline(0.5, color="gray", lw=0.8)
        ax.set(title="Held-out AUC of the projection", xlabel="layer", ylabel="AUC", ylim=(0.3, 1.02))
        ax.legend(fontsize=7)
    else:
        ax.set_axis_off()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def load_or_collect(args) -> dict:
    """Residuals from --acts_path, or collected with the model (and saved there) if it does not exist yet."""
    if os.path.exists(args.acts_path):
        saved = torch.load(args.acts_path)
        print(f"Loaded {len(saved['rows'])} rows from {args.acts_path}")
        return saved
    if args.input_path is None:
        raise ValueError(f"{args.acts_path} does not exist: pass --input_path to collect the activations")
    set_global_seed(SEED)
    dataset = sample_reviews(load_dataset(args.input_path), args.num_reviews, seed=args.seed)
    print(f"No {args.acts_path}: collecting activations for {len(dataset)} reviews")
    model, tokenizer = load_model_and_tokenizer(args.model)
    rows, acts = collect(model, tokenizer, dataset, args.roles, ANCHORS)
    del model
    torch.cuda.empty_cache()
    saved = {"rows": rows, "acts": acts, "roles": args.roles, "anchors": ANCHORS, "model": args.model}
    os.makedirs(os.path.dirname(args.acts_path) or ".", exist_ok=True)
    torch.save(saved, args.acts_path)
    print(f"Saved {args.acts_path}")
    return saved


def main(args):
    saved = load_or_collect(args)
    rows, acts, roles = saved["rows"], saved["acts"], saved["roles"]
    anchors = args.anchors or list(acts)
    reviews = complete_reviews(rows, roles)
    test = split_reviews(reviews, args.test_fraction, args.seed) if args.test_fraction > 0 else set()
    train = set(reviews) - test
    print(f"{len(rows)} rows, roles {roles} | {len(reviews)} complete reviews: {len(train)} train / {len(test)} test")

    # Full-prompt length per role (the yes/no prompt adds the same line to every role)
    lengths = {role: float(np.mean([r["positions"]["last"] + 1 for r in rows if r["role"] == role])) for role in [NO_ROLE] + roles}
    print("Mean full-prompt length (tokens): " + ", ".join(f"{k} {v:.1f}" for k, v in lengths.items()))

    for anchor in anchors:
        x = acts[anchor]
        directions = fit_directions(cell_means(x, rows, train, roles), roles)
        aucs = heldout_aucs(x, rows, test, directions, roles) if test else None
        summary = summarize(directions, aucs, roles, lengths) | {
            "anchor": anchor, "model": saved.get("model"), "n_train_reviews": len(train), "n_test_reviews": len(test)}

        out_dir = os.path.join(args.output_dir, anchor)
        os.makedirs(out_dir, exist_ok=True)
        torch.save({"directions": directions, "anchor": anchor, "roles": roles, "model": saved.get("model"),
                    "train_reviews": sorted(train), "test_reviews": sorted(test)}, os.path.join(out_dir, "directions.pt"))
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\n=== {anchor} ===")
        for l in range(0, len(summary["shared_share"]), args.print_stride):
            line = (f"L{l:<2} |framing| {summary['norm']['framing'][l]:6.2f} |shared| {summary['norm']['shared'][l]:6.2f} "
                    f"shared_share {summary['shared_share'][l]:.2f} cos(framing, shared) {summary['cos_framing_vs']['shared'][l]:+.2f}")
            if aucs is not None:
                line += (f" | AUC framing {aucs['framing']['framing'][l]:.2f} shared {aucs['shared']['shared'][l]:.2f} "
                         + " ".join(f"{a}-{b} {aucs[f'{a}_minus_{b}'][f'{a}_minus_{b}'][l]:.2f}" for a, b in itertools.combinations(roles, 2)))
            print(line)
        plot(summary, f"{saved.get('model')} | anchor: {anchor}", os.path.join(out_dir, "diffmeans.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Difference-of-means framing and role directions in the question-only prompt's residual stream")
    parser.add_argument("--output_dir", required=True, help="outputs go to <output_dir>/<anchor>/.")
    parser.add_argument("--acts_path", default=None, help="activations (run_belief_probe.py format); collected and saved here if missing (default: <output_dir>/activations.pt).")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="HuggingFace model name (only used when collecting).")
    parser.add_argument("--input_path", default=None, help="dataset with the framed questions (extracted question set); needed only when collecting.")
    parser.add_argument("--num_reviews", default=-1, type=int, help="when collecting: seeded sample of reviews (-1: all).")
    parser.add_argument("--roles", nargs="+", default=ROLES, help="when collecting: roles for 'My <role> said, <claim>' (the bare claim is always included).")
    parser.add_argument("--anchors", nargs="+", default=None, help=f"activation keys to use (default: all saved, i.e. the anchors and {YES_NO}).")
    parser.add_argument("--test_fraction", default=0.2, type=float, help="fraction of reviews held out for the AUCs (0: fit on all, no AUCs).")
    parser.add_argument("--seed", default=SEED, type=int, help="seed for the train/test split.")
    parser.add_argument("--print_stride", default=4, type=int, help="print every k-th layer.")
    args = parser.parse_args()
    args.acts_path = args.acts_path or os.path.join(args.output_dir, "activations.pt")

    print("\nArguments:")
    for k, v in vars(args).items():
        print(f"{k:<14} {v}")
    print()
    main(args)
