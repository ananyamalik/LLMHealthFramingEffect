"""
Is the role's effect on the yes/no answer a Bayesian update? Fits a hierarchical model to the existing yes_no
pairs.jsonl (one fit per model and prompt) and asks whether "My <role> said," acts like a fixed piece of evidence.

    L0      target log-odds without the source (run 2): D for positive claims, -D for negative (as in
            plot_logodds_shift.py; the target is the answer the claim asks for)
    L1      target log-odds with the source (run 1)

    L1[i] = alpha[role, framing] + b[role] * L0[i] + u[review] + eps,    eps ~ N(0, sigma[role]^2), u ~ N(0, tau^2)

An agent that treats the source as evidence independent of the claim adds a fixed log-likelihood ratio to its prior
log-odds: b = 1 and alpha = log LR(role). b < 1 means the source pulls the answer toward 50/50 (the prior is
discounted, not added to); b > 1 means it pushes it further out. With b = 1, alpha is also the same with and without
article summaries (conditionally independent evidence), which the printed table compares across prompts.
alpha is the shift at L0 = 0 (p = 0.5); the shift at the observed priors is alpha + (b - 1) * L0.

Framing fit: the same model with b[role, framing], so the two framings can differ in slope as well as intercept. In
the question_only prompt every negative claim has L0 > 0 (the model already agrees that it "won't work"), so the
framings' priors barely overlap and alpha[role, negative] there is an extrapolation to L0 = 0. The framing figure
therefore compares b and the mean shift over each framing's own pairs, mean(alpha + (b - 1) * L0), without the
review effects (both framings come from the same reviews).

Priors: alpha, b ~ N(0, 10^2); sigma^2, tau^2 ~ InvGamma(1, 1). Blocked Gibbs sampler (all coefficients and review
effects jointly, then the variances), --chains chains; R-hat is the split-chain R-hat of Gelman et al. (2013).

Outputs (in --outputs_dir):
    bayes_role_update.png       per (model, prompt): L1 against L0 with the posterior mean lines and y = x, the
                                posterior of alpha per role and framing, and the posterior of b per role
    bayes_role_update.json      per (model, prompt): posterior mean, 95% interval and R-hat of every parameter,
                                and P(b < 1) per role
    bayes_role_update_framing.png   framing fit, per (model, prompt): per role, L1 against L0 by framing (lines over
                                each framing's observed L0); b per role and framing; mean shift per role and framing
    bayes_role_update_framing.json  framing fit: as above, plus the negative - positive differences of b and of the
                                mean shift

    python bayes_role_update.py [--roles doctor friend enemy] [--model_tags ...] [--prompts question_only articles]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FRAMINGS = ["positive", "negative"]   # same as run_role_patching.py (not imported: that pulls in torch/transformers)
PROMPTS = ["question_only", "articles"]
PROMPT_TITLES = {"question_only": "question only", "articles": "with article summaries"}
ROLE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # categorical slots in fixed order
FRAMING_MARKERS = {"positive": "o", "negative": "s"}
FRAMING_COLORS = {"positive": "#2a78d6", "negative": "#eb6834"}   # same as plot_logodds_shift.py
FRAMING_LABELS = {"positive": "positive claim (target: Yes)", "negative": "negative claim (target: No)"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PRIOR_SD, IG_A, IG_B = 10.0, 1.0, 1.0


def load_pairs(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    out = []
    for r in records:
        sign = 1.0 if r["framing"] == "positive" else -1.0
        out.append({"review": r["review_id"], "framing": r["framing"], "l0": sign * r["d_target"],
                    "l1": sign * r["d_source"]})
    return out


def design(data: dict, roles: list[str], split_b: bool) -> tuple:
    """Columns: alpha[role, framing] indicators, then b[role] (or b[role, framing]) * L0, then review indicators."""
    reviews = sorted(set(data["review"]))
    n = len(data["l1"])
    n_alpha = len(roles) * len(FRAMINGS)
    n_fixed = 2 * n_alpha if split_b else n_alpha + len(roles)
    A = np.zeros((n, n_fixed + len(reviews)))
    for i in range(n):
        k = roles.index(data["role"][i])
        j = k * len(FRAMINGS) + FRAMINGS.index(data["framing"][i])
        A[i, j] = 1.0
        A[i, n_alpha + (j if split_b else k)] = data["l0"][i]
        A[i, n_fixed + reviews.index(data["review"][i])] = 1.0
    role_idx = np.array([roles.index(r) for r in data["role"]])
    return A, n_fixed, reviews, role_idx


def gibbs(y: np.ndarray, A: np.ndarray, n_fixed: int, role_idx: np.ndarray, n_roles: int, n_iter: int,
          burn: int, rng: np.random.Generator) -> dict:
    n_rev = A.shape[1] - n_fixed
    sigma2, tau2 = np.ones(n_roles), 1.0
    draws = {"theta": [], "sigma": [], "tau": []}
    for it in range(n_iter):
        w = 1.0 / sigma2[role_idx]
        prior_prec = np.concatenate([np.full(n_fixed, 1 / PRIOR_SD ** 2), np.full(n_rev, 1 / tau2)])
        prec = (A * w[:, None]).T @ A + np.diag(prior_prec)
        chol = np.linalg.cholesky(prec)
        mean = np.linalg.solve(prec, A.T @ (w * y))
        theta = mean + np.linalg.solve(chol.T, rng.standard_normal(len(mean)))
        resid = y - A @ theta
        for k in range(n_roles):
            m = role_idx == k
            sigma2[k] = 1 / rng.gamma(IG_A + m.sum() / 2, 1 / (IG_B + resid[m] @ resid[m] / 2))
        u = theta[n_fixed:]
        tau2 = 1 / rng.gamma(IG_A + n_rev / 2, 1 / (IG_B + u @ u / 2))
        if it >= burn:
            draws["theta"].append(theta[:n_fixed].copy())
            draws["sigma"].append(np.sqrt(sigma2))
            draws["tau"].append([np.sqrt(tau2)])
    return {k: np.array(v) for k, v in draws.items()}


def split_rhat(chains: np.ndarray) -> np.ndarray:
    """chains: [chain, draw, param] -> split-chain R-hat per param."""
    half = chains.shape[1] // 2
    x = np.concatenate([chains[:, :half], chains[:, half:2 * half]], axis=0)
    n = x.shape[1]
    between = n * x.mean(axis=1).var(axis=0, ddof=1)
    within = x.var(axis=1, ddof=1).mean(axis=0)
    return np.sqrt(((n - 1) / n * within + between / n) / within)


def fit(data: dict, roles: list[str], args, split_b: bool = False) -> dict:
    A, n_fixed, reviews, role_idx = design(data, roles, split_b)
    y = np.array(data["l1"])
    runs = [gibbs(y, A, n_fixed, role_idx, len(roles), args.n_iter, args.burn, np.random.default_rng(args.seed + c))
            for c in range(args.chains)]
    chains = np.stack([np.concatenate([r["theta"], r["sigma"], r["tau"]], axis=1) for r in runs])
    b_keys = [f"{k},{f}" for k in roles for f in FRAMINGS] if split_b else roles
    names = ([f"alpha[{k},{f}]" for k in roles for f in FRAMINGS] + [f"b[{k}]" for k in b_keys]
             + [f"sigma[{k}]" for k in roles] + ["tau"])
    flat = chains.reshape(-1, chains.shape[2])
    rhat = split_rhat(chains)
    params = {name: {"mean": float(flat[:, j].mean()), "lo": float(np.percentile(flat[:, j], 2.5)),
                     "hi": float(np.percentile(flat[:, j], 97.5)), "rhat": float(rhat[j])}
              for j, name in enumerate(names)}
    return {"params": params, "draws": dict(zip(names, flat.T)), "n": len(y), "n_reviews": len(reviews),
            "p_b_below_1": {k: float(np.mean(flat[:, names.index(f"b[{k}]")] < 1)) for k in b_keys}}


def summarize(x: np.ndarray) -> dict:
    return {"mean": float(x.mean()), "lo": float(np.percentile(x, 2.5)), "hi": float(np.percentile(x, 97.5))}


def framing_contrasts(data: dict, res: dict, roles: list[str]) -> dict:
    """Per role: mean shift per framing over its own pairs, and the negative - positive differences (framing fit)."""
    l0, role, framing = np.array(data["l0"]), np.array(data["role"]), np.array(data["framing"])
    d = res["draws"]
    out = {}
    for k in roles:
        shift = {}
        for f in FRAMINGS:
            mean_l0 = l0[(role == k) & (framing == f)].mean()
            shift[f] = d[f"alpha[{k},{f}]"] + (d[f"b[{k},{f}]"] - 1) * mean_l0
        b_diff = d[f"b[{k},negative]"] - d[f"b[{k},positive]"]
        shift_diff = shift["negative"] - shift["positive"]
        out[k] = {"shift": {f: summarize(shift[f]) for f in FRAMINGS}, "shift_diff": summarize(shift_diff),
                  "p_shift_diff_below_0": float(np.mean(shift_diff < 0)), "b_diff": summarize(b_diff),
                  "p_b_diff_below_0": float(np.mean(b_diff < 0))}
    return out


def style(ax) -> None:
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def interval(ax, y: float, p: dict, color: str, marker: str) -> None:
    ax.plot([p["lo"], p["hi"]], [y, y], color=color, linewidth=2.2, solid_capstyle="round", zorder=3)
    ax.scatter([p["mean"]], [y], s=48, color=color, marker=marker, edgecolor="white", linewidth=1.2, zorder=4)


def plot(args, panels: list) -> None:
    n_cols = len(panels)
    fig, axes = plt.subplots(3, n_cols, figsize=(5.2 * n_cols, 11.0), squeeze=False, facecolor="white",
                             gridspec_kw={"height_ratios": [3.2, 1.6, 1.1], "hspace": 0.45, "wspace": 0.3})
    for c, (model_tag, prompt, data, res) in enumerate(panels):
        p = res["params"]
        ax = axes[0][c]
        l0, l1 = np.array(data["l0"]), np.array(data["l1"])
        lim = (min(l0.min(), l1.min()) - 0.5, max(l0.max(), l1.max()) + 0.5)
        ax.plot(lim, lim, color=MUTED, linestyle="--", linewidth=1, label="y = x (no shift)")
        grid = np.linspace(l0.min(), l0.max(), 50)
        for k, role in enumerate(args.roles):
            color = ROLE_COLORS[k]
            for framing in FRAMINGS:
                m = (np.array(data["role"]) == role) & (np.array(data["framing"]) == framing)
                ax.scatter(l0[m], l1[m], s=26, color=color, marker=FRAMING_MARKERS[framing], edgecolor="white",
                           linewidth=0.8, alpha=0.75, zorder=3)
            # mean over framings of the alpha intercepts, for one line per role
            a = np.mean([p[f"alpha[{role},{f}]"]["mean"] for f in FRAMINGS])
            ax.plot(grid, a + p[f"b[{role}]"]["mean"] * grid, color=color, linewidth=2, zorder=4,
                    label=f"My {role} said,")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("L0: target log-odds without source", color=TEXT, fontsize=10)
        if c == 0:
            ax.set_ylabel("L1: target log-odds with source", color=TEXT, fontsize=10)
        ax.set_title(f"{model_tag} · {PROMPT_TITLES[prompt]} (n={res['n']})", color=TEXT, fontsize=10.5, loc="left")
        style(ax)

        ax = axes[1][c]
        ticks, labels = [], []
        for k, role in enumerate(args.roles):
            for j, framing in enumerate(FRAMINGS):
                y = -(k * 2.6 + j)
                interval(ax, y, p[f"alpha[{role},{framing}]"], ROLE_COLORS[k], FRAMING_MARKERS[framing])
                ticks.append(y)
                labels.append(f"{role} · {framing}")
        ax.axvline(0, color=MUTED, linewidth=1)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels if c == 0 else [], color=TEXT, fontsize=9)
        ax.set_xlabel("α: shift at p = 0.5 (nats)", color=TEXT, fontsize=10)
        style(ax)
        ax.grid(False, axis="y")

        ax = axes[2][c]
        for k, role in enumerate(args.roles):
            interval(ax, -k, p[f"b[{role}]"], ROLE_COLORS[k], "D")
            ax.text(1.0, -k + 0.32, f"P(b<1)={res['p_b_below_1'][role]:.2f}", transform=ax.get_yaxis_transform(),
                    ha="right", fontsize=8, color=MUTED)
        ax.axvline(1, color=TEXT, linestyle="--", linewidth=1.2)
        ax.set_yticks(range(0, -len(args.roles), -1))
        ax.set_yticklabels(args.roles if c == 0 else [], color=TEXT, fontsize=9)
        ax.set_ylim(-len(args.roles) + 0.4, 0.7)
        ax.set_xlabel("b: weight on prior (Bayes: 1)", color=TEXT, fontsize=10)
        style(ax)
        ax.grid(False, axis="y")

    handles, labels = axes[0][0].get_legend_handles_labels()
    for framing in FRAMINGS:
        handles.append(plt.Line2D([], [], marker=FRAMING_MARKERS[framing], color=MUTED, linestyle="", markersize=7))
        labels.append(f"{framing} claim")
    fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Yes/No answer: L1 = α[role, framing] + b[role]·L0 + u[review] (posterior means, 95% intervals)",
                 color=TEXT, fontsize=12.5, y=0.985)
    out_path = os.path.join(args.outputs_dir, "bayes_role_update.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_framing(args, panels: list) -> None:
    """Framing fit: per role, L1 against L0 by framing; b and the mean shift per role and framing."""
    n_cols, n_roles = len(panels), len(args.roles)
    fig, axes = plt.subplots(n_roles + 2, n_cols, figsize=(5.2 * n_cols, 3.3 * n_roles + 4.2), squeeze=False,
                             facecolor="white", gridspec_kw={"height_ratios": [3] * n_roles + [1.3, 1.3],
                                                             "hspace": 0.55, "wspace": 0.3})
    offsets = {"positive": 0.17, "negative": -0.17}
    for c, (model_tag, prompt, data, res, contrasts) in enumerate(panels):
        p = res["params"]
        l0, l1 = np.array(data["l0"]), np.array(data["l1"])
        role_arr, framing_arr = np.array(data["role"]), np.array(data["framing"])
        lim = (min(l0.min(), l1.min()) - 0.5, max(l0.max(), l1.max()) + 0.5)
        for k, role in enumerate(args.roles):
            ax = axes[k][c]
            ax.plot(lim, lim, color=MUTED, linestyle="--", linewidth=1, label="y = x (no shift)")
            for framing in FRAMINGS:
                m = (role_arr == role) & (framing_arr == framing)
                color = FRAMING_COLORS[framing]
                ax.scatter(l0[m], l1[m], s=30, color=color, marker=FRAMING_MARKERS[framing], edgecolor="white",
                           linewidth=0.9, alpha=0.8, zorder=3, label=FRAMING_LABELS[framing])
                grid = np.linspace(l0[m].min(), l0[m].max(), 50)   # only over this framing's own priors
                ax.plot(grid, p[f"alpha[{role},{framing}]"]["mean"] + p[f"b[{role},{framing}]"]["mean"] * grid,
                        color=color, linewidth=2.2, zorder=4)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.text(0.03, 0.97, f"My {role} said,", transform=ax.transAxes, va="top", fontsize=10, color=TEXT,
                    fontweight="bold")
            if k == 0:
                ax.set_title(f"{model_tag} · {PROMPT_TITLES[prompt]}", color=TEXT, fontsize=10.5, loc="left")
            if k == n_roles - 1:
                ax.set_xlabel("L0: target log-odds without source", color=TEXT, fontsize=10)
            if c == 0:
                ax.set_ylabel("L1: with source", color=TEXT, fontsize=10)
            style(ax)

        for row, (key, xlabel, ref) in enumerate([("b", "b: weight on prior (Bayes: 1)", 1.0),
                                                  ("shift", "mean shift L1 − L0 over the pairs (nats)", 0.0)]):
            ax = axes[n_roles + row][c]
            for k, role in enumerate(args.roles):
                for framing in FRAMINGS:
                    q = p[f"b[{role},{framing}]"] if key == "b" else contrasts[role]["shift"][framing]
                    interval(ax, -k + offsets[framing], q, FRAMING_COLORS[framing], FRAMING_MARKERS[framing])
                prob = (contrasts[role]["p_b_diff_below_0"] if key == "b"
                        else contrasts[role]["p_shift_diff_below_0"])
                ax.text(1.02, -k, f"P(neg<pos) {prob:.2f}", transform=ax.get_yaxis_transform(), ha="left",
                        va="center", fontsize=8, color=MUTED)
            ax.axvline(ref, color=TEXT, linestyle="--", linewidth=1.2)
            ax.set_yticks(range(0, -n_roles, -1))
            ax.set_yticklabels(args.roles if c == 0 else [], color=TEXT, fontsize=9)
            ax.set_ylim(-n_roles + 0.5, 0.5)
            ax.set_xlabel(xlabel, color=TEXT, fontsize=10)
            style(ax)
            ax.grid(False, axis="y")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("Yes/No answer by framing: L1 = α[role, framing] + b[role, framing]·L0 + u[review]",
                 color=TEXT, fontsize=12.5, y=0.99)
    out_path = os.path.join(args.outputs_dir, "bayes_role_update_framing.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def print_framing_table(args, framing_results: dict) -> None:
    fmt = lambda q: f" {q['mean']:+.2f} [{q['lo']:+.2f},{q['hi']:+.2f}]"
    print("\nFraming fit (b[role, framing]); differences are negative - positive")
    print(f"{'model':<20}{'prompt':<15}{'role':<8}{'b +':>19}{'b -':>19}{'b diff':>19}{'shift +':>19}"
          f"{'shift -':>19}{'shift diff':>19}{'P(<0)':>7}")
    for (model_tag, prompt), (res, contrasts) in framing_results.items():
        p = res["params"]
        for role in args.roles:
            con = contrasts[role]
            print(f"{model_tag:<20}{prompt:<15}{role:<8}{fmt(p[f'b[{role},positive]']):>19}"
                  f"{fmt(p[f'b[{role},negative]']):>19}{fmt(con['b_diff']):>19}{fmt(con['shift']['positive']):>19}"
                  f"{fmt(con['shift']['negative']):>19}{fmt(con['shift_diff']):>19}"
                  f"{con['p_shift_diff_below_0']:>7.2f}")
        print(f"{'':<35}max R-hat {max(q['rhat'] for q in p.values()):.3f}")


def print_table(args, results: dict) -> None:
    print(f"\n{'model':<20}{'prompt':<15}{'role':<8}{'alpha +':>19}{'alpha -':>19}{'b':>19}{'P(b<1)':>8}"
          f"{'max R-hat':>10}")
    fmt = lambda q: f" {q['mean']:+.2f} [{q['lo']:+.2f},{q['hi']:+.2f}]"
    for (model_tag, prompt), res in results.items():
        p = res["params"]
        max_rhat = max(q["rhat"] for q in p.values())
        for role in args.roles:
            print(f"{model_tag:<20}{prompt:<15}{role:<8}{fmt(p[f'alpha[{role},positive]']):>19}"
                  f"{fmt(p[f'alpha[{role},negative]']):>19}{fmt(p[f'b[{role}]']):>19}"
                  f"{res['p_b_below_1'][role]:>8.2f}{max_rhat:>10.3f}")
        print(f"{'':<35}tau {fmt(p['tau'])}   " + "  ".join(f"sigma[{r}] {fmt(p[f'sigma[{r}]'])}" for r in args.roles))

    # Under b = 1 the source's log LR should not depend on whether the articles are in the prompt.
    print("\nalpha (articles) - alpha (question only), independent fits; 95% interval")
    for model_tag in args.model_tags:
        if not all((model_tag, pr) in results for pr in PROMPTS):
            continue
        for role in args.roles:
            cells = []
            for framing in FRAMINGS:
                name = f"alpha[{role},{framing}]"
                diff = results[(model_tag, "articles")]["draws"][name] - results[(model_tag, "question_only")]["draws"][name]
                cells.append(f"{framing} {diff.mean():+.2f} [{np.percentile(diff, 2.5):+.2f},"
                             f"{np.percentile(diff, 97.5):+.2f}]")
            print(f"  {model_tag:<20}{role:<8}" + "   ".join(cells))


def main(args):
    panels, results, framing_panels, framing_results = [], {}, [], {}
    for model_tag in args.model_tags:
        for prompt in args.prompts:
            run_dir = "role_patching" if prompt == "articles" else f"role_patching_{prompt}"
            data = {"review": [], "framing": [], "role": [], "l0": [], "l1": []}
            for role in args.roles:
                path = os.path.join(args.outputs_dir, model_tag, run_dir, "yes_no", role, "pairs.jsonl")
                if not os.path.exists(path):
                    print(f"Missing {path}")
                    continue
                for r in load_pairs(path):
                    for key in ["review", "framing", "l0", "l1"]:
                        data[key].append(r[key])
                    data["role"].append(role)
            if not all(role in data["role"] for role in args.roles):
                print(f"Skipping {model_tag} {prompt}: not every role has pairs")
                continue
            res = fit(data, args.roles, args)
            results[(model_tag, prompt)] = res
            panels.append((model_tag, prompt, data, res))
            res_f = fit(data, args.roles, args, split_b=True)
            contrasts = framing_contrasts(data, res_f, args.roles)
            framing_results[(model_tag, prompt)] = (res_f, contrasts)
            framing_panels.append((model_tag, prompt, data, res_f, contrasts))
    if not panels:
        return

    plot(args, panels)
    plot_framing(args, framing_panels)
    print_table(args, results)
    print_framing_table(args, framing_results)
    out = {f"{m}/{p}": {"n": r["n"], "n_reviews": r["n_reviews"], "params": r["params"],
                        "p_b_below_1": r["p_b_below_1"]} for (m, p), r in results.items()}
    out_path = os.path.join(args.outputs_dir, "bayes_role_update.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")
    out = {f"{m}/{p}": {"n": r["n"], "n_reviews": r["n_reviews"], "params": r["params"],
                        "p_b_below_1": r["p_b_below_1"], "framing_contrasts": con}
           for (m, p), (r, con) in framing_results.items()}
    out_path = os.path.join(args.outputs_dir, "bayes_role_update_framing.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hierarchical Bayesian model of the role's yes/no log-odds update")
    parser.add_argument("--roles", nargs="+", default=["doctor", "friend", "enemy"])
    parser.add_argument("--model_tags", nargs="+", default=["llama3-8B-instruct", "olmo2-7B-instruct"])
    parser.add_argument("--prompts", nargs="+", default=PROMPTS, choices=PROMPTS)
    parser.add_argument("--outputs_dir", default="./outputs")
    parser.add_argument("--chains", default=4, type=int)
    parser.add_argument("--n_iter", default=3000, type=int, help="iterations per chain, including burn-in.")
    parser.add_argument("--burn", default=1000, type=int)
    parser.add_argument("--seed", default=0, type=int)
    main(parser.parse_args())
