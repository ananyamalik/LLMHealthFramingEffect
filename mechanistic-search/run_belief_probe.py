"""
Belief probe: before it writes a full answer, does the residual stream already say whether the model thinks the
treatment works?

Data: every review's claim in both framings (as in run_role_patching.py), with no role ("<claim>") and with each of
--roles ("My <role> said, <claim>"), in the question-only prompts:
    full    prompts/question_only.jinja2         "Question: <question>"                 (the full-answer setup)
    yes_no  prompts/question_only_yes_no.jinja2  ... + 'Does the treatment in the question work? Answer with only
                                                 "Yes" or "No".'
The yes/no prompt gives the model's stated belief

    D = log p(Yes) - log p(No)      (> 0: the model says the treatment works, whatever the claim's framing)

Every claim is "<intervention> will|won't work for <condition>. What do you think?", so the residual stream (decoder
layer output) of the full prompt is cached at every layer at these anchors (--anchors), in reading order:
    intervention   last token of <intervention>: positive and negative rows are identical up to here, so this is a
                   prior about the treatment that cannot know the framing (with a role: after reading the role)
    verdict        "work": the claim's direction is known, the condition is not
    claim_end      the "." of the claim
    question_end   the "?" of "What do you think?"
    last           the last prompt token (generation prompt), where the full answer starts
and at the last token of the yes/no prompt (where D is read out). Comparing a bare row with the same review's role row
at an anchor gives the belief before (prior) and after (posterior) the role, at that point of the claim.

Probes (per layer, ridge regression of D on the standardized residual, in kernel form). Predictions are out of fold:
--folds folds grouped by review, so all rows of a review (both framings, every role) are held out together, with the
penalty chosen per fold by an inner grouped CV on the training reviews:
    <anchor>          trained and tested at that anchor of the full prompt (all roles): where is the belief?
    last_to_<anchor>  the `last` probe applied at an earlier anchor: is the answer's belief direction already there?
    last_bare         trained at `last` without a role, tested on every role (does the role move the belief?)
    yes_no            trained and tested on the yes/no prompt (reference: there D is read out from this position)
    yes_no_to_last    trained on the yes/no prompt, tested at `last` of the full prompt (is it the same direction?)

Framing confound: framing is trivially decodable and the stated belief depends on it (in the question-only prompt,
every negative claim gets "No"), so a probe that only reads framing would score a high overall r. The headline metric
is r_within: the correlation of predicted and actual D after subtracting the mean of each (framing, role) cell from
both, i.e. how well the probe ranks reviews within a framing and role. Also reported: r overall and per framing, AUC
of D > 0 within positive claims (negative ones are all "No"), and per role, the correlation of the role's shift
(role row - bare row of the same review and framing) predicted vs actual, within framing.

Null: --permutations copies of D are shuffled across reviews within each framing (a review keeps its rows together,
so the role structure stays), and every probe is refit on them. p_max is the fraction of permutations whose best layer
reaches the observed best r_within (corrects for picking the best of all layers).

Outputs (in --output_dir):
    rows.jsonl          one line per (review, framing, role): question, D, top token, p(Yes) + p(No)
    activations.pt      residuals [rows, layers, d_model] (float16) per anchor and of the yes/no prompt, and the row
                        metadata (--save_activations; ~0.8 GB per anchor for all reviews and 3 roles); --analyze_only
                        re-runs the probes from it
    predictions.pt      out-of-fold predictions [rows, layers, 1 + permutations] per probe
    summary.json        per probe and layer: metrics (all rows and per role); per probe, the best layer by r_within
                        with the permutation null
    belief_probe.png    r_within by layer per anchor; best-layer r_within along the claim (own probe, last-token
                        probe, null); per role, r of the role shift along the claim; predicted vs actual D at `last`
"""
import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from patching_utils import (
    DEFAULT_MODEL_NAME, SEED, set_global_seed, load_model_and_tokenizer, get_decoder_layers, load_dataset,
    sample_reviews, cache_residuals,
)
from run_role_patching import FRAMINGS, ROLES, YES_STRINGS, NO_STRINGS, role_prefix, get_claims, encode_prompt, single_token_ids

NO_ROLE = "none"
ANCHORS = ["intervention", "verdict", "claim_end", "question_end", "last"]
YES_NO = "yes_no"   # activation key of the yes/no prompt's last token (the full prompt's are keyed by anchor)
CLAIM_PATTERN = re.compile(r"(.+?) (will|won't) work for (.+)\. What do you think\?")
FRAMING_COLORS = {"positive": "#2a78d6", "negative": "#eb6834"}   # same as plot_logodds_shift.py
ROLE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
ANCHOR_COLORS = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]   # light -> dark in reading order
MUTED = "#52514e"


def build_probes(anchors: list[str]) -> dict[str, tuple[str, str, bool]]:
    """probe -> (training activations, test activations, train on bare rows only)."""
    probes = {a: (a, a, False) for a in anchors}
    probes |= {f"last_to_{a}": ("last", a, False) for a in anchors if a != "last"}
    probes |= {"last_bare": ("last", "last", True), YES_NO: (YES_NO, YES_NO, False), "yes_no_to_last": (YES_NO, "last", False)}
    return probes


def anchor_chars(claim: str) -> dict[str, int]:
    """Character index (in the claim) of the last character of each anchor inside the question."""
    m = CLAIM_PATTERN.fullmatch(claim)
    if not m:
        raise ValueError(f"Claim does not match '<intervention> will|won't work for <condition>. ...': {claim}")
    return {"intervention": m.end(1) - 1, "verdict": m.end(2) + len(" work") - 1, "claim_end": m.end(3),
            "question_end": len(claim) - 1}


# Activations
@torch.no_grad()
def collect(model, tokenizer, dataset: list[dict], roles: list[str], anchors: list[str]) -> tuple[list[dict], dict[str, torch.Tensor]]:
    """
    Rows (one per review, framing, role) and residuals [rows, layers, d_model]: the full prompt at every anchor, and
    the yes/no prompt's last token.
    """
    num_layers = len(get_decoder_layers(model))
    yes_ids, no_ids = single_token_ids(tokenizer, YES_STRINGS), single_token_ids(tokenizer, NO_STRINGS)
    rows, acts = [], {k: [] for k in anchors + [YES_NO]}
    for example in tqdm(dataset, desc="reviews"):
        for framing, claim in get_claims(example).items():
            chars = anchor_chars(claim)
            for role in [NO_ROLE] + roles:
                question = claim if role == NO_ROLE else f"{role_prefix(role)} {claim}"
                full_ids, encoding, question_start = encode_prompt(tokenizer, "full", question, [], model.device, "question_only")
                claim_start = question_start + len(question) - len(claim)
                positions = {a: encoding.char_to_token(claim_start + c) for a, c in chars.items()} | {"last": full_ids.shape[1] - 1}
                positions = {a: positions[a] for a in anchors}

                with cache_residuals(model, list(positions.values())) as cache:
                    model.model(input_ids=full_ids, use_cache=False)
                for i, a in enumerate(anchors):
                    acts[a].append(torch.stack([cache[l][i].float().cpu() for l in range(num_layers)]).half())

                # D as in yes_no_logit_diff, and p(Yes) + p(No): how much of the answer the readout covers
                yes_no_ids = encode_prompt(tokenizer, "yes_no", question, [], model.device, "question_only")[0]
                with cache_residuals(model, [yes_no_ids.shape[1] - 1]) as cache:
                    hidden = model.model(input_ids=yes_no_ids, use_cache=False).last_hidden_state[:, -1]
                acts[YES_NO].append(torch.stack([cache[l][0].float().cpu() for l in range(num_layers)]).half())
                log_probs = torch.log_softmax(model.lm_head(hidden).float(), dim=-1)[0]
                d = (torch.logsumexp(log_probs[yes_ids], 0) - torch.logsumexp(log_probs[no_ids], 0)).item()
                rows.append({
                    "review_id": example["ReviewID"], "framing": framing, "role": role, "question": question,
                    "d": d, "top_token": tokenizer.decode([int(log_probs.argmax())]),
                    "p_yes_no": log_probs[yes_ids + no_ids].exp().sum().item(),
                    "positions": positions,
                    "anchor_tokens": {a: tokenizer.decode([full_ids[0, p].item()]) for a, p in positions.items()},
                })
    return rows, {k: torch.stack(v) for k, v in acts.items()}


# Ridge probes
def grouped_folds(groups: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    """k folds of row indices, with every group (review) in exactly one fold."""
    unique = np.random.default_rng(seed).permutation(np.unique(groups))
    return [np.where(np.isin(groups, part))[0] for part in np.array_split(unique, k)]


def ix(indices: np.ndarray) -> torch.Tensor:
    """numpy row indices as a tensor index (some torch versions do not index with numpy integer arrays)."""
    return torch.tensor(indices.tolist(), dtype=torch.long)


def standardize(x_fit: torch.Tensor, *xs: torch.Tensor) -> list[torch.Tensor]:
    mean, std = x_fit.mean(0), x_fit.std(0) + 1e-6
    return [(x - mean) / std for x in (x_fit,) + xs]


def ridge_predict(x_tr: torch.Tensor, y_tr: torch.Tensor, x_te: torch.Tensor, alphas: list[float]) -> torch.Tensor:
    """
    Kernel ridge (linear kernel, intercept by centering y) for every alpha and every column of y [n_tr, c]:
    predictions [len(alphas), n_te, c]. Alphas are scaled by d, so they are comparable across layers and models.
    """
    d = x_tr.shape[1]
    y_mean = y_tr.mean(0, keepdim=True)
    kernel, k_te = x_tr @ x_tr.T, x_te @ x_tr.T
    eye = torch.eye(len(x_tr), dtype=x_tr.dtype, device=x_tr.device)
    return torch.stack([k_te @ torch.cholesky_solve(y_tr - y_mean, torch.linalg.cholesky(kernel + a * d * eye)) + y_mean
                        for a in alphas])


def fit_fold(x_tr, y_tr, groups_tr, x_tes: list[torch.Tensor], alphas, inner_folds, seed) -> list[torch.Tensor]:
    """
    Predictions [n_te, c] for every test set in x_tes (one fit), with alpha picked per column of y by inner grouped CV
    on the training rows.
    """
    sq_err = torch.zeros(len(alphas), y_tr.shape[1], dtype=y_tr.dtype, device=y_tr.device)
    for val in grouped_folds(groups_tr, inner_folds, seed):
        fit = np.setdiff1d(np.arange(len(groups_tr)), val)
        xf, xv = standardize(x_tr[ix(fit)], x_tr[ix(val)])
        sq_err += ((ridge_predict(xf, y_tr[ix(fit)], xv, alphas) - y_tr[ix(val)]) ** 2).sum(1)
    best = sq_err.argmin(0)                                  # [c]
    xf, *xts = standardize(x_tr, *x_tes)
    preds = ridge_predict(xf, y_tr, torch.cat(xts), alphas)  # [alphas, sum n_te, c]
    preds = preds[best, :, torch.arange(y_tr.shape[1])].T
    return list(preds.split([len(x) for x in x_tes]))


def permuted_targets(rows: list[dict], n: int, seed: int) -> np.ndarray:
    """[rows, n] copies of D, shuffled across reviews within each framing (a review's roles stay together)."""
    rng = np.random.default_rng(seed)
    reviews = sorted({r["review_id"] for r in rows})
    lookup = {(r["review_id"], r["framing"], r["role"]): r["d"] for r in rows}
    out = np.zeros((len(rows), n))
    for j in range(n):
        perm = {f: dict(zip(reviews, rng.permutation(reviews))) for f in FRAMINGS}
        for i, r in enumerate(rows):
            # a review missing this role/framing after shuffling keeps its own D (does not happen with complete data)
            out[i, j] = lookup.get((perm[r["framing"]][r["review_id"]], r["framing"], r["role"]), r["d"])
    return out


def run_probes(rows: list[dict], acts: dict[str, torch.Tensor], probes: dict, args, device) -> dict[str, torch.Tensor]:
    """Out-of-fold predictions per probe [rows, layers, 1 + permutations] (column 0: real D)."""
    groups = np.array([r["review_id"] for r in rows])
    bare = np.array([r["role"] == NO_ROLE for r in rows])
    y = np.concatenate([np.array([[r["d"]] for r in rows]), permuted_targets(rows, args.permutations, args.seed)], 1)
    y = torch.tensor(y.tolist(), dtype=torch.float64, device=device)
    folds = grouped_folds(groups, args.folds, args.seed)
    num_layers = acts[YES_NO].shape[1]
    fits = {}   # (training activations, bare only) -> probes that share the fit
    for name, (train_key, _, bare_only) in probes.items():
        fits.setdefault((train_key, bare_only), []).append(name)

    preds = {name: torch.zeros(len(rows), num_layers, y.shape[1], dtype=torch.float64) for name in probes}
    for layer in tqdm(range(num_layers), desc="probes"):
        x = {k: v[:, layer].to(device, torch.float64) for k, v in acts.items()}
        for test in folds:
            train = np.setdiff1d(np.arange(len(rows)), test)
            for (train_key, bare_only), names in fits.items():
                tr = train[bare[train]] if bare_only else train
                x_tes = []
                for name in names:
                    test_key = probes[name][1]
                    x_te = x[test_key][ix(test)]
                    if test_key != train_key:
                        # fit_fold standardizes x_te with the training activations' statistics; map the test
                        # activations so that this gives them their own training-fold standardization instead
                        _, x_te = standardize(x[test_key][ix(tr)], x_te)
                        x_te = x_te * (x[train_key][ix(tr)].std(0) + 1e-6) + x[train_key][ix(tr)].mean(0)
                    x_tes.append(x_te)
                out = fit_fold(x[train_key][ix(tr)], y[ix(tr)], groups[tr], x_tes, args.alphas, args.inner_folds, args.seed)
                for name, p in zip(names, out):
                    preds[name][ix(test), layer] = p.cpu()
    return preds


# Metrics
def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def auc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def within_r(pred: np.ndarray, d: np.ndarray, cells: np.ndarray) -> float:
    """Correlation after removing the mean of each cell from both."""
    pc, dc = pred.copy(), d.copy()
    for c in np.unique(cells):
        m = cells == c
        pc[m] -= pc[m].mean()
        dc[m] -= dc[m].mean()
    return pearson(pc, dc)


def metrics(rows: list[dict], pred: np.ndarray, d: np.ndarray, roles: list[str]) -> dict:
    framing = np.array([r["framing"] for r in rows])
    role = np.array([r["role"] for r in rows])
    cells = np.array([f"{f}|{r}" for f, r in zip(framing, role)])

    def block(m):
        pos = m & (framing == "positive")
        return {
            "n": int(m.sum()),
            "r_within": within_r(pred[m], d[m], cells[m]),
            "r": pearson(pred[m], d[m]),
            **{f"r_{f}": pearson(pred[m & (framing == f)], d[m & (framing == f)]) for f in FRAMINGS},
            "auc_positive": auc(pred[pos], d[pos] > 0),
        }

    out = {"all": block(np.ones(len(rows), bool)), "roles": {r: block(role == r) for r in [NO_ROLE] + roles}}
    # role shift: role row - bare row of the same review and framing
    index = {(r["review_id"], r["framing"], r["role"]): i for i, r in enumerate(rows)}
    for r in roles:
        pairs = [(index[k[:2] + (r,)], i) for k, i in index.items() if k[2] == NO_ROLE and k[:2] + (r,) in index]
        if pairs:
            ri, bi = np.array(pairs).T
            # within framing, so a role that shifts one framing more than the other does not count as tracking
            out["roles"][r]["shift_r"] = within_r(pred[ri] - pred[bi], d[ri] - d[bi], framing[bi])
            out["roles"][r]["shift_mean_actual"] = float((d[ri] - d[bi]).mean())
            out["roles"][r]["shift_mean_pred"] = float((pred[ri] - pred[bi]).mean())
    return out


def summarize(rows: list[dict], preds: dict[str, torch.Tensor], probes: dict, roles: list[str], args) -> dict:
    d = np.array([r["d"] for r in rows])
    framing = np.array([r["framing"] for r in rows])
    cells = np.array([f"{r['framing']}|{r['role']}" for r in rows])
    summary = {
        "n_rows": len(rows), "n_reviews": len({r["review_id"] for r in rows}), "roles": roles,
        "anchors": args.anchors, "permutations": args.permutations,
        "framing_only_r": pearson(np.array([d[framing == f].mean() for f in framing]), d),
        "yes_rate": {f: {r: float((d[(framing == f) & (np.array([x["role"] for x in rows]) == r)] > 0).mean())
                         for r in [NO_ROLE] + roles} for f in FRAMINGS},
        "mean_p_yes_no": float(np.mean([r["p_yes_no"] for r in rows])),
        "probes": {},
    }
    y_perm = permuted_targets(rows, args.permutations, args.seed)   # the same shuffles run_probes fit
    for name, p in preds.items():
        p = np.array(p.tolist())
        per_layer = [{"layer": l, **metrics(rows, p[:, l, 0], d, roles)} for l in range(p.shape[1])]
        best = max(per_layer, key=lambda m: m["all"]["r_within"])
        null_max = [max(within_r(p[:, l, j + 1], y_perm[:, j], cells) for l in range(p.shape[1]))
                    for j in range(args.permutations)]
        summary["probes"][name] = {
            "train": probes[name][0], "test": probes[name][1], "train_bare_only": probes[name][2],
            "per_layer": per_layer, "best_layer": best["layer"], "best": best,
            "null_max_r_within": {"mean": float(np.mean(null_max)) if null_max else None,
                                  "q95": float(np.quantile(null_max, 0.95)) if null_max else None,
                                  "p_max": float((1 + sum(v >= best["all"]["r_within"] for v in null_max)) / (1 + len(null_max)))
                                  if null_max else None},
        }
    return summary


def plot(rows: list[dict], preds: dict[str, torch.Tensor], summary: dict, path: str) -> None:
    anchors, probes = summary["anchors"], summary["probes"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    layers = [m["layer"] for m in probes[anchors[-1]]["per_layer"]]
    best = {name: s["best"] for name, s in probes.items()}

    ax = axes[0, 0]
    for i, a in enumerate(anchors):
        ax.plot(layers, [m["all"]["r_within"] for m in probes[a]["per_layer"]], marker="o", ms=3,
                color=ANCHOR_COLORS[i % len(ANCHOR_COLORS)], label=a)
    ax.plot(layers, [m["all"]["r_within"] for m in probes[YES_NO]["per_layer"]], ls="--", color=MUTED,
            label="yes/no prompt (reference)")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set(title="Where is the belief? Probe per anchor of the full prompt", xlabel="layer",
           ylabel="r_within(predicted D, D)\n(framing x role means removed)")
    ax.legend(title="anchor", fontsize=8)

    ax = axes[0, 1]
    x = np.arange(len(anchors))
    ax.plot(x, [best[a]["all"]["r_within"] for a in anchors], marker="o", color=ANCHOR_COLORS[-1], label="probe trained at the anchor")
    transfer = [best[f"last_to_{a}"]["all"]["r_within"] if a != "last" else best["last"]["all"]["r_within"] for a in anchors]
    ax.plot(x, transfer, marker="s", color="#eb6834", label="last-token probe applied at the anchor")
    ax.plot(x, [probes[a]["null_max_r_within"]["q95"] or np.nan for a in anchors], ls=":", color=MUTED,
            label="permutation null (95%, best layer)")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x, anchors)
    ax.set(title="Belief along the claim (each probe at its best layer)", ylabel="r_within")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for i, role in enumerate(summary["roles"]):
        ax.plot(x, [best[a]["roles"][role].get("shift_r", np.nan) for a in anchors], marker="o",
                color=ROLE_COLORS[i % len(ROLE_COLORS)], label=role)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x, anchors)
    ax.set(title="Prior -> posterior: does the probe track the role's shift?\n(role row - bare row, same review and framing)",
           ylabel="r(predicted shift, actual shift of D)")
    ax.legend(title="role", fontsize=8)

    ax = axes[1, 1]
    layer = probes["last"]["best_layer"]
    p, d = np.array(preds["last"][:, layer, 0].tolist()), np.array([r["d"] for r in rows])
    for framing in FRAMINGS:
        m = np.array([r["framing"] == framing for r in rows])
        ax.scatter(p[m], d[m], s=6, alpha=0.4, color=FRAMING_COLORS[framing], label=framing)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set(title=f"last-token probe, layer {layer} (out of fold)", xlabel="predicted D from the full prompt",
           ylabel="D = log p(Yes) - log p(No) (yes/no prompt)")
    ax.legend(title="claim framing", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


def main(args):
    set_global_seed(SEED)
    os.makedirs(args.output_dir, exist_ok=True)
    acts_path = os.path.join(args.output_dir, "activations.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.analyze_only:
        saved = torch.load(acts_path)
        rows, acts, args.roles, args.anchors = saved["rows"], saved["acts"], saved["roles"], saved["anchors"]
        print(f"Loaded {len(rows)} rows from {acts_path}")
    else:
        dataset = sample_reviews(load_dataset(args.input_path), args.num_reviews, seed=args.seed)
        print(f"Number of reviews: {len(dataset)}")
        model, tokenizer = load_model_and_tokenizer(args.model)
        rows, acts = collect(model, tokenizer, dataset, args.roles, args.anchors)
        del model
        torch.cuda.empty_cache()
        with open(os.path.join(args.output_dir, "rows.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        if args.save_activations:
            torch.save({"rows": rows, "acts": acts, "roles": args.roles, "anchors": args.anchors, "model": args.model}, acts_path)
            print(f"Saved {acts_path}")

    probes = build_probes(args.anchors)
    preds = run_probes(rows, acts, probes, args, device)
    torch.save({"predictions": preds, "rows": rows}, os.path.join(args.output_dir, "predictions.pt"))
    summary = summarize(rows, preds, probes, args.roles, args) | {"model": args.model, "folds": args.folds, "alphas": args.alphas}
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{summary['n_rows']} rows, {summary['n_reviews']} reviews | mean p(Yes) + p(No) {summary['mean_p_yes_no']:.3f}")
    print("Yes rate (D > 0): " + " | ".join(f"{f}: " + ", ".join(f"{r} {v:.2f}" for r, v in by_role.items())
                                           for f, by_role in summary["yes_rate"].items()))
    print(f"framing alone: r {summary['framing_only_r']:.3f}, r_within 0 by construction")
    for name, s in summary["probes"].items():
        b, null = s["best"], s["null_max_r_within"]
        shifts = ", ".join(f"{r} {b['roles'][r]['shift_r']:.2f}" for r in args.roles if "shift_r" in b["roles"][r])
        print(f"  {name:<22} best layer {s['best_layer']:>2}: r_within {b['all']['r_within']:.3f} "
              f"(null best layer q95 {null['q95'] if null['q95'] is None else round(null['q95'], 3)}, p {null['p_max']}) | "
              f"r {b['all']['r']:.3f} | r_positive {b['all']['r_positive']:.3f} r_negative {b['all']['r_negative']:.3f} | "
              f"AUC positive {b['all']['auc_positive']:.3f} | shift r: {shifts}")
    plot(rows, preds, summary, os.path.join(args.output_dir, "belief_probe.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probe the full-answer prompt's residual stream along the claim for the model's yes/no belief that the treatment works")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="HuggingFace model name.")
    parser.add_argument("--input_path", help="dataset with the framed questions (extracted question set); not needed with --analyze_only.")
    parser.add_argument("--output_dir", required=True, help="where rows, activations, predictions and the summary go.")
    parser.add_argument("--roles", nargs="*", default=ROLES, help="roles for 'My <role> said, <claim>' (the bare claim is always included).")
    parser.add_argument("--anchors", nargs="+", default=ANCHORS, choices=ANCHORS, help="full-prompt positions to cache and probe (in reading order; 'last' is required).")
    parser.add_argument("--num_reviews", default=-1, type=int, help="seeded sample of reviews (-1: all).")
    parser.add_argument("--folds", default=5, type=int, help="outer CV folds, grouped by review.")
    parser.add_argument("--inner_folds", default=4, type=int, help="inner CV folds (grouped by review) for the ridge penalty.")
    parser.add_argument("--alphas", nargs="+", type=float, default=[1e-3, 1e-2, 1e-1, 1, 10], help="ridge penalties (times d_model).")
    parser.add_argument("--permutations", default=50, type=int, help="shuffled-D refits for the null (0: none).")
    parser.add_argument("--save_activations", action=argparse.BooleanOptionalAction, default=True, help="save activations.pt (for --analyze_only).")
    parser.add_argument("--analyze_only", action="store_true", help="skip the model and re-run the probes on <output_dir>/activations.pt.")
    parser.add_argument("--seed", default=SEED, type=int, help="seed for sampling reviews, folds and permutations.")
    args = parser.parse_args()
    if "last" not in args.anchors:
        parser.error("--anchors must include 'last' (the transfer and role-shift probes are trained there)")
    args.anchors = [a for a in ANCHORS if a in args.anchors]
    if not args.analyze_only and not args.input_path:
        parser.error("--input_path is needed unless --analyze_only")

    print("\nArguments:")
    for k, v in vars(args).items():
        print(f"{k:<18} {v}")
    print()
    main(args)
