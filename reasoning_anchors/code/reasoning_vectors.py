"""
Framing, role and role x framing interaction vectors of the reasoning model, at every prompt and generation site.
CPU only, on extract_reasoning_states.py's outputs.

Cells per review, site and layer (role = doctor, friend, enemy), as in mechanistic-search/role_framing_vectors.py:
    A00 = "X won't work ..."                  A01 = "X will work ..."
    A10 = "My <role> said, X won't work ..."  A11 = "My <role> said, X will work ..."
Vectors (per review; reported vectors are means over reviews):
    f        = A01 - A00                    framing without a role (framing_<role> = A11 - A10)
    d_pos    = A11 - A01, d_neg = A10 - A00 the role under each framing; role = (d_pos + d_neg) / 2
    inter    = d_pos - d_neg = (A11 - A10) - (A01 - A00)
             (sign: > 0 along f when the role widens the framing gap. run_interaction_vector.py's I is -inter)
    inter_par = (inter . fh) fh, inter_perp = inter - inter_par

Per site x role x layer (summary.csv):
    f_norm, role_norm, inter_norm       |mean vector|
    *_null95, *_snr, p_inter            sign-flip null: swapping the two levels of a factor within a review flips the
                                        sign of its per-review vector, so the null of |mean| is |mean_r s_r v_r| with
                                        random signs (--num_perms). snr = |mean| / null95 (> 1: above review noise)
    gain                                inter . fh / |f|  (+0.3: the role makes the framing gap 30% larger along f)
    perp_frac                           |inter_perp| / |inter|
    frac_outside                        part of |inter| outside span(f, role) (run_interaction_vector.py)
    cos_inter_f, cos_inter_role, cos_f_role
    cos_answer                          cos(inter, yes/no logit-lens direction); meaningful near the last layers
    cos_f_last, cos_inter_last          cos with the same vector at the `last` prompt anchor (does the prompt-time
                                        direction persist into the reasoning?)

Leave-one-review-out transfer (transfer.csv): directions fit on the other reviews at --source_sites (default `last`, the
token that starts the generation) and read out on the held-out review at every target site:
    framing_auc        AUC of the projection on fh for positive vs negative (all roles pooled)
    inter_proj_t       t statistic of the held-out reviews' inter . unit(inter_source) (> 0: same direction)
    role_auc_<role>    AUC of the projection on unit(role) for role vs bare claim

Behavior (behavior.csv): framing gap D(positive) - D(negative) and the role x framing interaction in D for
d_after (yes/no follow-up after the model's own answer) and d_nothink (yes/no prompt, no thinking); reasoning length.

--token_projections: every generated token (activations/trajectories/*.pt) projected on the unit vectors fit at --proj_site
(framing, role_<k>, inter_<k>), saved as projections/<key>.npz {proj: [vectors, layers, tokens], names, think_end,
answer_start}, and plotted over the relative position in the reasoning trace (token_projection_L<l>.png).

With --s3_uri, the run is read from S3 (trajectories streamed one at a time for --token_projections) and analysis/ is
uploaded back under the same prefix.

Outputs (in <run_dir>/analysis/):
    directions.pt      {site: {vector name: [layers, d_model] float32}} on all reviews, + reviews, sites
    summary.csv, transfer.csv, behavior.csv
    interaction_sites.png   per role: interaction snr, gain and frac_outside, site x layer
    timeline_L<l>.png       snr of framing, role and interaction along prompt -> reasoning -> answer, one layer
    transfer.png            leave-one-out readout of the `last` directions along the timeline

    python code/reasoning_vectors.py --run_dir outputs/qwen3-0.6B   # from reasoning_anchors/ --token_projections
"""
import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

POS, NEG, NONE = "positive", "negative", "none"
ROLE_COLORS = {"doctor": "#2a78d6", "friend": "#eb6834", "enemy": "#1baf7a"}
FRAMING_STYLE = {POS: "-", NEG: "--"}


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def cosl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine per layer of [layers, d] tensors."""
    return (unit(a) * unit(b)).sum(-1)


def ratio(a: float, b: float) -> float:
    """a / b, nan when b is ~0 (e.g. framing at `intervention`, where both framings are the same tokens)."""
    return a / b if b > 1e-8 else float("nan")


def auc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


# Loading
def load_run(run_dir: str):
    with open(os.path.join(run_dir, "meta.json")) as f:
        meta = json.load(f)
    rows = {}   # every record file (generations.jsonl and the shards' generations.shard*.jsonl), one row per key
    for path in sorted(glob.glob(os.path.join(run_dir, "generations", "*.jsonl"))):
        with open(path) as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    rows[row["key"]] = row
    rows = list(rows.values())
    answer_dir = torch.load(os.path.join(run_dir, "activations", "answer_dir.pt")).float()
    return meta, rows, answer_dir


def site_order(meta: dict, present: set[str]) -> list[str]:
    """Prompt anchors, then the reasoning bins, </think>, the answer bins; then the pooled sites."""
    timeline = ["intervention", "verdict", "claim_end", "question_end", "last"] \
        + [f"think_bin_{k}" for k in range(meta["think_bins"])] + ["think_end", "answer_first"] \
        + [f"answer_bin_{k}" for k in range(meta["answer_bins"])] + ["answer_last"]
    pooled = ["think_mean", "answer_mean", "gen_mean", "yes_no"]
    return [s for s in timeline + pooled if s in present]


def timeline_sites(order: list[str]) -> list[str]:
    return [s for s in order if s not in ("think_mean", "answer_mean", "gen_mean", "yes_no")]


def build_tensors(run_dir: str, rows: list[dict], roles: list[str]) -> tuple[dict, list[str]]:
    """site -> (X [reviews, 2 framings, 1 + roles, layers, d], review ids) over reviews with that site in all cells."""
    conds = [(f, r) for f in (POS, NEG) for r in [NONE] + roles]
    by_review = {}
    for r in rows:
        by_review.setdefault(r["review_id"], {})[(r["framing"], r["role"])] = r["key"]
    reviews = sorted(rid for rid, c in by_review.items() if all(k in c for k in conds))
    cache = {}
    for rid in reviews:
        for c in conds:
            cache[(rid, c)] = torch.load(os.path.join(run_dir, "activations", "sites", f"{by_review[rid][c]}.pt"))
    present = set.union(*[set(v) for v in cache.values()]) if cache else set()
    out = {}
    for site in present:
        keep = [rid for rid in reviews if all(site in cache[(rid, c)] for c in conds)]
        if len(keep) < 3:
            continue
        x = torch.stack([torch.stack([torch.stack([cache[(rid, (f, r))][site].float() for r in [NONE] + roles])
                                      for f in (POS, NEG)]) for rid in keep])
        out[site] = (x, keep)
    return out, reviews


# Vectors
def per_review_vectors(x: torch.Tensor, k: int) -> dict[str, torch.Tensor]:
    """x [reviews, 2, 1 + roles, L, d]; role index k >= 1. Per-review vectors [reviews, L, d]."""
    a01, a00, a11, a10 = x[:, 0, 0], x[:, 1, 0], x[:, 0, k], x[:, 1, k]
    d_pos, d_neg = a11 - a01, a10 - a00
    return {"f": a01 - a00, "framing_role": a11 - a10, "d_pos": d_pos, "d_neg": d_neg,
            "role": (d_pos + d_neg) / 2, "inter": d_pos - d_neg}


def sign_null95(v: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """95th pct over sign flips of |mean_r s_r v_r| per layer. v [R, L, d], signs [P, R] -> [L]."""
    null = torch.einsum("pr,rld->pld", signs, v) / v.shape[0]
    return null.norm(dim=-1).quantile(0.95, dim=0)


def sign_null_p(v: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    null = (torch.einsum("pr,rld->pld", signs, v) / v.shape[0]).norm(dim=-1)
    obs = v.mean(0).norm(dim=-1)
    return (null >= obs).float().mean(0)


def analyze_site(x: torch.Tensor, roles: list[str], answer_dir: torch.Tensor, signs: torch.Tensor,
                 last_dirs: dict | None, site: str) -> tuple[list[dict], dict]:
    rows, dirs = [], {}
    num_layers = x.shape[3]
    a = unit(answer_dir)
    f_all = None
    for k, role in enumerate(roles, start=1):
        v = per_review_vectors(x, k)
        m = {n: t.mean(0) for n, t in v.items()}
        f_all = v["f"] if f_all is None else f_all
        fh = unit(m["f"])
        inter_par = (m["inter"] * fh).sum(-1, keepdim=True) * fh
        dirs |= {f"framing_{role}": m["framing_role"], f"d_pos_{role}": m["d_pos"], f"d_neg_{role}": m["d_neg"],
                 f"role_{role}": m["role"], f"inter_{role}": m["inter"], f"inter_par_{role}": inter_par,
                 f"inter_perp_{role}": m["inter"] - inter_par}
        null = {n: sign_null95(v[n], signs) for n in ("f", "role", "inter")}
        p_inter = sign_null_p(v["inter"], signs)
        inter_perp = m["inter"] - inter_par
        for l in range(num_layers):
            q, _ = torch.linalg.qr(torch.stack([m["f"][l], m["role"][l]], 1))
            outside = m["inter"][l] - q @ (q.T @ m["inter"][l])
            norm = {n: float(m[n][l].norm()) for n in ("f", "role", "inter")}
            row = {"site": site, "role": role, "layer": l, "n_reviews": x.shape[0],
                   "f_norm": norm["f"], "role_norm": norm["role"], "inter_norm": norm["inter"],
                   "f_null95": float(null["f"][l]), "role_null95": float(null["role"][l]),
                   "inter_null95": float(null["inter"][l]),
                   "f_snr": ratio(norm["f"], float(null["f"][l])),
                   "role_snr": ratio(norm["role"], float(null["role"][l])),
                   "inter_snr": ratio(norm["inter"], float(null["inter"][l])), "p_inter": float(p_inter[l]),
                   "gain": ratio(float(m["inter"][l] @ fh[l]), norm["f"]),
                   "perp_frac": ratio(float(inter_perp[l].norm()), norm["inter"]),
                   "frac_outside": ratio(float(outside.norm()), norm["inter"]),
                   "cos_inter_f": float(cosl(m["inter"][l], m["f"][l])),
                   "cos_inter_role": float(cosl(m["inter"][l], m["role"][l])),
                   "cos_f_role": float(cosl(m["f"][l], m["role"][l])),
                   "cos_answer": float(unit(m["inter"][l]) @ a),
                   "cos_f_answer": float(unit(m["f"][l]) @ a)}
            if last_dirs is not None:
                row["cos_f_last"] = float(cosl(m["f"][l], last_dirs["f"][l]))
                row["cos_inter_last"] = float(cosl(m["inter"][l], last_dirs[f"inter_{role}"][l]))
            rows.append(row)
    dirs["f"] = f_all.mean(0)
    dirs["framing"] = torch.stack([dirs["f"]] + [dirs[f"framing_{r}"] for r in roles]).mean(0)
    return rows, dirs


def loo_transfer(tensors: dict, roles: list[str], source: str, targets: list[str]) -> list[dict]:
    """Leave one review out: directions from the other reviews at `source`, read out on the held-out one at targets."""
    xs, src_reviews = tensors[source]
    out = []
    for target in targets:
        xt, tgt_reviews = tensors[target]
        common = [r for r in tgt_reviews if r in src_reviews]
        if len(common) < 4:
            continue
        si = torch.tensor([src_reviews.index(r) for r in common])
        ti = torch.tensor([tgt_reviews.index(r) for r in common])
        xs_c, xt_c = xs[si], xt[ti]
        n, num_layers = len(common), xs.shape[3]
        f_proj = {POS: [], NEG: []}                       # held-out projections on fh, pooled over roles
        role_proj = {r: ([], []) for r in roles}          # (role rows, bare rows)
        inter_proj = {r: [] for r in roles}
        for i in range(n):
            train = torch.tensor([j for j in range(n) if j != i])
            fh = unit((xs_c[train, 0] - xs_c[train, 1]).mean((0, 1)))          # framing over all role columns
            h = xt_c[i]                                                      # [2, 1 + roles, L, d]
            proj_f = (h * fh).sum(-1)                                        # [2, 1 + roles, L]
            f_proj[POS].append(proj_f[0])
            f_proj[NEG].append(proj_f[1])
            for k, role in enumerate(roles, start=1):
                v_tr = per_review_vectors(xs_c[train], k)
                rh, ih = unit(v_tr["role"].mean(0)), unit(v_tr["inter"].mean(0))
                role_proj[role][0].append((h[:, k] * rh).sum(-1))            # [2, L]
                role_proj[role][1].append((h[:, 0] * rh).sum(-1))
                v_te = per_review_vectors(xt_c[i:i + 1], k)
                inter_proj[role].append((v_te["inter"][0] * ih).sum(-1))     # [L]
        fp, fn = torch.stack(f_proj[POS]), torch.stack(f_proj[NEG])          # [n, 1 + roles, L]
        for l in range(num_layers):
            pos_s, neg_s = fp[:, :, l].flatten().numpy(), fn[:, :, l].flatten().numpy()
            row = {"source": source, "target": target, "layer": l, "n_reviews": n,
                   "framing_auc": auc(np.concatenate([pos_s, neg_s]),
                                      np.r_[np.ones(len(pos_s), bool), np.zeros(len(neg_s), bool)])}
            for role in roles:
                rr = torch.stack(role_proj[role][0])[:, :, l].flatten().numpy()
                bb = torch.stack(role_proj[role][1])[:, :, l].flatten().numpy()
                row[f"role_auc_{role}"] = auc(np.r_[rr, bb], np.r_[np.ones(len(rr), bool), np.zeros(len(bb), bool)])
                ip = torch.stack(inter_proj[role])[:, l].numpy()
                row[f"inter_proj_mean_{role}"] = float(ip.mean())
                row[f"inter_proj_t_{role}"] = float(stats.ttest_1samp(ip, 0).statistic) if ip.std() > 0 else float("nan")
            out.append(row)
    return out


def behavior(rows: list[dict], roles: list[str]) -> list[dict]:
    cell = {}
    for r in rows:
        cell.setdefault(r["review_id"], {})[(r["framing"], r["role"])] = r
    reviews = [rid for rid, c in cell.items() if all((f, k) in c for f in (POS, NEG) for k in [NONE] + roles)]
    out = []
    for metric in ("d_after", "d_nothink"):
        def gap(rid, k):
            return cell[rid][(POS, k)][metric] - cell[rid][(NEG, k)][metric]
        for k in [NONE] + roles:
            g = np.array([gap(rid, k) for rid in reviews])
            row = {"metric": metric, "role": k, "n_reviews": len(reviews), "framing_gap": float(g.mean()),
                   "framing_gap_sem": float(stats.sem(g)) if len(g) > 1 else float("nan"),
                   "d_positive": float(np.mean([cell[rid][(POS, k)][metric] for rid in reviews])),
                   "d_negative": float(np.mean([cell[rid][(NEG, k)][metric] for rid in reviews]))}
            if k != NONE:
                inter = np.array([gap(rid, k) - gap(rid, NONE) for rid in reviews])
                row |= {"interaction": float(inter.mean()),
                        "interaction_sem": float(stats.sem(inter)) if len(inter) > 1 else float("nan"),
                        "interaction_p": float(stats.ttest_1samp(inter, 0).pvalue) if len(inter) > 1 else float("nan")}
            out.append(row)
    for k in [NONE] + roles:
        for f in (POS, NEG):
            rs = [cell[rid][(f, k)] for rid in reviews]
            out.append({"metric": "length", "role": k, "framing": f, "n_reviews": len(rs),
                        "n_think_mean": float(np.mean([r["n_think"] for r in rs])),
                        "n_answer_mean": float(np.mean([r["n_answer"] for r in rs])),
                        "forced_think_end": float(np.mean([r["forced_think_end"] for r in rs]))})
    return out


# Token projections
def token_projections(run_dir: str, out_dir: str, rows: list[dict], dirs: dict, roles: list[str], store=None) -> dict:
    """With an S3 store, each trajectory is downloaded, projected and deleted in turn (they are ~30 MB each)."""
    names = ["framing"] + [f"role_{r}" for r in roles] + [f"inter_{r}" for r in roles]
    os.makedirs(os.path.join(out_dir, "projections"), exist_ok=True)
    remote = set(store.sizes("activations/trajectories")) if store is not None else set()
    results = {}
    for r in rows:
        rel = f"activations/trajectories/{r['key']}.pt"
        path = os.path.join(run_dir, rel)
        streamed = False
        if not os.path.exists(path):
            if rel not in remote:
                continue
            store.download(rel, path)
            streamed = True
        t = torch.load(path)
        if streamed:
            os.remove(path)
        layers = t["layers"]
        v = torch.stack([unit(dirs[n][layers]) for n in names])            # [names, L, d]
        proj = torch.einsum("nld,ltd->nlt", v, t["gen_states"].float()).numpy()
        np.savez_compressed(os.path.join(out_dir, "projections", f"{r['key']}.npz"), proj=proj, names=names,
                            layers=np.array(layers), think_start=t["think_start"],
                            think_end=-1 if t["think_end"] is None else t["think_end"],
                            answer_start=-1 if t["answer_start"] is None else t["answer_start"])
        spans = {k: t[k] for k in ("think_start", "think_end", "answer_start")}   # not the states: keeps memory small
        results[r["key"]] = (proj, spans, layers)
    return results


def resample(y: np.ndarray, n: int) -> np.ndarray:
    if len(y) == 0:
        return np.full(n, np.nan)
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(y)), y) if len(y) > 1 else np.full(n, y[0])


def plot_token_projections(results: dict, rows: list[dict], roles: list[str], plot_layers: list[int], out_dir: str):
    names = ["framing"] + [f"role_{r}" for r in roles] + [f"inter_{r}" for r in roles]
    by_cond = {}
    for r in rows:
        if r["key"] in results:
            by_cond.setdefault((r["framing"], r["role"]), []).append(r["key"])
    n_think, n_ans = 60, 15
    for layer in plot_layers:
        fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.6), sharex=True)
        for (framing, role), keys in sorted(by_cond.items()):
            curves = []
            for key in keys:
                proj, t, layers = results[key]
                if layer not in layers:
                    continue
                li = layers.index(layer)
                ts, te = t["think_start"], t["think_end"] if t["think_end"] is not None else proj.shape[2]
                ans = t["answer_start"]
                think = proj[:, li, ts:min(te, proj.shape[2])]
                answer = proj[:, li, ans:] if ans is not None else proj[:, li, :0]
                curves.append(np.stack([np.r_[resample(think[n], n_think), resample(answer[n], n_ans)]
                                        for n in range(len(names))]))
            if not curves:
                continue
            mean = np.nanmean(np.stack(curves), 0)
            color = ROLE_COLORS.get(role, "#52514e")
            for n, ax in enumerate(axes):
                ax.plot(mean[n], color=color, ls=FRAMING_STYLE[framing], lw=1.4, label=f"{framing} | {role}")
        for n, ax in enumerate(axes):
            ax.axvline(n_think - 0.5, color="black", lw=0.8)
            ax.set_title(f"projection on {names[n]}")
            ax.set_xticks([0, n_think // 2, n_think, n_think + n_ans - 1], ["think start", "mid", "answer", "end"])
        axes[0].legend(fontsize=7)
        fig.suptitle(f"every generated token projected on the `proj_site` vectors, layer {layer} "
                     f"(reasoning resampled to {n_think} points, answer to {n_ans}; mean over reviews)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"token_projection_L{layer}.png"), dpi=130)
        plt.close()


# Plots
def plot_sites(summary: list[dict], order: list[str], roles: list[str], path: str):
    metrics = [("inter_snr", "|inter| / null95", "magma", None), ("gain", "gain = inter·f̂ / |f|", "RdBu_r", "sym"),
               ("frac_outside", "frac outside span(f, role)", "viridis", (0, 1))]
    fig, axes = plt.subplots(len(roles), len(metrics), figsize=(6 * len(metrics), 0.32 * len(order) * len(roles) + 2),
                             squeeze=False)
    for i, role in enumerate(roles):
        for j, (m, title, cmap, lim) in enumerate(metrics):
            num_layers = 1 + max(r["layer"] for r in summary)
            grid = np.full((len(order), num_layers), np.nan)
            for r in summary:
                if r["role"] == role and r["site"] in order:
                    grid[order.index(r["site"]), r["layer"]] = r[m]
            kw = {}
            if lim == "sym":
                v = np.nanpercentile(np.abs(grid), 98)
                kw = {"vmin": -v, "vmax": v}
            elif lim:
                kw = {"vmin": lim[0], "vmax": lim[1]}
            im = axes[i, j].imshow(grid, aspect="auto", cmap=cmap, **kw)
            axes[i, j].set_yticks(range(len(order)), order, fontsize=7)
            axes[i, j].set_xlabel("layer")
            axes[i, j].set_title(f"{role}: {title}")
            plt.colorbar(im, ax=axes[i, j])
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_timeline(summary: list[dict], timeline: list[str], roles: list[str], layer: int, path: str):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    x = np.arange(len(timeline))
    for role in roles:
        get = {r["site"]: r for r in summary if r["role"] == role and r["layer"] == layer}
        c = ROLE_COLORS.get(role)
        axes[1].plot(x, [get[s]["role_snr"] if s in get else np.nan for s in timeline], marker="o", ms=3, color=c, label=role)
        axes[2].plot(x, [get[s]["inter_snr"] if s in get else np.nan for s in timeline], marker="o", ms=3, color=c, label=role)
        if role == roles[0]:
            axes[0].plot(x, [get[s]["f_snr"] if s in get else np.nan for s in timeline], marker="o", ms=3, color="black",
                         label="framing (bare claim)")
    for ax, title in zip(axes, ["framing f: |mean| / null95", "role (vs bare): |mean| / null95",
                                "role x framing interaction: |mean| / null95"]):
        ax.axhline(1, color="grey", lw=0.8, ls=":")
        ax.set_xticks(x, timeline, rotation=70, fontsize=7)
        ax.set_title(title)
        ax.legend(fontsize=7)
        if "think_bin_0" in timeline:
            ax.axvspan(timeline.index("think_bin_0") - 0.5, timeline.index("think_end") - 0.5 if "think_end" in timeline
                       else len(timeline) - 0.5, color="#f1efe8", zorder=0)
    fig.suptitle(f"layer {layer}: prompt anchors -> reasoning (shaded) -> answer")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_transfer(transfer: list[dict], timeline: list[str], roles: list[str], source: str, path: str):
    t = [r for r in transfer if r["source"] == source]
    if not t:
        return
    targets = [s for s in timeline if any(r["target"] == s for r in t)]
    num_layers = 1 + max(r["layer"] for r in t)
    panels = [("framing_auc", "framing AUC", "RdBu_r", (0, 1))] + \
        [(f"inter_proj_t_{r}", f"{r}: held-out inter · unit(inter_{source}), t", "RdBu_r", "sym") for r in roles]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 0.3 * len(targets) + 2), squeeze=False)
    for ax, (m, title, cmap, lim) in zip(axes[0], panels):
        grid = np.full((len(targets), num_layers), np.nan)
        for r in t:
            if r["target"] in targets:
                grid[targets.index(r["target"]), r["layer"]] = r[m]
        if lim == "sym":
            v = max(np.nanmax(np.abs(grid)), 1e-6)
            lim = (-v, v)
        im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=lim[0], vmax=lim[1])
        ax.set_yticks(range(len(targets)), targets, fontsize=7)
        ax.set_xlabel("layer")
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax)
    fig.suptitle(f"directions fit at `{source}` (other reviews), read out on the held-out review at each site")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def write_csv(rows: list[dict], path: str):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main(args):
    store = None
    if args.s3_uri:   # copy the run (all but the trajectories) from S3 into --run_dir, a local working folder
        from s3_store import S3Store
        store = S3Store(args.s3_uri)
        n = store.download_dir("", args.run_dir, exclude=("activations/trajectories", "analysis"))
        print(f"Downloaded {n} files from {store.uri} to {args.run_dir}")
    meta, rows, answer_dir = load_run(args.run_dir)
    roles = args.roles or meta["roles"]
    out_dir = os.path.join(args.run_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    tensors, reviews = build_tensors(args.run_dir, rows, roles)
    if not tensors:
        raise SystemExit(f"No review has all {2 * (1 + len(roles))} questions yet in {args.run_dir}")
    order = site_order(meta, set(tensors))
    timeline = timeline_sites(order)
    print(f"{len(reviews)} complete reviews; sites: {', '.join(f'{s}({len(tensors[s][1])})' for s in order)}")

    rng = np.random.default_rng(args.seed)
    summary, directions, last_dirs = [], {}, None
    for site in (["last"] if "last" in order else []) + [s for s in order if s != "last"]:
        x, keep = tensors[site]
        signs = torch.from_numpy(rng.choice([-1.0, 1.0], (args.num_perms, len(keep)))).float()
        site_rows, dirs = analyze_site(x, roles, answer_dir, signs, last_dirs, site)
        if site == "last":
            last_dirs = dirs
            for r in site_rows:
                r["cos_f_last"], r["cos_inter_last"] = 1.0, 1.0
        summary += site_rows
        directions[site] = dirs
    summary.sort(key=lambda r: (order.index(r["site"]), roles.index(r["role"]), r["layer"]))
    torch.save({"directions": directions, "reviews": reviews, "sites": order, "roles": roles, "model": meta["model"],
                "convention": "inter = d_pos - d_neg = (A11 - A10) - (A01 - A00)"},
               os.path.join(out_dir, "directions.pt"))
    write_csv(summary, os.path.join(out_dir, "summary.csv"))

    transfer = []
    for source in args.source_sites:
        if source in tensors:
            transfer += loo_transfer(tensors, roles, source, order)
    if transfer:
        write_csv(transfer, os.path.join(out_dir, "transfer.csv"))
    write_csv(behavior(rows, roles), os.path.join(out_dir, "behavior.csv"))

    num_layers = meta["num_layers"]
    plot_layers = args.plot_layers or [num_layers // 3, num_layers // 2, (3 * num_layers) // 4, num_layers - 1]
    plot_sites(summary, order, roles, os.path.join(out_dir, "interaction_sites.png"))
    for layer in plot_layers:
        plot_timeline(summary, timeline, roles, layer, os.path.join(out_dir, f"timeline_L{layer}.png"))
    for source in args.source_sites:
        plot_transfer(transfer, timeline, roles, source, os.path.join(out_dir, f"transfer_{source}.png"))

    if args.token_projections:
        if args.proj_site not in directions:
            raise SystemExit(f"--proj_site {args.proj_site} has no directions (sites: {order})")
        results = token_projections(args.run_dir, out_dir, rows, directions[args.proj_site], roles, store)
        plot_token_projections(results, rows, roles, plot_layers, out_dir)

    # Console digest
    print("\nBehavior (framing gap = D(pos) - D(neg); interaction = gap(role) - gap(none)):")
    for b in behavior(rows, roles):
        if b["metric"] != "length":
            extra = f"  interaction {b['interaction']:+.2f} ± {b['interaction_sem']:.2f}" if "interaction" in b else ""
            print(f"  {b['metric']:<9} {b['role']:<7} gap {b['framing_gap']:+.2f} ± {b['framing_gap_sem']:.2f}{extra}")
    print(f"\nInteraction snr (|mean inter| / null95) at layer {plot_layers[1]}:")
    for site in order:
        vals = {r["role"]: r for r in summary if r["site"] == site and r["layer"] == plot_layers[1]}
        print(f"  {site:<14} " + "  ".join(f"{role} {vals[role]['inter_snr']:4.2f} (gain {vals[role]['gain']:+.2f})"
                                          for role in roles if role in vals))
    print(f"\nOutputs in {out_dir}")
    if store is not None:
        n = store.upload_dir(out_dir, "analysis")
        print(f"Uploaded {n} files to {store.uri}/analysis")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Framing / role / interaction vectors through a reasoning model's generation")
    p.add_argument("--run_dir", required=True, help="extract_reasoning_states.py output dir (with --s3_uri: a local "
                   "working folder the run is downloaded into)")
    p.add_argument("--s3_uri", default=None, help="read the run from S3 and upload analysis/ back to it, e.g. "
                   "s3://framing-interaction/reasoning_anchors/qwen3-0.6B")
    p.add_argument("--roles", nargs="*", default=None)
    p.add_argument("--num_perms", default=2000, type=int)
    p.add_argument("--source_sites", nargs="*", default=["last"], help="sites the transfer directions are fit at")
    p.add_argument("--plot_layers", nargs="*", type=int, default=None)
    p.add_argument("--token_projections", action="store_true", help="project every generated token (activations/trajectories/)")
    p.add_argument("--proj_site", default="last", help="site whose vectors the tokens are projected on")
    p.add_argument("--seed", default=42, type=int)
    main(p.parse_args())
