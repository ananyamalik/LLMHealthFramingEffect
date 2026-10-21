"""
Is there a "risk" direction in the residual stream, and does the doctor/friend source effect use it?

All prompts use the Yes/No template (prompts/question_answering_yes_no.jinja2), and states are the residual stream at
the last prompt token (where the verdict is read out), at every layer.

1. Directions (fit on the first half of the reviews):
       risk     = mean over pairs of h(harm wording) - h(safe wording)     safety: safe/dangerous, studies: good/harmful
       valence  = mean over pairs of h(negative wording) - h(positive)     effectiveness, efficacy
       risk_orth = risk with its valence component removed
2. Projection test (held-out reviews): project bare / "My doctor said," / "My friend said," prompts (same claims as
   the doctor contrast, positive and negative) onto each unit direction. Differences are reported in units of the
   held-out harm-vs-safe separation along that direction (1 = as far apart as "dangerous" vs "safe"). Also: cosine
   between the mean friend-minus-doctor difference and each direction, against random directions, and the per-pair
   correlation between the projection difference and the Yes/No verdict difference.
3. Steering test (held-out reviews): add alpha * separation * unit direction to the last token at a few layers of the
   doctor prompt and measure the change in the Yes/No readout, against random directions of the same norm.

Outputs (in --output_dir):
    directions.pt                 unit directions and separations per layer
    states_records.jsonl          per prompt: readout and projections onto each direction at every layer
    steering_records.jsonl        per prompt x direction x layer x alpha: readout change
    risk_direction_summary.json   projection and steering summaries
    risk_direction.png
"""
import argparse
import json
import os
import math
import gc

import numpy as np
import torch
from tqdm import tqdm

from patching_utils import (
    DEFAULT_MODEL_NAME, SEED, DOCTOR_PREFIX, set_global_seed, load_model_and_tokenizer, get_decoder_layers,
    format_rct_inputs, build_prompt_ids, load_dataset, sample_reviews, get_prompt_pairs, cache_residuals,
    add_to_residual,
)
from run_yes_no_patching import TEMPLATE_NAME, TEMPLATE_DIR, YES_STRINGS, NO_STRINGS, single_token_ids, readout

RISK_TYPES = ["safety", "studies"]          # positive_question = safe/good wording, negative_question = dangerous/harmful
VALENCE_TYPES = ["effectiveness", "efficacy"]
SOURCES = {"bare": "{claim}", "doctor": f"{DOCTOR_PREFIX}, {{claim}}", "friend": "My friend said, {claim}"}
DIRECTIONS = ["risk", "valence", "risk_orth"]


@torch.no_grad()
def last_token_states(model, prompt_ids: torch.Tensor) -> torch.Tensor:
    """Residual stream at the last prompt token after every decoder layer: [num_layers, d_model] float32 on CPU."""
    with cache_residuals(model, [prompt_ids.shape[1] - 1], device="cpu") as cache:
        model.model(input_ids=prompt_ids, use_cache=False)
    return torch.stack([cache[l][0].float() for l in sorted(cache)])


def encode(tokenizer, question, rct_inputs, device):
    return build_prompt_ids(tokenizer, question, rct_inputs, device, template_name=TEMPLATE_NAME, template_dir=TEMPLATE_DIR)


def framing_pair_states(model, tokenizer, examples, q_types) -> tuple[list, list]:
    """(h(positive wording), h(negative wording)) per pair for the given question types."""
    pos, neg = [], []
    for ex in tqdm(examples, desc=f"states {'/'.join(q_types)}"):
        rct = format_rct_inputs(ex["Inputs"])
        for q in q_types:
            if q not in ex["Questions"]:
                continue
            pos.append(last_token_states(model, encode(tokenizer, ex["Questions"][q]["positive_question"], rct, model.device)))
            neg.append(last_token_states(model, encode(tokenizer, ex["Questions"][q]["negative_question"], rct, model.device)))
    return pos, neg


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def fit_directions(risk_pos, risk_neg, val_pos, val_neg) -> dict[str, torch.Tensor]:
    """Unit directions per layer [num_layers, d_model]."""
    risk = unit(torch.stack([n - p for p, n in zip(risk_pos, risk_neg)]).mean(0))
    valence = unit(torch.stack([n - p for p, n in zip(val_pos, val_neg)]).mean(0))
    orth = unit(risk - (risk * valence).sum(-1, keepdim=True) * valence)
    return {"risk": risk, "valence": valence, "risk_orth": orth}


def separation(pos_states, neg_states, direction: torch.Tensor) -> torch.Tensor:
    """Mean projection of negative-wording minus positive-wording states along `direction`, per layer."""
    proj = lambda states: torch.stack([(s * direction).sum(-1) for s in states]).mean(0)
    return proj(neg_states) - proj(pos_states)


def main(args):
    set_global_seed(SEED)
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    dataset = sample_reviews(load_dataset(args.input_path), args.sample_pool, seed=args.seed)[:args.num_reviews]
    if args.debug:
        dataset = dataset[:4]
    half = len(dataset) // 2
    fit_reviews, test_reviews = dataset[:half], dataset[half:]
    print(f"Reviews: {len(fit_reviews)} to fit directions, {len(test_reviews)} held out")

    model, tokenizer = load_model_and_tokenizer(args.model)
    num_layers = len(get_decoder_layers(model))
    yes_ids, no_ids = single_token_ids(tokenizer, YES_STRINGS), single_token_ids(tokenizer, NO_STRINGS)

    # 1. Directions on the fit reviews; separations on held-out reviews (so the unit of the projection test is not fit on itself)
    risk_pos, risk_neg = framing_pair_states(model, tokenizer, fit_reviews, RISK_TYPES)
    val_pos, val_neg = framing_pair_states(model, tokenizer, fit_reviews, VALENCE_TYPES)
    directions = fit_directions(risk_pos, risk_neg, val_pos, val_neg)
    held_risk_pos, held_risk_neg = framing_pair_states(model, tokenizer, test_reviews, RISK_TYPES)
    held_val_pos, held_val_neg = framing_pair_states(model, tokenizer, test_reviews, VALENCE_TYPES)
    sep = {
        "risk": separation(held_risk_pos, held_risk_neg, directions["risk"]),
        "risk_orth": separation(held_risk_pos, held_risk_neg, directions["risk_orth"]),
        "valence": separation(held_val_pos, held_val_neg, directions["valence"]),
    }
    cos_risk_valence = (directions["risk"] * directions["valence"]).sum(-1)
    torch.save({"directions": directions, "separation": sep, "cos_risk_valence": cos_risk_valence}, os.path.join(args.output_dir, "directions.pt"))
    del risk_pos, risk_neg, val_pos, val_neg, held_val_pos, held_val_neg
    gc.collect()

    # 2. Source prompts on held-out reviews
    records, states = [], {}
    for ex in tqdm(test_reviews, desc="source prompts"):
        rct = format_rct_inputs(ex["Inputs"])
        for pair in get_prompt_pairs(ex, "doctor", None):
            claim, framing = pair["target_question"], pair["question_key"].rsplit("_", 1)[1]
            for source, template in SOURCES.items():
                ids = encode(tokenizer, template.format(claim=claim), rct, model.device)
                h = last_token_states(model, ids)
                key = (ex["ReviewID"], pair["question_key"], source)
                states[key] = h
                records.append({
                    "review_id": ex["ReviewID"], "question_key": pair["question_key"], "framing": framing, "source": source,
                    "readout_d": readout(model, ids, yes_ids, no_ids)["d"],
                    "projection": {d: (h * directions[d]).sum(-1).tolist() for d in DIRECTIONS},
                })
    with open(os.path.join(args.output_dir, "states_records.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    summary = {"num_fit_reviews": len(fit_reviews), "num_test_reviews": len(test_reviews),
               "cos_risk_valence_by_layer": cos_risk_valence.tolist(),
               "separation_by_layer": {d: v.tolist() for d, v in sep.items()},
               "projection": {}, "steering": {}}

    # Projection differences (paired by claim), in separation units, per layer
    by_key = {(r["review_id"], r["question_key"], r["source"]): r for r in records}
    pairs = sorted({(r["review_id"], r["question_key"]) for r in records})
    random_dirs = unit(torch.from_numpy(rng.standard_normal((args.num_random, directions["risk"].shape[-1]))).float())
    for comparison, (a, b) in {"friend_minus_doctor": ("friend", "doctor"), "doctor_minus_bare": ("doctor", "bare"),
                               "friend_minus_bare": ("friend", "bare")}.items():
        summary["projection"][comparison] = {}
        for framing in ["positive", "negative", "all"]:
            sel = [p for p in pairs if framing == "all" or p[1].endswith(framing)]
            diffs = torch.stack([states[(*p, a)] - states[(*p, b)] for p in sel])          # [n, layers, d]
            verdict = np.array([by_key[(*p, a)]["readout_d"] - by_key[(*p, b)]["readout_d"] for p in sel])
            mean_diff = diffs.mean(0)                                                     # [layers, d]
            out = {"n": len(sel), "verdict_change_median_logodds": float(np.median(verdict)), "by_direction": {}}
            for d in DIRECTIONS:
                raw = (diffs * directions[d]).sum(-1).numpy()                              # [n, layers]
                sep_d = sep[d].numpy()
                proj = raw / sep_d                                                          # separation units
                proj[:, np.abs(sep_d) < 0.05 * np.abs(sep_d).max()] = np.nan               # layers where the separation is ~0
                corr = [float(np.corrcoef(proj[:, l], verdict)[0, 1]) if np.std(proj[:, l]) > 0 and np.std(verdict) > 0 else float("nan")
                        for l in range(num_layers)]
                out["by_direction"][d] = {
                    "mean_raw": raw.mean(0).tolist(),
                    "mean_sep_units": proj.mean(0).tolist(),
                    "sem_sep_units": (proj.std(0, ddof=1) / math.sqrt(len(sel))).tolist() if len(sel) > 1 else [0.0] * num_layers,
                    "cosine_mean_diff": torch.nn.functional.cosine_similarity(mean_diff, directions[d], dim=-1).tolist(),
                    "corr_with_verdict_change": corr,
                }
            rand_cos = torch.nn.functional.cosine_similarity(mean_diff.unsqueeze(0), random_dirs.unsqueeze(1), dim=-1).abs()  # [num_random, layers]
            out["random_cosine_abs_p95"] = torch.quantile(rand_cos, 0.95, dim=0).tolist()
            summary["projection"][comparison][framing] = out

    # 3. Steering the doctor prompt on held-out reviews
    steer_records = []
    steer_dirs = {d: directions[d] for d in DIRECTIONS}
    for i in range(args.num_random_steer):
        steer_dirs[f"random{i}"] = random_dirs[i].expand(num_layers, -1)
    steer_path = os.path.join(args.output_dir, "steering_records.jsonl")
    with open(steer_path, "w", encoding="utf-8") as f:
        for ex in tqdm(test_reviews, desc="steering"):
            rct = format_rct_inputs(ex["Inputs"])
            for pair in get_prompt_pairs(ex, "doctor", None):
                ids = encode(tokenizer, SOURCES[args.steer_source].format(claim=pair["target_question"]), rct, model.device)
                base = by_key[(ex["ReviewID"], pair["question_key"], args.steer_source)]["readout_d"]
                for name, dvec in steer_dirs.items():
                    scale_dir = "risk" if name.startswith("random") else name
                    for layer in args.steer_layers:
                        for alpha in args.alphas:
                            vec = alpha * abs(sep[scale_dir][layer].item()) * dvec[layer]
                            with add_to_residual(model, layer, [ids.shape[1] - 1], vec):
                                d = readout(model, ids, yes_ids, no_ids)["d"]
                            rec = {"review_id": ex["ReviewID"], "question_key": pair["question_key"],
                                   "framing": pair["question_key"].rsplit("_", 1)[1], "direction": name,
                                   "layer": layer, "alpha": alpha, "base_d": base, "steered_d": d}
                            steer_records.append(rec)
                            f.write(json.dumps(rec) + "\n")
            gc.collect()
            torch.cuda.empty_cache()

    sig = lambda x: 1 / (1 + math.exp(-x))
    for framing in ["positive", "negative"]:
        summary["steering"][framing] = {}
        for name in steer_dirs:
            for layer in args.steer_layers:
                for alpha in args.alphas:
                    sel = [r for r in steer_records if r["framing"] == framing and r["direction"] == name and r["layer"] == layer and r["alpha"] == alpha]
                    if not sel:
                        continue
                    dd = np.array([r["steered_d"] - r["base_d"] for r in sel])
                    dp = np.array([sig(r["steered_d"]) - sig(r["base_d"]) for r in sel])
                    summary["steering"][framing][f"{name}|L{layer}|a{alpha}"] = {
                        "n": len(sel), "mean_delta_logodds": float(dd.mean()), "median_delta_p_yes": float(np.median(dp)),
                        "mean_delta_p_yes": float(dp.mean()), "frac_p_yes_up_0.1": float((dp > 0.1).mean()),
                    }

    with open(os.path.join(args.output_dir, "risk_direction_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print_summary(summary, args)
    plot(summary, args, num_layers)


def print_summary(summary, args):
    print(f"\ncos(risk, valence) by layer: " + " ".join(f"{c:+.2f}" for c in summary["cos_risk_valence_by_layer"][::4]))
    for comparison, by_framing in summary["projection"].items():
        for framing, out in by_framing.items():
            print(f"\n[{comparison} | {framing} claims, n={out['n']}] verdict change (median log-odds) {out['verdict_change_median_logodds']:+.2f}")
            for d, v in out["by_direction"].items():
                print(f"  {d:<10} sep units: " + " ".join(f"L{l}:{v['mean_sep_units'][l]:+.2f}" for l in range(0, len(v['mean_sep_units']), 4))
                      + f" | corr w/ verdict @L15: {v['corr_with_verdict_change'][15]:+.2f}")
    print(f"\nSteering the '{args.steer_source}' prompt (mean change in log-odds of Yes):")
    for framing, by_cond in summary["steering"].items():
        print(f"  [{framing} claims]")
        for cond, s in by_cond.items():
            print(f"    {cond:<22} {s['mean_delta_logodds']:+.2f}  (median dP(Yes) {s['median_delta_p_yes']:+.3f}, P(Yes) up >0.1 in {100*s['frac_p_yes_up_0.1']:.0f}%)")


def plot(summary, args, num_layers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"risk": "#e34948", "valence": "#2a78d6", "risk_orth": "#eda100"}
    layers = list(range(num_layers))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for ax, framing in zip(axes[:2], ["positive", "negative"]):
        out = summary["projection"]["friend_minus_doctor"][framing]
        for d, v in out["by_direction"].items():
            m, s = np.array(v["mean_sep_units"]), np.array(v["sem_sep_units"])
            ax.plot(layers, m, color=colors[d], lw=2, label=d)
            ax.fill_between(layers, m - s, m + s, color=colors[d], alpha=0.15, lw=0)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title(f"Friend − doctor along each direction\n{framing} claims (n={out['n']}; 1 = dangerous vs safe gap)")
        ax.set_xlabel("Layer"); ax.legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("Projection difference (separation units)")
    ax = axes[2]
    neg = summary["steering"]["negative"]
    for name in ["risk", "risk_orth", "valence"] + [f"random{i}" for i in range(args.num_random_steer)]:
        for layer in args.steer_layers[1:2] if len(args.steer_layers) > 1 else args.steer_layers:
            xs = [a for a in args.alphas if f"{name}|L{layer}|a{a}" in neg]
            ys = [neg[f"{name}|L{layer}|a{a}"]["mean_delta_logodds"] for a in xs]
            ax.plot(xs, ys, marker="o", lw=2 if not name.startswith("random") else 1,
                    color=colors.get(name, "#9a9fa9"), label=name if not name.startswith("random") or name == "random0" else None)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title(f"Steering '{args.steer_source}' prompt, negative claims\nlayer {args.steer_layers[1] if len(args.steer_layers) > 1 else args.steer_layers[0]}")
    ax.set_xlabel("alpha (× dangerous–safe separation)"); ax.set_ylabel("Mean change in log-odds of Yes")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    path = os.path.join(args.output_dir, "risk_direction.png")
    fig.savefig(path, dpi=180)
    print(f"Saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Risk direction: projection and steering tests for the doctor/friend source effect")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="HuggingFace model name.")
    parser.add_argument("--input_path", required=True, help="dataset with RCT inputs and framed questions (extracted question set).")
    parser.add_argument("--output_dir", required=True, help="directory to save the outputs.")
    parser.add_argument("--num_reviews", default=30, type=int, help="reviews used (first half fits the directions, second half is held out).")
    parser.add_argument("--sample_pool", default=30, type=int, help="reviews are the first --num_reviews of the seeded sample of this size used by the main sweeps.")
    parser.add_argument("--steer_source", default="doctor", choices=list(SOURCES), help="which source prompt to steer.")
    parser.add_argument("--steer_layers", nargs="+", default=[12, 15, 18], type=int, help="layers to steer at (last prompt token).")
    parser.add_argument("--alphas", nargs="+", default=[-2.0, -1.0, 1.0, 2.0], type=float, help="steering sizes, in units of the held-out separation along the direction.")
    parser.add_argument("--num_random", default=200, type=int, help="random directions for the cosine baseline.")
    parser.add_argument("--num_random_steer", default=2, type=int, help="random directions for the steering baseline.")
    parser.add_argument("--seed", default=SEED, type=int, help="seed for sampling reviews and random directions.")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, help="only 4 reviews (2 fit, 2 held out).")
    args = parser.parse_args()

    print()
    print("Arguments Provided for Risk Direction:")
    for k, v in vars(args).items():
        print(f"{k:<18} {v}")
    print()

    main(args)
