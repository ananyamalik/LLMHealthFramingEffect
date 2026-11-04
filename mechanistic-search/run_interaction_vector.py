"""
Does the model hold something about role x framing that is not in framing or role alone?

Prompt: prompts/question_only_yes_no.jinja2, as run_framing_role_mediation.py. D = log p(Yes) - log p(No) at the last
prompt token. h = the last prompt token's decoder layer output, for both framings x no role / "My <role> said,".

Interaction vector, per review and layer (for each role):
    I = h(negative, role) - h(negative, none) - h(positive, role) + h(positive, none)
    (= role shift under negative framing - role shift under positive framing; 0 if the two add up)

Tests (per role and layer, over reviews):
    size        |mean I| against a null from shuffled role labels: swapping role <-> none within a review flips the
                sign of its I, so the null is |mean(s_r * I_r)| for random signs s_r (--num_perms)
    answer      the yes/no answer direction a = (W_U[Yes] - W_U[No]) * final norm gain (logit lens, ignoring the
                1/rms scale). cos(mean I, a) against the same sign-flip null, and the per-review projection I_r . a/|a|
                (t-test against 0). At the last layer this should track the behavioral interaction in D.
    outside     mean I with the framing and role directions projected out (both fit on the same reviews: mean of
                h(positive, none) - h(negative, none), and of h(f, role) - h(f, none) over f): the fraction of |mean I|
                left, and its projection on a. What is left is not along framing or role alone.

Outputs (--output_dir):
    states.pt      {"states": [reviews, prompts, layers, d_model] float16, "prompts", "review_ids", "d": [reviews, prompts],
                   "answer_dir": [d_model]}; reused if present
    summary.csv    one row per role x layer
    interaction_vector.png
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from tqdm import tqdm

from patching_utils import DEFAULT_MODEL_NAME, SEED, set_global_seed, load_model_and_tokenizer, load_dataset
from run_role_patching import FRAMINGS, ROLES, YES_STRINGS, NO_STRINGS, get_claims, single_token_ids
from run_unit_steering import NO_ROLE, prompt_ids

ANSWER = "yes_no"
PROMPTS = [(f, r) for f in FRAMINGS for r in [NO_ROLE] + ROLES]


@torch.no_grad()
def cache_states(model, tokenizer, dataset: list[dict], yes_no_ids) -> dict:
    yes_ids, no_ids = yes_no_ids
    states, ds = [], []
    for example in tqdm(dataset, desc="caching"):
        claims = get_claims(example)
        per_prompt, d = [], []
        for framing, role in PROMPTS:
            ids = prompt_ids(tokenizer, claims[framing], role, model.device, ANSWER)
            out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
            per_prompt.append(torch.stack([h[0, -1] for h in out.hidden_states[1:]]).half().cpu())
            log_probs = torch.log_softmax(out.logits[0, -1].float(), -1)
            d.append(float(torch.logsumexp(log_probs[yes_ids], -1) - torch.logsumexp(log_probs[no_ids], -1)))
        states.append(torch.stack(per_prompt))
        ds.append(d)
    w_u = model.lm_head.weight.float()
    answer_dir = (w_u[yes_ids].mean(0) - w_u[no_ids].mean(0)) * model.model.norm.weight.float()
    return {"states": torch.stack(states), "prompts": [f"{f}|{r}" for f, r in PROMPTS],
            "review_ids": [e["ReviewID"] for e in dataset], "d": torch.tensor(ds), "answer_dir": answer_dir.cpu()}


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True)


def summarize(cache: dict, num_perms: int, seed: int) -> list[dict]:
    h, d = cache["states"], cache["d"]
    idx = {p: i for i, p in enumerate(cache["prompts"])}
    a = unit(cache["answer_dir"].float())
    num_reviews, _, num_layers, _ = h.shape
    signs = torch.from_numpy(np.random.default_rng(seed).choice([-1.0, 1.0], (num_perms, num_reviews))).float()

    def H(f, r, l):
        return h[:, idx[f"{f}|{r}"], l].float()

    rows = []
    for role in ROLES:
        d_inter = (d[:, idx[f"negative|{role}"]] - d[:, idx[f"negative|{NO_ROLE}"]]
                   - d[:, idx[f"positive|{role}"]] + d[:, idx[f"positive|{NO_ROLE}"]])
        for l in range(num_layers):
            inter = H("negative", role, l) - H("negative", NO_ROLE, l) - H("positive", role, l) + H("positive", NO_ROLE, l)
            mean = inter.mean(0)
            null = signs @ inter / num_reviews   # [perms, d_model]
            null_norm = null.norm(dim=-1)
            cos_a = float(unit(mean) @ a)
            null_cos = unit(null) @ a
            proj = inter @ a

            framing = unit((H("positive", NO_ROLE, l) - H("negative", NO_ROLE, l)).mean(0))
            role_dir = unit(((H("positive", role, l) - H("positive", NO_ROLE, l))
                             + (H("negative", role, l) - H("negative", NO_ROLE, l))).mean(0) / 2)
            q, _ = torch.linalg.qr(torch.stack([framing, role_dir], 1))
            outside = mean - q @ (q.T @ mean)

            rows.append({"role": role, "layer": l, "n_reviews": num_reviews,
                         "norm": float(mean.norm()), "null_norm_mean": float(null_norm.mean()),
                         "null_norm_95": float(null_norm.quantile(0.95)),
                         "p_size": float((null_norm >= mean.norm()).float().mean()),
                         "cos_answer": cos_a, "p_cos_answer": float((null_cos.abs() >= abs(cos_a)).float().mean()),
                         "answer_proj_mean": float(proj.mean()), "answer_proj_sem": float(stats.sem(proj.numpy())),
                         "p_answer_proj": float(stats.ttest_1samp(proj.numpy(), 0).pvalue),
                         "cos_framing": float(unit(mean) @ framing), "cos_role": float(unit(mean) @ role_dir),
                         "frac_outside": float(outside.norm() / mean.norm()),
                         "outside_answer_proj": float(outside @ a),
                         "d_interaction": float(d_inter.mean())})
    return rows


def plot(rows: list[dict], path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for role in ROLES:
        r = [x for x in rows if x["role"] == role]
        layers = [x["layer"] for x in r]
        line, = axes[0].plot(layers, [x["norm"] for x in r], marker="o", ms=3, label=role)
        axes[0].plot(layers, [x["null_norm_95"] for x in r], ls="--", color=line.get_color(), lw=1)
        axes[1].plot(layers, [x["cos_answer"] for x in r], marker="o", ms=3, label=role)
        axes[2].plot(layers, [x["frac_outside"] for x in r], marker="o", ms=3, label=role)
    axes[0].set_title("|mean interaction vector|\n(dashed: 95th pct of shuffled-label null)")
    axes[1].set_title("cos(mean interaction, yes/no answer direction)")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[2].set_title("fraction of |mean interaction| outside\nthe framing + role directions")
    for ax in axes:
        ax.set_xlabel("layer")
        ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main(args):
    set_global_seed(SEED)
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "states.pt")
    if os.path.exists(path):
        cache = torch.load(path)
        print(f"Loaded {path}")
    else:
        dataset = load_dataset(args.input_path)
        if args.num_reviews > 0:
            dataset = dataset[:args.num_reviews]
        model, tokenizer = load_model_and_tokenizer(args.model)
        yes_no_ids = (single_token_ids(tokenizer, YES_STRINGS), single_token_ids(tokenizer, NO_STRINGS))
        cache = cache_states(model, tokenizer, dataset, yes_no_ids)
        torch.save(cache, path)
        print(f"Saved {path}")

    rows = summarize(cache, args.num_perms, args.seed)
    with open(os.path.join(args.output_dir, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, os.path.join(args.output_dir, "interaction_vector.png"))
    for row in rows:
        if row["layer"] % 4 == 0 or row["layer"] == cache["states"].shape[2] - 1:
            print(f"{row['role']:<7} L{row['layer']:<3} |I| {row['norm']:7.2f} (null 95% {row['null_norm_95']:6.2f}, "
                  f"p {row['p_size']:.3f})  cos answer {row['cos_answer']:+.3f} (p {row['p_cos_answer']:.3f})  "
                  f"outside {row['frac_outside']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Role x framing interaction vector in the residual stream")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--input_path", required=True, help="dataset with the framed questions (extracted question set).")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_reviews", default=-1, type=int, help="first n reviews (-1: all).")
    parser.add_argument("--num_perms", default=2000, type=int, help="sign flips for the shuffled-label null.")
    parser.add_argument("--seed", default=SEED, type=int)
    args = parser.parse_args()
    print("\nArguments:")
    for k, v in vars(args).items():
        print(f"{k:<14} {v}")
    print()
    main(args)
