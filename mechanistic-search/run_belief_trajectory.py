"""
Belief trajectory: how the belief probe's reading of the residual stream evolves token by token, through the
question-only full-answer prompt and the whole generated answer.

Probe: one ridge regression of D = log p(Yes) - log p(No) (the yes/no prompt's "Does the treatment in the question
work?", see run_belief_probe.py) per layer, fit on the residuals at --train_anchors pooled (so it is not tied to one
position), with the penalty chosen by inner CV grouped by review. Training data: <probe_dir>/activations.pt from
run_belief_probe.py if it exists, else the anchors of --probe_reviews other reviews collected here. The trajectory
reviews are never in the training data. Scores are in units of D (nats; > 0: reads as "the treatment works").

Trajectories: for --num_reviews reviews x both framings x (bare claim + --roles), the full answer is generated
greedily (prompts/question_only.jinja2, as in run_role_patching.py), then prompt + answer are run once (teacher
forced, so the residuals are those of generation) and every token's residual at every layer is projected onto the
probe. Also per question:
    d_before  D of the yes/no prompt (the probe's label)
    d_after   D when the yes/no question is asked as a follow-up turn after the model's own answer

Outputs (in --output_dir):
    probe.pt                    per layer: weight [layers, d_model], bias, mean, std; alpha; training rows
    trajectories.jsonl          one line per question: question, answer, d_before, d_after, token positions of the
                                role, claim, anchors and answer start, file
    trajectories/<key>.npz      score [layers, tokens], norm [layers, tokens] (residual L2 norm), tokens
    trajectory_<review>.png     per review: score at --plot_layer over tokens (prompt then answer, aligned at the
                                answer start), per framing, lines = roles; and score per layer x token for each question
                                (token 0, an attention sink with a huge norm, is left out of the plots, not the data)
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from patching_utils import (
    DEFAULT_MODEL_NAME, SEED, render_prompt, set_global_seed, load_model_and_tokenizer, get_decoder_layers, load_dataset,
    sample_reviews, cache_residuals, greedy_generate,
)
from run_role_patching import FRAMINGS, ROLES, YES_STRINGS, NO_STRINGS, role_prefix, get_claims, encode_prompt, single_token_ids
from run_belief_probe import (
    ANCHORS, NO_ROLE, FRAMING_COLORS, ROLE_COLORS, anchor_chars, collect, grouped_folds, ix, standardize, ridge_predict,
)

YES_NO_FOLLOW_UP = 'Does the treatment in the question work? Answer with only "Yes" or "No".'   # as in question_only_yes_no


# Probe
def fit_probe(acts: dict[str, torch.Tensor], rows: list[dict], train_anchors: list[str], alphas: list[float],
              inner_folds: int, seed: int, device) -> dict:
    """Primal ridge weights per layer on the pooled anchors, with alpha chosen per layer by grouped CV."""
    x_all = torch.cat([acts[a] for a in train_anchors])                     # [rows * anchors, layers, d]
    y = torch.tensor([r["d"] for r in rows] * len(train_anchors), dtype=torch.float64, device=device)[:, None]
    groups = np.array([r["review_id"] for r in rows] * len(train_anchors))
    probe = {"weight": [], "bias": [], "mean": [], "std": [], "alpha": []}
    for layer in tqdm(range(x_all.shape[1]), desc="probe"):
        x = x_all[:, layer].to(device, torch.float64)
        sq_err = torch.zeros(len(alphas), dtype=torch.float64, device=device)
        for val in grouped_folds(groups, inner_folds, seed):
            fit = np.setdiff1d(np.arange(len(groups)), val)
            xf, xv = standardize(x[ix(fit)], x[ix(val)])
            sq_err += ((ridge_predict(xf, y[ix(fit)], xv, alphas) - y[ix(val)]) ** 2).sum((1, 2))
        alpha = alphas[int(sq_err.argmin())]
        mean, std = x.mean(0), x.std(0) + 1e-6
        xs = (x - mean) / std
        eye = torch.eye(len(xs), dtype=xs.dtype, device=device)
        dual = torch.cholesky_solve(y - y.mean(), torch.linalg.cholesky(xs @ xs.T + alpha * xs.shape[1] * eye))
        for k, v in zip(probe, [(xs.T @ dual)[:, 0], y.mean(), mean, std, alpha]):
            probe[k].append(v.float().cpu() if torch.is_tensor(v) else v)
    return {k: torch.stack(v) if k != "alpha" else v for k, v in probe.items()}


def training_data(args, model, tokenizer, dataset: list[dict], exclude: set[str]) -> tuple[list[dict], dict]:
    acts_path = os.path.join(args.probe_dir, "activations.pt")
    if os.path.exists(acts_path):
        saved = torch.load(acts_path)
        rows, acts = saved["rows"], saved["acts"]
        print(f"Probe training data: {acts_path}")
    else:
        others = sample_reviews([e for e in dataset if e["ReviewID"] not in exclude], args.probe_reviews, seed=args.seed)
        print(f"No {acts_path}: collecting the anchors of {len(others)} other reviews for the probe")
        rows, acts = collect(model, tokenizer, others, args.roles, ANCHORS)
    keep = [i for i, r in enumerate(rows) if r["review_id"] not in exclude]
    missing = set(args.train_anchors) - set(acts)
    if missing:
        raise ValueError(f"--train_anchors {sorted(missing)} are not in the training activations ({sorted(acts)})")
    return [rows[i] for i in keep], {a: acts[a][ix(np.array(keep))] for a in args.train_anchors}


# Trajectories
def yes_no_d(model, ids: torch.Tensor, yes_ids, no_ids) -> float:
    hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[:, -1]
    log_probs = torch.log_softmax(model.lm_head(hidden).float(), dim=-1)[0]
    return (torch.logsumexp(log_probs[yes_ids], 0) - torch.logsumexp(log_probs[no_ids], 0)).item()


def chat_ids(tokenizer, messages: list[dict], device) -> torch.Tensor:
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if not isinstance(ids, list):  # newer transformers can return a BatchEncoding
        ids = ids["input_ids"]
    return torch.tensor([ids], device=device)


@torch.no_grad()
def trajectory(model, tokenizer, probe: dict, question: str, claim: str, args, yes_ids, no_ids) -> dict:
    full_ids, encoding, question_start = encode_prompt(tokenizer, "full", question, [], model.device, "question_only")
    prompt_len = full_ids.shape[1]
    answer_ids = greedy_generate(model, tokenizer, full_ids, args.max_new_tokens)
    ids = torch.cat([full_ids, answer_ids[None]], 1)

    num_layers = len(get_decoder_layers(model))
    with cache_residuals(model, list(range(ids.shape[1]))) as cache:
        model.model(input_ids=ids, use_cache=False)
    resid = torch.stack([cache[l].float().cpu() for l in range(num_layers)])   # [layers, tokens, d]
    score = torch.einsum("ltd,ld->lt", (resid - probe["mean"][:, None]) / probe["std"][:, None], probe["weight"]) + probe["bias"][:, None]

    claim_start = question_start + len(question) - len(claim)
    positions = {a: encoding.char_to_token(claim_start + c) for a, c in anchor_chars(claim).items()}
    positions |= {"last": prompt_len - 1, "claim_start": encoding.char_to_token(claim_start), "answer_start": prompt_len}
    if question != claim:
        positions["role_start"] = encoding.char_to_token(question_start)

    answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
    d_before = yes_no_d(model, encode_prompt(tokenizer, "yes_no", question, [], model.device, "question_only")[0], yes_ids, no_ids)
    first_turn = render_prompt("question_only", template_dir="../mechanistic-search/prompts", question=question, abstracts=[])
    follow_up = chat_ids(tokenizer, [{"role": "user", "content": first_turn}, {"role": "assistant", "content": answer},
                                     {"role": "user", "content": YES_NO_FOLLOW_UP}], model.device)
    d_after = yes_no_d(model, follow_up, yes_ids, no_ids)
    return {
        "answer": answer, "d_before": d_before, "d_after": d_after, "positions": positions,
        "num_tokens": ids.shape[1], "prompt_tokens": prompt_len,
        "score": np.array(score.tolist(), dtype=np.float32),
        "norm": np.array(resid.norm(dim=-1).tolist(), dtype=np.float32),
        "tokens": [tokenizer.decode([t]) for t in ids[0].tolist()],
    }


def plot_review(review_id: str, records: list[dict], data: dict, roles: list[str], layer: int, path: str) -> None:
    """Token 0 (the first token, an attention sink with a huge residual norm) is left out of the plots."""
    keys = [(f, r) for f in FRAMINGS for r in [NO_ROLE] + roles]
    n_rows = (len(keys) + 1) // 2
    fig = plt.figure(figsize=(16, 4.5 + 2.4 * n_rows), constrained_layout=True)
    grid = fig.add_gridspec(1 + n_rows, 2, height_ratios=[1.8] + [1] * n_rows)
    by_key = {(rec["framing"], rec["role"]): rec for rec in records}
    vmax = float(np.quantile(np.concatenate([np.abs(data[rec["key"]]["score"][:, 1:]).ravel() for rec in records]), 0.99))

    for col, framing in enumerate(FRAMINGS):
        ax = fig.add_subplot(grid[0, col])
        for i, role in enumerate([NO_ROLE] + roles):
            rec = by_key.get((framing, role))
            if rec is None:
                continue
            x = np.arange(1, rec["num_tokens"]) - rec["positions"]["answer_start"]   # 0 = first answer token
            ax.plot(x, data[rec["key"]]["score"][layer, 1:], lw=1, color=ROLE_COLORS[i % len(ROLE_COLORS)],
                    label=f"{role} (D before {rec['d_before']:+.1f}, after answer {rec['d_after']:+.1f})")
        ax.axvline(-0.5, color="gray", lw=0.8, ls="--")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set(title=f"{review_id}, {framing} claim: probe score at layer {layer}",
               xlabel="token (0 = first answer token; < 0 prompt)", ylabel="probe score (D units; > 0: works)")
        ax.legend(fontsize=7)

    for k, (framing, role) in enumerate(keys):
        rec = by_key.get((framing, role))
        if rec is None:
            continue
        ax = fig.add_subplot(grid[1 + k % n_rows, k // n_rows])
        start, score = rec["positions"]["answer_start"], data[rec["key"]]["score"][:, 1:]
        im = ax.imshow(score, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax, origin="lower", interpolation="nearest",
                       extent=(0.5 - start, rec["num_tokens"] - 0.5 - start, -0.5, score.shape[0] - 0.5))
        for name in ("role_start", "claim_start", "answer_start"):
            if name in rec["positions"]:
                ax.axvline(rec["positions"][name] - start - 0.5, color="black", lw=0.6, ls="--" if name == "answer_start" else ":")
        ax.set_title(f"{framing} / {role}  (dotted: role, claim start; dashed: answer start)", fontsize=9)
        ax.set_ylabel("layer")
        ax.tick_params(labelsize=7)
    fig.colorbar(im, ax=fig.axes[2:], shrink=0.6, label="probe score (clipped at the 99th percentile)")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved {path}")


def main(args):
    set_global_seed(SEED)
    os.makedirs(os.path.join(args.output_dir, "trajectories"), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full_dataset = load_dataset(args.input_path)
    dataset = sample_reviews(full_dataset, args.num_reviews, seed=args.seed)
    review_ids = {e["ReviewID"] for e in dataset}
    print(f"Trajectory reviews: {sorted(review_ids)}")

    model, tokenizer = load_model_and_tokenizer(args.model)
    rows, acts = training_data(args, model, tokenizer, full_dataset, review_ids)
    print(f"Probe: {len(rows)} rows x anchors {args.train_anchors}")
    probe = fit_probe(acts, rows, args.train_anchors, args.alphas, args.inner_folds, args.seed, device)
    torch.save(probe | {"train_anchors": args.train_anchors, "n_rows": len(rows), "model": args.model},
               os.path.join(args.output_dir, "probe.pt"))
    del acts
    num_layers = len(probe["alpha"])
    layer = args.plot_layer if args.plot_layer is not None else num_layers // 2

    yes_ids, no_ids = single_token_ids(tokenizer, YES_STRINGS), single_token_ids(tokenizer, NO_STRINGS)
    records, data = [], {}
    with open(os.path.join(args.output_dir, "trajectories.jsonl"), "w", encoding="utf-8") as f:
        for example in tqdm(dataset, desc="reviews"):
            for framing, claim in get_claims(example).items():
                for role in [NO_ROLE] + args.roles:
                    question = claim if role == NO_ROLE else f"{role_prefix(role)} {claim}"
                    key = f"{example['ReviewID']}_{framing}_{role}"
                    out = trajectory(model, tokenizer, probe, question, claim, args, yes_ids, no_ids)
                    np.savez_compressed(os.path.join(args.output_dir, "trajectories", f"{key}.npz"),
                                        score=out["score"], norm=out["norm"], tokens=np.array(out["tokens"]))
                    data[key] = out
                    record = {"key": key, "review_id": example["ReviewID"], "framing": framing, "role": role,
                              "question": question, "file": f"trajectories/{key}.npz"}
                    record |= {k: out[k] for k in ["answer", "d_before", "d_after", "positions", "num_tokens", "prompt_tokens"]}
                    records.append(record)
                    f.write(json.dumps(record) + "\n")
                    f.flush()

    for review_id in sorted(review_ids):
        plot_review(review_id, [r for r in records if r["review_id"] == review_id], data, args.roles, layer,
                    os.path.join(args.output_dir, f"trajectory_{review_id}.png"))
    print("\nD before -> after the answer, and the probe at layer", layer, "(last prompt token -> last answer token):")
    for r in records:
        s = data[r["key"]]["score"][layer]
        print(f"  {r['key']:<40} D {r['d_before']:+6.2f} -> {r['d_after']:+6.2f} | probe {s[r['prompt_tokens'] - 1]:+6.2f} -> {s[-1]:+6.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Belief probe read out at every token of the prompt and the generated answer")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="HuggingFace model name.")
    parser.add_argument("--input_path", required=True, help="dataset with the framed questions (extracted question set).")
    parser.add_argument("--output_dir", required=True, help="where the probe, trajectories and plots go.")
    parser.add_argument("--probe_dir", required=True, help="run_belief_probe.py output dir; its activations.pt is the probe's training data if it exists.")
    parser.add_argument("--roles", nargs="*", default=ROLES, help="roles for 'My <role> said, <claim>' (the bare claim is always included).")
    parser.add_argument("--num_reviews", default=5, type=int, help="seeded sample of reviews to trace.")
    parser.add_argument("--probe_reviews", default=100, type=int, help="other reviews collected for the probe when <probe_dir>/activations.pt does not exist.")
    parser.add_argument("--train_anchors", nargs="+", default=ANCHORS, choices=ANCHORS, help="anchors pooled to train the probe.")
    parser.add_argument("--alphas", nargs="+", type=float, default=[1e-3, 1e-2, 1e-1, 1, 10], help="ridge penalties (times d_model).")
    parser.add_argument("--inner_folds", default=4, type=int, help="CV folds (grouped by review) for the penalty.")
    parser.add_argument("--max_new_tokens", default=512, type=int, help="max tokens for the greedy answers.")
    parser.add_argument("--plot_layer", default=None, type=int, help="layer of the line plots (default: the middle layer).")
    parser.add_argument("--seed", default=SEED, type=int, help="seed for sampling reviews and folds.")
    args = parser.parse_args()

    print("\nArguments:")
    for k, v in vars(args).items():
        print(f"{k:<16} {v}")
    print()
    main(args)
