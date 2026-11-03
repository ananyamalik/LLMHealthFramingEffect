"""
Steering the bare claim with the diffmeans framing and role directions, separately and together.

Prompt: the bare claim with no role ("<claim>"), in the question-only prompts, for the held-out test reviews of
get_diffmeans_directions.py (the directions never saw them), both framings. --answer picks the prompt and the score:
    yes_no  prompts/question_only_yes_no.jinja2. D = log p(Yes) - log p(No) at the last prompt token
    full    prompts/question_only.jinja2 ("Question: <claim>"). The model writes its full answer (greedy, with the
            steering), then the yes/no question is asked as a follow-up turn without steering, as d_after in
            run_belief_trajectory.py: D = log p(Yes) - log p(No) there, i.e. what the written answer concluded.
            The answers are saved. Every condition is a generation, so the defaults are a smaller grid (alphas +-1,
            layers 8 16 24, 20 reviews)
    (D > 0: the model says the treatment works.)

Directions (<diffmeans_dir>/<anchor>/directions.pt; default anchor: the last token of the same prompt, i.e. yes_no
for yes_no and last for full):
    framing   positive - negative claim, averaged over roles
    role      --role_direction:
              shared               (default) mean over doctor/friend/enemy of (My <role> said, - no role):
                                   + pushes toward "someone said it", - away from it
              doctor_minus_enemy   the main axis between roles: + toward "my doctor said", - toward "my enemy said"
    random_*  a fixed random direction per layer with the norm of framing / role (controls)
alpha = 1 adds the full average difference (e.g. negative claim + 1 x framing ~ the positive claim's state).

Each vector is added to the residual stream (decoder layer output) at one layer, at --positions:
    last      the last prompt token only (where the answer starts)
    question  every token from the start of the question to the end of the prompt
The vector is added only while the prompt is read; generated tokens see it through the cached keys/values.
With --steer_generation (full), it is also added at every generated token, so the push lasts the whole answer.

Conditions per test (review, framing) and layer:
    framing|a        alpha * framing                       does framing steering move the answer like the real claim?
    role|a           alpha * role                          does role steering change how framing is used? (the
                                                           framing effect D(pos) - D(neg) under the same role push)
    both|af|ar       af * framing + ar * role              does the role push change what the framing push does?
    random_framing|a, random_role|a                        controls
Reference: the clean D of the real role prompts ("My doctor/friend/enemy said, <claim>") of the same reviews.

Outputs (in --output_dir/<answer>/<positions>[_gen][_role-shared]/; no suffix = doctor_minus_enemy, as the first runs):
    pairs.jsonl     one line per test (review, framing): clean D, real-role D, steered D per condition and layer
                    (full: also answers, clean_answer, role_answers[role], steered_answers[condition][layer]);
                    resumable
    summary.json    per layer: framing recovery, framing effect under role steering, framing steering effect under
                    role steering, random controls, and the real-role framing effects
    steering.png    the same as four panels
"""
import argparse
import gc
import json
import os
from contextlib import contextmanager, nullcontext

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from patching_utils import (
    DEFAULT_MODEL_NAME, SEED, render_prompt, set_global_seed, load_model_and_tokenizer, get_decoder_layers, load_dataset,
    yes_no_logit_diff, greedy_generate,
)
from run_role_patching import FRAMINGS, ROLES, YES_STRINGS, NO_STRINGS, role_prefix, get_claims, encode_prompt, single_token_ids
from run_belief_trajectory import YES_NO_FOLLOW_UP, chat_ids, yes_no_d

ANSWERS = ["yes_no", "full"]
DEFAULT_ANCHOR = {"yes_no": "yes_no", "full": "last"}   # the last token of the steered prompt
# Defaults per answer type (full answers are generations, so a smaller grid)
DEFAULTS = {
    "yes_no": {"layers": None, "framing_alphas": [-2, -1, -0.5, 0.5, 1, 2], "role_alphas": [-2, -1, -0.5, 0.5, 1, 2],
               "random_alphas": [-2, 2], "num_reviews": -1},
    "full": {"layers": [8, 16, 24], "framing_alphas": [-1, 1], "role_alphas": [-1, 1], "random_alphas": [-1, 1],
             "num_reviews": 20},
}

POSITIONS = ["last", "question"]
# role direction -> what its + and - ends push toward (plot labels)
ROLE_DIRECTIONS = {"shared": ("has a role", "no role"), "doctor_minus_enemy": ("doctor", "enemy")}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ROLE_COLORS = {"none": "#8f8e89", "doctor": "#2a78d6", "friend": "#eb6834", "enemy": "#1baf7a"}


def strength(a: float) -> dict:
    """Line style by |alpha|: thicker and darker for a bigger push."""
    return {"lw": {0.5: 1.2, 1.0: 2.0, 2.0: 3.0}.get(abs(a), 4.0), "alpha": {0.5: 0.55, 1.0: 0.8}.get(abs(a), 1.0)}


def real_lines(ax, values: dict) -> None:
    """Dotted reference lines for the real role prompts, named in the legend."""
    for role, v in values.items():
        ax.axhline(v, color=ROLE_COLORS[role], lw=1.2, ls=":", label=f"real '{role}' prompt" if role != "none" else "real bare claim")


@contextmanager
def add_at_positions(model, layer_idx: int, positions: list[int], vector: torch.Tensor, during_generation: bool = False):
    """Adds `vector` to the output of decoder layer `layer_idx` at `positions` (and at every generated token if
    `during_generation`)."""
    def hook(module, inputs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        vec = vector.to(device=hidden_states.device, dtype=hidden_states.dtype)
        if hidden_states.shape[1] > max(positions):   # the prompt pass
            hidden_states[0, positions, :] += vec
        elif during_generation:                       # a decode step: the newest token
            hidden_states[0, -1, :] += vec
        return output

    handle = get_decoder_layers(model)[layer_idx].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def conditions(args) -> list[tuple[str, float, float]]:
    """(name, framing alpha, role alpha); random controls are listed with their alpha as the framing / role alpha."""
    conds = [(f"framing|{a}", a, 0.0) for a in args.framing_alphas]
    conds += [(f"role|{a}", 0.0, a) for a in args.role_alphas]
    conds += [(f"both|{af}|{ar}", af, ar) for af in args.both_alphas for ar in args.both_alphas]
    conds += [(f"random_framing|{a}", a, 0.0) for a in args.random_alphas]
    conds += [(f"random_role|{a}", 0.0, a) for a in args.random_alphas]
    return conds


def steering_vectors(directions: dict, role_direction: str, seed: int) -> dict[str, torch.Tensor]:
    framing, role = directions["framing"], directions[role_direction]
    generator = torch.Generator().manual_seed(seed)
    out = {"framing": framing, "role": role}
    for name, v in [("random_framing", framing), ("random_role", role)]:
        r = torch.randn(v.shape, generator=generator)
        out[name] = r / r.norm(dim=-1, keepdim=True) * v.norm(dim=-1, keepdim=True)
    return out


@torch.no_grad()
def score(model, tokenizer, question: str, ids: torch.Tensor, args, yes_no_ids, steer=None) -> tuple[float, str | None]:
    """(D, full answer or None) of the prompt `ids` for `question`, with the steering context `steer` if given."""
    with steer if steer is not None else nullcontext():
        if args.answer == "yes_no":
            return yes_no_logit_diff(model, ids, *yes_no_ids)[0], None
        answer = tokenizer.decode(greedy_generate(model, tokenizer, ids, args.max_new_tokens), skip_special_tokens=True).strip()
    # The follow-up is read without steering: D is what the written answer concluded
    first_turn = render_prompt("question_only", template_dir="../mechanistic-search/prompts", question=question, abstracts=[])
    follow_up = chat_ids(tokenizer, [{"role": "user", "content": first_turn}, {"role": "assistant", "content": answer},
                                     {"role": "user", "content": YES_NO_FOLLOW_UP}], model.device)
    return yes_no_d(model, follow_up, *yes_no_ids), answer


def run_pair(model, tokenizer, claim: str, ids: torch.Tensor, positions: list[int], vectors: dict, conds, layers, args, yes_no_ids) -> dict:
    steered, answers = {}, {}
    for name, af, ar in conds:
        kind = name.split("|")[0]
        f_vec = vectors["random_framing" if kind == "random_framing" else "framing"]
        r_vec = vectors["random_role" if kind == "random_role" else "role"]
        steered[name], answers[name] = {}, {}
        for l in layers:
            vector = af * f_vec[l] + ar * r_vec[l]
            steer = add_at_positions(model, l, positions, vector, args.steer_generation)
            d, answer = score(model, tokenizer, claim, ids, args, yes_no_ids, steer)
            steered[name][str(l)] = d
            answers[name][str(l)] = answer
    return {"d_steered": steered} | ({"steered_answers": answers} if args.answer == "full" else {})


def summarize(records: list[dict], args, layers: list[int]) -> dict:
    by_review = {}
    for r in records:
        by_review.setdefault(r["review_id"], {})[r["framing"]] = r
    pairs = [v for v in by_review.values() if set(v) == set(FRAMINGS)]   # reviews with both framings done
    pos = [p["positive"] for p in pairs]
    neg = [p["negative"] for p in pairs]

    def mean(xs):
        return float(np.mean(xs)) if len(xs) else float("nan")

    def d(r, cond, l):
        return r["d_clean"] if cond is None else r["d_steered"][cond][str(l)]

    effect_clean = mean([p["d_clean"] - n["d_clean"] for p, n in zip(pos, neg)])
    real = {"none": effect_clean} | {role: mean([p["d_roles"][role] - n["d_roles"][role] for p, n in zip(pos, neg)])
                                    for role in ROLES}
    real_d = {"none": mean([r["d_clean"] for r in pos + neg])} | {role: mean([r["d_roles"][role] for r in pos + neg]) for role in ROLES}
    summary = {"n_reviews": len(pairs), "positions": args.positions, "anchor": args.anchor, "role_direction": args.role_direction,
               "layers": layers,
               "real_framing_effect": real, "real_mean_d": real_d,
               "framing_recovery": {}, "framing_effect_under_role": {}, "mean_d_under_role": {},
               "framing_steer_effect_under_role": {}, "random_abs_shift": {}}

    # Framing steering: how much of the real pos/neg difference does +/- alpha * framing reproduce?
    for a in args.framing_alphas:
        claims = neg if a > 0 else pos   # push the claim toward the other framing
        summary["framing_recovery"][str(a)] = [
            mean([d(r, f"framing|{a}", l) - r["d_clean"] for r in claims]) / effect_clean * np.sign(a) for l in layers]

    # Role steering: the framing effect D(pos) - D(neg) when both claims get the same role push
    for a in args.role_alphas:
        summary["framing_effect_under_role"][str(a)] = [
            mean([d(p, f"role|{a}", l) - d(n, f"role|{a}", l) for p, n in zip(pos, neg)]) for l in layers]
        summary["mean_d_under_role"][str(a)] = [mean([d(r, f"role|{a}", l) for r in pos + neg]) for l in layers]

    # Both: what +/- framing push does to D, at each role push (role push 0 = framing alone)
    hi, lo = max(args.both_alphas), min(args.both_alphas)
    for ar in sorted(set(args.both_alphas) | {0.0}):
        def cond(af):
            return f"framing|{af}" if ar == 0 else f"both|{af}|{ar}"
        if ar == 0 and not {hi, lo} <= set(args.framing_alphas):
            continue
        summary["framing_steer_effect_under_role"][str(ar)] = [
            mean([(d(r, cond(hi), l) - d(r, cond(lo), l)) / (hi - lo) for r in pos + neg]) for l in layers]

    for kind in ["random_framing", "random_role"]:
        summary["random_abs_shift"][kind] = {str(a): [mean([abs(d(r, f"{kind}|{a}", l) - r["d_clean"]) for r in pos + neg])
                                                     for l in layers] for a in args.random_alphas}

    # Full answers: the fraction of steered answers word-for-word the same as the unsteered one
    if args.answer == "full":
        summary["answer_unchanged"] = {cond: [mean([r["steered_answers"][cond][str(l)] == r["clean_answer"] for r in pos + neg])
                                              for l in layers] for cond in pos[0]["steered_answers"]} if pos else {}
    return summary


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=8)


def plot(summary: dict, title: str, path: str) -> None:
    layers = summary["layers"]
    plus, minus = ROLE_DIRECTIONS[summary.get("role_direction", "doctor_minus_enemy")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))

    ax = axes[0, 0]
    for a, values in summary["framing_recovery"].items():
        a = float(a)
        ax.plot(layers, values, marker="o", ms=4, color="#2a78d6" if a > 0 else "#eb6834", **strength(a),
                label=f"{'negative claim + ' if a > 0 else 'positive claim − '}{abs(a):g} × framing")
    for a, values in summary["random_abs_shift"]["random_framing"].items():
        ax.plot(layers, np.array(values) / summary["real_framing_effect"]["none"], color=INK_2, ls=":", lw=1.5,
                label=f"random direction, {float(a):+g} × framing size (|shift|)")
    ax.axhline(1, color=INK, lw=0.8)
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set(xlabel="layer steered", ylabel="fraction of the real framing difference")
    ax.set_title("Framing steering: 1 = answer moves as far as actually\nswitching 'will' ↔ 'won't'", fontsize=10, loc="left")
    ax.legend(fontsize=7, frameon=False)
    style(ax)

    ax = axes[0, 1]
    for a, values in summary["framing_effect_under_role"].items():
        a = float(a)
        ax.plot(layers, values, marker="o", ms=4, color=ROLE_COLORS["doctor" if a > 0 else "enemy"], **strength(a),
                label=f"{a:+g} × role (toward {plus if a > 0 else minus})")
    real_lines(ax, summary["real_framing_effect"])
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set(xlabel="layer steered", ylabel="framing effect  D(will) − D(won't)")
    ax.set_title("Role steering on the bare claim: does it change how much\nframing matters? (dotted = real role prompts)", fontsize=10, loc="left")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    style(ax)

    ax = axes[1, 0]
    for ar, values in summary["framing_steer_effect_under_role"].items():
        ar = float(ar)
        color = ROLE_COLORS["none"] if ar == 0 else ROLE_COLORS["doctor" if ar > 0 else "enemy"]
        ax.plot(layers, values, lw=2, marker="o", ms=4, color=color,
                label="framing push alone" if ar == 0 else f"framing push + {ar:+g} × role (toward {plus if ar > 0 else minus})")
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set(xlabel="layer steered (both vectors at the same layer)", ylabel="change in D per 1 × framing")
    ax.set_title("Both together: does the role push change what\nthe framing push does?", fontsize=10, loc="left")
    ax.legend(fontsize=7, frameon=False)
    style(ax)

    ax = axes[1, 1]
    for a, values in summary["mean_d_under_role"].items():
        a = float(a)
        ax.plot(layers, values, marker="o", ms=4, color=ROLE_COLORS["doctor" if a > 0 else "enemy"], **strength(a),
                label=f"{a:+g} × role")
    real_lines(ax, summary["real_mean_d"])
    ax.set(xlabel="layer steered", ylabel="mean D (both framings)")
    ax.set_title("Role steering: overall push toward Yes / No\n(dotted = real role prompts)", fontsize=10, loc="left")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    style(ax)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main(args):
    set_global_seed(SEED)
    fit = torch.load(os.path.join(args.diffmeans_dir, args.anchor, "directions.pt"))
    print(f"Directions: {args.diffmeans_dir}/{args.anchor} | answer: {args.answer}")
    if fit.get("model") and fit["model"] != args.model:
        raise ValueError(f"directions were fit on {fit['model']}, not {args.model}")
    test_reviews = set(fit["test_reviews"])
    vectors = steering_vectors(fit["directions"], args.role_direction, args.seed)
    dataset = [e for e in load_dataset(args.input_path) if e["ReviewID"] in test_reviews]
    if args.num_reviews > 0:
        dataset = dataset[:args.num_reviews]
    print(f"Test reviews: {len(dataset)} (of {len(test_reviews)} held out)")

    model, tokenizer = load_model_and_tokenizer(args.model)
    num_layers = len(get_decoder_layers(model))
    layers = args.layers if args.layers else sorted(set(range(0, num_layers, args.layer_stride)) | {num_layers - 1})
    yes_no_ids = (single_token_ids(tokenizer, YES_STRINGS), single_token_ids(tokenizer, NO_STRINGS))
    conds = conditions(args)
    print(f"Layers: {layers} | {len(conds)} conditions -> {len(conds) * len(layers)} steered passes per question")

    out_dir = os.path.join(args.output_dir, args.answer, args.positions + ("_gen" if args.steer_generation else "")
                           + ("" if args.role_direction == "doctor_minus_enemy" else f"_role-{args.role_direction}"))
    os.makedirs(out_dir, exist_ok=True)
    pairs_path = os.path.join(out_dir, "pairs.jsonl")
    done = set()
    if os.path.exists(pairs_path):
        with open(pairs_path, "r", encoding="utf-8") as f:
            done = {json.loads(line)["pair_key"] for line in f if line.strip()}
    print(f"{len(done)} questions already done")

    with open(pairs_path, "a", encoding="utf-8") as out_file:
        for example in tqdm(dataset, desc="reviews"):
            for framing, claim in get_claims(example).items():
                key = f"{example['ReviewID']}_{framing}"
                if key in done:
                    continue
                ids, encoding, question_start = encode_prompt(tokenizer, args.answer, claim, [], model.device, "question_only")
                positions = [ids.shape[1] - 1] if args.positions == "last" else list(range(encoding.char_to_token(question_start), ids.shape[1]))
                d_clean, clean_answer = score(model, tokenizer, claim, ids, args, yes_no_ids)
                d_roles, role_answers = {}, {}
                for role in ROLES:
                    question = f"{role_prefix(role)} {claim}"
                    role_ids = encode_prompt(tokenizer, args.answer, question, [], model.device, "question_only")[0]
                    d_roles[role], role_answers[role] = score(model, tokenizer, question, role_ids, args, yes_no_ids)
                record = {"pair_key": key, "review_id": example["ReviewID"], "framing": framing, "question": claim,
                          "answer": args.answer, "positions": positions, "d_clean": d_clean, "d_roles": d_roles}
                if args.answer == "full":
                    record |= {"clean_answer": clean_answer, "role_answers": role_answers}
                record |= run_pair(model, tokenizer, claim, ids, positions, vectors, conds, layers, args, yes_no_ids)
                out_file.write(json.dumps(record) + "\n")
                out_file.flush()
            gc.collect()
            torch.cuda.empty_cache()

    with open(pairs_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    summary = summarize(records, args, layers) | {"model": args.model}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{summary['n_reviews']} reviews | real framing effect: " + ", ".join(f"{k} {v:+.2f}" for k, v in summary["real_framing_effect"].items()))
    for a, v in summary["framing_recovery"].items():
        print(f"framing recovery, {a:>5} x framing: " + " ".join(f"L{l} {x:+.2f}" for l, x in zip(layers, v)))
    for a, v in summary["framing_effect_under_role"].items():
        print(f"framing effect,   {a:>5} x role:    " + " ".join(f"L{l} {x:+.2f}" for l, x in zip(layers, v)))
    for cond, v in summary.get("answer_unchanged", {}).items():
        print(f"answer unchanged, {cond:<20} " + " ".join(f"L{l} {x:.0%}" for l, x in zip(layers, v)))
    scored = "D = yes/no answer" if args.answer == "yes_no" else "D = yes/no follow-up after the written answer"
    added_at = f"{args.positions}, role = {args.role_direction}" + (" + every generated token" if args.steer_generation else "")
    plot(summary, f"{args.model} | steering the bare claim ({args.answer}; {scored}) | vectors from '{args.anchor}', added at: {added_at}",
         os.path.join(out_dir, "steering.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Steer the bare claim with the diffmeans framing and role directions")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="HuggingFace model name (must match the directions).")
    parser.add_argument("--input_path", required=True, help="dataset with the framed questions (extracted question set).")
    parser.add_argument("--diffmeans_dir", required=True, help="output dir of get_diffmeans_directions.py.")
    parser.add_argument("--output_dir", required=True, help="outputs go to <output_dir>/<answer>/<positions>/.")
    parser.add_argument("--answer", default="yes_no", choices=ANSWERS, help="yes_no (single-token answer) or full (written answer, scored by a yes/no follow-up).")
    parser.add_argument("--anchor", default=None, help="which anchor's directions to steer with (default: yes_no for yes_no, last for full).")
    parser.add_argument("--max_new_tokens", default=256, type=int, help="full only: max tokens of each written answer.")
    parser.add_argument("--positions", default="last", choices=POSITIONS, help="last prompt token only, or every token of the question.")
    parser.add_argument("--role_direction", default="shared", choices=list(ROLE_DIRECTIONS),
                        help="shared: mean of (My <role> said, - no role) over roles; doctor_minus_enemy: the earlier runs.")
    parser.add_argument("--steer_generation", action="store_true", help="full only: also add the vector at every generated token (outputs in <positions>_gen/).")
    parser.add_argument("--layers", nargs="*", type=int, default=None, help="layers to steer (default yes_no: every --layer_stride and the last; full: 8 16 24).")
    parser.add_argument("--layer_stride", default=4, type=int)
    parser.add_argument("--framing_alphas", nargs="+", type=float, default=None, help="default yes_no: -2 -1 -0.5 0.5 1 2; full: -1 1.")
    parser.add_argument("--role_alphas", nargs="+", type=float, default=None, help="default yes_no: -2 -1 -0.5 0.5 1 2; full: -1 1.")
    parser.add_argument("--both_alphas", nargs="+", type=float, default=[-1.0, 1.0], help="framing x role grid (the role-0 row is framing alone).")
    parser.add_argument("--random_alphas", nargs="+", type=float, default=None, help="default yes_no: -2 2; full: -1 1.")
    parser.add_argument("--num_reviews", default=None, type=int, help="first n test reviews (-1: all; default yes_no: all, full: 20).")
    parser.add_argument("--seed", default=SEED, type=int, help="seed for the random control directions.")
    args = parser.parse_args()
    args.anchor = args.anchor or DEFAULT_ANCHOR[args.answer]
    for k, v in DEFAULTS[args.answer].items():
        if getattr(args, k) is None:
            setattr(args, k, [float(x) for x in v] if k.endswith("alphas") else v)

    print("\nArguments:")
    for k, v in vars(args).items():
        print(f"{k:<16} {v}")
    print()
    main(args)
