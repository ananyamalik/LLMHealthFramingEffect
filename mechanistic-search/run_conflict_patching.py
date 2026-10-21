"""
Conflicting doctor claims: two "My doctor said" claims with opposite polarity in one question, Yes/No readout.

Question (same prompt as run_yes_no_patching.py, prompts/question_answering_yes_no.jinja2):
    "My doctor said <claim 1>, but my doctor also said <claim 2>. What do you think?"
where each claim is the positive or negative version of a claim-style question with its attribution and follow-up
removed, in two wordings per review:
    doctor: "X will work for Y"  / "X won't work for Y"
    ai:     "X works well for Y" / "X doesn't work for Y"

Prompts per (review, wording), named by the polarity of each slot:
    +- and -+   the two conflicting orders (the prompts of interest)
    ++ and --   consistent versions (same connective), used as patching sources
    +  and -    a single claim, "My doctor said <claim>. What do you think?" (behavioural references)

1. Behaviour (who wins the conflict). D = log p(Yes) - log p(No) at the last prompt position, and
       negative_weight(order) = (D(conflict) - D(+)) / (D(-) - D(+))
   0 = the model answers as if only the positive claim were there, 1 = as if only the negative claim were there.
   Mean over the two orders > 0.5 means the negative claim wins; weight(+-) - weight(-+) > 0 means the later claim wins.

2. Where each claim's polarity reaches the answer. For each conflict prompt T and slot s, the source S is T with slot s
   flipped (e.g. T = +-, slot 1 -> S = --). S's residual stream is written into T at layer L, every layer:
       last:   the last prompt token
       after:  every position after the flipped words (rest of the question, summaries, chat template), aligned from
               the end, so for slot 1 this includes claim 2
       diff:   the flipped words, only when both sides have the same number of tokens (rare: "won't" is two tokens)
   recovery(setup, L) = (D(T, patched at L) - D(T)) / (D(S) - D(T)). The clean effect D(S) - D(T) is itself the leverage
   of slot s in that conflict: a slot the model ignores has a small effect.

Outputs (in --output_dir):
    conflict_patching_pairs.jsonl    one line per (review, wording): prompts, clean readouts, per-setup per-layer D
    conflict_patching_summary.json   behaviour + per-(order, slot) per-layer recovery
    conflict_patching.png
"""
import argparse
import json
import os
import gc

import torch
from tqdm import tqdm

from patching_utils import (
    DEFAULT_MODEL_NAME, SEED, DOCTOR_PREFIX, set_global_seed, load_model_and_tokenizer, get_decoder_layers,
    format_rct_inputs, build_prompt_ids, load_dataset, sample_reviews, strip_attribution, cache_residuals,
    patch_residual,
)
from run_layer_sweep import pair_key, load_done_pairs, aggregate
from run_span_patching import find_spans, setup_positions
from run_yes_no_patching import TEMPLATE_NAME, TEMPLATE_DIR, YES_STRINGS, NO_STRINGS, single_token_ids, readout

PAIRS_FILE_NAME = "conflict_patching_pairs.jsonl"
SUMMARY_FILE_NAME = "conflict_patching_summary.json"
WORDINGS = {"doctor": " What do you think?", "ai": " Is this true?"}   # question type -> follow-up removed from the claim
FOLLOW_UP = "What do you think?"
CONFLICTS = ["+-", "-+"]
ORDER_NAMES = {"+-": "pos_first", "-+": "neg_first"}


def get_claims(example: dict, wording: str) -> dict[str, str]:
    """{'+': positive claim, '-': negative claim} without attribution, follow-up or final period."""
    claims = {}
    for sign, framing in [("+", "positive"), ("-", "negative")]:
        sentence = strip_attribution(example["Questions"][wording][f"{framing}_question"], wording)
        suffix = "." + WORDINGS[wording]
        if not sentence.endswith(suffix):
            raise ValueError(f"'{wording}' question does not end with '{suffix}': {sentence}")
        claims[sign] = sentence[:-len(suffix)]
    return claims


def build_questions(claims: dict[str, str], connective: str) -> dict[str, str]:
    questions = {sign: f"{DOCTOR_PREFIX} {claims[sign]}. {FOLLOW_UP}" for sign in "+-"}
    for s1 in "+-":
        for s2 in "+-":
            questions[s1 + s2] = f"{DOCTOR_PREFIX} {claims[s1]}{connective} {claims[s2]}. {FOLLOW_UP}"
    return questions


def flip(prompt_name: str, slot: int) -> str:
    """Prompt name with slot (1 or 2) flipped, e.g. flip('+-', 1) = '--'."""
    signs = list(prompt_name)
    signs[slot - 1] = "+" if signs[slot - 1] == "-" else "-"
    return "".join(signs)


@torch.no_grad()
def score_slot(model, src_ids, tgt_ids, spans: dict, layers: list[int], yes_ids, no_ids) -> dict[str, dict]:
    positions = {"last": ([src_ids.shape[1] - 1], [tgt_ids.shape[1] - 1])}
    positions.update({k: v for k, v in setup_positions(spans).items() if v is not None})

    patched = {}
    for setup, (src_pos, tgt_pos) in positions.items():
        with cache_residuals(model, src_pos, layer_indices=layers) as cache:
            model.model(input_ids=src_ids, use_cache=False)
        patched[setup] = {}
        for layer_idx in layers:
            with patch_residual(model, layer_idx, tgt_pos, cache[layer_idx]):
                patched[setup][str(layer_idx)] = readout(model, tgt_ids, yes_ids, no_ids)["d"]
        del cache
    return patched


def summarize(pairs_path: str, summary_path: str, min_effect: float) -> dict:
    with open(pairs_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    d_by_prompt, weights, bias, recency = {}, {o: [] for o in CONFLICTS}, [], []
    slot_effects, recovery = {}, {}   # group -> effects; group -> setup -> layer -> values
    for record in records:
        d = {name: r["d"] for name, r in record["clean"].items()}
        for name, value in d.items():
            d_by_prompt.setdefault(name, []).append(value)

        single_effect = d["-"] - d["+"]
        if abs(single_effect) >= min_effect:
            w = {o: (d[o] - d["+"]) / single_effect for o in CONFLICTS}
            for o in CONFLICTS:
                weights[o].append(w[o])
            bias.append((w["+-"] + w["-+"]) / 2)
            recency.append(w["+-"] - w["-+"])   # the negative claim is second in +-, first in -+

        for target, by_slot in record["patched"].items():
            for slot, by_setup in by_slot.items():
                effect = d[flip(target, int(slot))] - d[target]
                groups = [f"{ORDER_NAMES[target]}_slot{slot}", f"slot{slot}"]
                for group in groups:
                    slot_effects.setdefault(group, []).append(abs(effect))
                if abs(effect) < min_effect:
                    continue
                for setup, by_layer in by_setup.items():
                    for layer, d_patch in by_layer.items():
                        for group in groups:
                            recovery.setdefault(group, {}).setdefault(setup, {}).setdefault(layer, []).append(
                                (d_patch - d[target]) / effect)

    summary = {
        "min_effect": min_effect,
        "num_records": len(records),
        "behaviour": {
            "d_by_prompt": {name: aggregate(v) for name, v in d_by_prompt.items()},
            "num_records_with_single_claim_effect": len(bias),
            "negative_weight": {ORDER_NAMES[o]: aggregate(v) for o, v in weights.items()},
            "negative_weight_mean_over_orders": aggregate(bias),
            "fraction_negative_wins": sum(b > 0.5 for b in bias) / len(bias) if bias else None,
            "recency_weight_pos_first_minus_neg_first": aggregate(recency),
        },
        "abs_slot_effect": {g: aggregate(v) for g, v in sorted(slot_effects.items())},
        "recovery": {
            group: {
                setup: {l: aggregate(v) for l, v in sorted(by_layer.items(), key=lambda kv: int(kv[0]))}
                for setup, by_layer in by_setup.items()
            }
            for group, by_setup in sorted(recovery.items())
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    return summary


def plot(summary: dict, output_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), gridspec_kw={"width_ratios": [0.8, 1, 1]})

    ax = axes[0]
    names = ["+", "-", "+-", "-+", "++", "--"]
    stats = [summary["behaviour"]["d_by_prompt"][n] for n in names]
    ax.bar(range(len(names)), [s["mean"] for s in stats], yerr=[s["sem"] for s in stats], capsize=3,
           color=["#2a78d6", "#eb6834", "#7a7f8a", "#7a7f8a", "#9fc3ec", "#f4b08f"])
    ax.set_xticks(range(len(names)), ["doctor +", "doctor -", "+ but -", "- but +", "+ but +", "- but -"])
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_ylabel("Mean log p(Yes) - log p(No)")
    w = summary["behaviour"]["negative_weight"]
    ax.set_title(f"Negative weight: + first {w['pos_first']['mean']:.2f}, - first {w['neg_first']['mean']:.2f}")

    colors = {"slot1": "#2a78d6", "slot2": "#eb6834"}
    styles = {"pos_first": "-", "neg_first": "--"}
    for ax, setup in zip(axes[1:], ["last", "after"]):
        for group, by_setup in summary["recovery"].items():
            if group.startswith("slot") or setup not in by_setup:
                continue
            order, slot = group.rsplit("_", 1)
            by_layer = by_setup[setup]
            layers = [int(l) for l in by_layer]
            means = [by_layer[str(l)]["mean"] for l in layers]
            sems = [by_layer[str(l)]["sem"] for l in layers]
            ax.plot(layers, means, color=colors[slot], linestyle=styles[order],
                    label=f"{order.replace('_', ' ')}, flip claim {slot[-1]} (n={by_layer[str(layers[0])]['n']})")
            ax.fill_between(layers, [m - s for m, s in zip(means, sems)], [m + s for m, s in zip(means, sems)],
                            color=colors[slot], alpha=0.15)
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.axhline(1, color="grey", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Layer patched")
        ax.set_ylabel("Recovery")
        ax.set_title("last prompt token" if setup == "last" else "every token after the flipped claim")
        ax.legend(fontsize=8)
    fig.suptitle(f"'My doctor said A, but my doctor also said B', Yes/No readout ({summary['num_records']} records, "
                 f"|effect| >= {summary['min_effect']})")
    fig.tight_layout()
    path = os.path.join(output_dir, "conflict_patching.png")
    fig.savefig(path, dpi=200)
    print(f"Saved {path}")


def main(args):
    set_global_seed(SEED)
    os.makedirs(args.output_dir, exist_ok=True)
    pairs_path = os.path.join(args.output_dir, PAIRS_FILE_NAME)
    summary_path = os.path.join(args.output_dir, SUMMARY_FILE_NAME)

    dataset = sample_reviews(load_dataset(args.input_path), args.sample_pool, seed=args.seed)[:args.num_reviews]
    if args.debug:
        dataset = dataset[:1]
    print(f"Number of reviews: {len(dataset)}")

    model, tokenizer = load_model_and_tokenizer(args.model)
    layers = list(range(0, len(get_decoder_layers(model)), args.layer_stride))
    yes_ids, no_ids = single_token_ids(tokenizer, YES_STRINGS), single_token_ids(tokenizer, NO_STRINGS)
    print(f"Patching layers: {layers}")
    print(f"Connective: {args.connective!r}")

    done = load_done_pairs(pairs_path)
    print(f"Records already done (resuming): {len(done)}")

    with open(pairs_path, "a", encoding="utf-8") as out_file:
        for example in tqdm(dataset, desc="Reviews"):
            review_id = example["ReviewID"]
            rct_inputs = format_rct_inputs(example["Inputs"])

            for wording in args.wordings:
                key = pair_key(review_id, wording)
                if key in done:
                    continue
                questions = build_questions(get_claims(example, wording), args.connective)
                ids = {name: build_prompt_ids(tokenizer, q, rct_inputs, model.device,
                                              template_name=TEMPLATE_NAME, template_dir=TEMPLATE_DIR)
                       for name, q in questions.items()}
                clean = {name: readout(model, prompt_ids, yes_ids, no_ids) for name, prompt_ids in ids.items()}
                for r in clean.values():
                    r["top_token"] = tokenizer.decode([r["top_token_id"]])

                patched, spans = {}, {}
                for target in CONFLICTS:
                    patched[target], spans[target] = {}, {}
                    for slot in [1, 2]:
                        source = flip(target, slot)
                        slot_spans = find_spans(tokenizer, ids[source], ids[target])
                        spans[target][str(slot)] = slot_spans
                        patched[target][str(slot)] = score_slot(model, ids[source], ids[target], slot_spans, layers,
                                                                yes_ids, no_ids)

                record = {
                    "pair_key": key,
                    "review_id": review_id,
                    "wording": wording,
                    "connective": args.connective,
                    "questions": questions,
                    "clean": clean,
                    "spans": spans,
                    "patched": patched,   # target prompt -> flipped slot -> setup -> layer -> D
                }
                out_file.write(json.dumps(record) + "\n")
                out_file.flush()

            gc.collect()
            torch.cuda.empty_cache()

    summary = summarize(pairs_path, summary_path, args.min_effect)
    behaviour = summary["behaviour"]
    print(f"Records: {summary['num_records']} (single-claim |effect| >= {args.min_effect}: "
          f"{behaviour['num_records_with_single_claim_effect']})")
    for name, r in behaviour["d_by_prompt"].items():
        print(f"  D({name}): {r['mean']:.3f} +/- {r['sem']:.3f}")
    print(f"Negative weight (0 = positive claim wins, 1 = negative wins): {behaviour['negative_weight']}")
    print(f"Fraction of records where the negative claim wins (mean over orders): {behaviour['fraction_negative_wins']}")
    print(f"Recency (> 0 = later claim wins): {behaviour['recency_weight_pos_first_minus_neg_first']}")
    print(f"|Slot effect|: {summary['abs_slot_effect']}")
    for group, by_setup in summary["recovery"].items():
        if "last" in by_setup:
            print(f"Mean recovery, {group}, last:")
            for layer, r in by_setup["last"].items():
                print(f"  layer {int(layer):>2}: {r['mean']:.3f} +/- {r['sem']:.3f} (n={r['n']})")
    plot(summary, args.output_dir)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conflicting doctor claims: behaviour + per-claim patching with a Yes/No readout")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="HuggingFace model name.")
    parser.add_argument("--input_path", required=True, help="dataset with RCT inputs and framed questions (extracted question set).")
    parser.add_argument("--output_dir", required=True, help="directory to save the outputs.")
    parser.add_argument("--wordings", nargs="+", default=list(WORDINGS), choices=list(WORDINGS), help="claim wordings to use.")
    parser.add_argument("--connective", default=", but my doctor also said", help="text between the two claims.")
    parser.add_argument("--num_reviews", default=30, type=int, help="number of reviews.")
    parser.add_argument("--sample_pool", default=30, type=int, help="reviews are the first --num_reviews of the seeded sample of this size used by the main sweeps.")
    parser.add_argument("--layer_stride", default=1, type=int, help="patch every k-th layer starting at 0.")
    parser.add_argument("--min_effect", default=0.5, type=float, help="effects below this (in nats) are left out of the weights / recovery.")
    parser.add_argument("--seed", default=SEED, type=int, help="seed for sampling reviews.")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, help="only run 1 review.")
    args = parser.parse_args()

    print()
    print("Arguments Provided for Conflict Patching:")
    for k, v in vars(args).items():
        print(f"{k:<18} {v}")
    print()

    main(args)
