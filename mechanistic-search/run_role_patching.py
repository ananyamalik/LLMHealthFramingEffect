"""
Role patching: "My <role> said, <claim>" -> "<claim>".

Each review gives one claim in two framings (the `doctor` question with "My doctor says " removed):
    positive: "amantadine will work for fatigue in multiple sclerosis. What do you think?"
    negative: "amantadine won't work for fatigue in multiple sclerosis. What do you think?"
For each role (doctor, friend, enemy) and framing:
    run 1 (source): "My <role> said, <claim>"
    run 2 (target): "<claim>"
The claim tokens and everything after them (articles, chat template) are identical in both runs (checked).

Prompts (--prompt):
    question_only (default): the user message is just "Question: <question>" (yes_no adds 'Does the treatment in the
                  question work? Answer with only "Yes" or "No".'). No instructions, no article summaries.
    articles:     the prompts below, with instructions and the review's article summaries.

Setups (run 1's residual stream written into run 2 at one layer L, every layer, one setup at a time):
    last:  last prompt token of run 1  -> last prompt token of run 2
    comma: the "," of "My <role> said," in run 1 -> the token right before <claim> in run 2 (the start of the
           question, where the role would be)

Answers (--answer):
    yes_no: articles: prompts/question_answering_yes_no.jinja2 (ends in 'Answer with only "Yes" or "No"');
            question_only: prompts/question_only_yes_no.jinja2.
            D(prompt) = log p(Yes) - log p(No) at the last prompt position.
    full:   articles: code/prompts/question_answering.jinja2 (the main pipeline's prompt); question_only:
            prompts/question_only.jinja2. R1, R2 = greedy answers to run 1/run 2.
            D(prompt) = log p(R1 | prompt) - log p(R2 | prompt), teacher forced.
            Saved answers: run1_answer, run2_answer (clean), and patched_run2_answers[setup][layer], run 2's greedy
            answer with that patch, at --generate_layers (default every 10th layer).

    recovery(setup, L) = (D(run 2, patched at L) - D(run 2)) / (D(run 1) - D(run 2))
        0 = the patch does nothing, 1 = the patched run 2 behaves like run 1.

Outputs (in --output_dir/<role>/):
    pairs.jsonl     one line per (review, framing): prompts, clean D, patched D per setup and layer (resumable)
    summary.json    recovery per setup and layer (all / positive / negative), and the clean role effect
"""
import argparse
import gc
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from patching_utils import (
    DEFAULT_MODEL_NAME, SEED, render_prompt, set_global_seed, load_model_and_tokenizer, get_decoder_layers,
    get_context_length, format_rct_inputs, load_dataset, sample_reviews, cache_residuals, patch_residual,
    greedy_generate, response_logprob, yes_no_logit_diff,
)

ROLES = ["doctor", "friend", "enemy"]
FRAMINGS = ["positive", "negative"]
SETUPS = ["last", "comma"]
CLAIM_ATTRIBUTION = "My doctor says "   # removed from the `doctor` question to get the bare claim
ANSWERS = ["yes_no", "full"]
PROMPTS = ["question_only", "articles"]
# Template dirs are relative to code/, where render_prompt resolves them.
# articles: instructions + question + article summaries (the main pipeline's prompt)
TEMPLATES = {
    "yes_no": ("question_answering_yes_no", "../mechanistic-search/prompts"),
    "full": ("question_answering", "./prompts"),
}
# question_only: "Question: <question>" and nothing else (yes_no adds the Yes/No instruction). The "Question: " prefix
# keeps the claim's first token identical with and without the role (" incision" after "said," vs. at message start)
QUESTION_ONLY_TEMPLATES = {
    "yes_no": ("question_only_yes_no", "../mechanistic-search/prompts"),
    "full": ("question_only", "../mechanistic-search/prompts"),
}
QUESTION_MARKERS = {"articles": "**QUESTION**: ", "question_only": "Question: "}
DEFAULT_MIN_EFFECT = {"yes_no": 0.5, "full": 3.0}   # nats; pairs with a smaller |D(run 1) - D(run 2)| are not summarized
YES_STRINGS, NO_STRINGS = ["Yes", "yes"], ["No", "no"]


def role_prefix(role: str) -> str:
    return f"My {role} said,"


def get_claims(example: dict) -> dict[str, str]:
    claims = {}
    for framing in FRAMINGS:
        question = example["Questions"]["doctor"][f"{framing}_question"]
        if not question.startswith(CLAIM_ATTRIBUTION):
            raise ValueError(f"doctor question does not start with '{CLAIM_ATTRIBUTION}': {question}")
        claims[framing] = question[len(CLAIM_ATTRIBUTION):]
    return claims


def encode_prompt(tokenizer, answer: str, question: str, rct_inputs: list[dict], device, prompt: str = "articles"):
    """Chat-template prompt ids [1, seq], the encoding with character offsets, and the character index of the question."""
    template_name, template_dir = (TEMPLATES if prompt == "articles" else QUESTION_ONLY_TEMPLATES)[answer]
    user_text = render_prompt(template_name, template_dir=template_dir, question=question, abstracts=rct_inputs)
    messages = [{"role": "user", "content": user_text}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)

    reference_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if not isinstance(reference_ids, list):  # newer transformers can return a BatchEncoding
        reference_ids = reference_ids["input_ids"]
    if encoding["input_ids"] != reference_ids:
        raise ValueError("Offset tokenization does not match the chat-template ids")

    marker = QUESTION_MARKERS[prompt]
    question_start = text.index(marker + question) + len(marker)
    return torch.tensor([encoding["input_ids"]], device=device), encoding, question_start


def find_positions(tokenizer, src, tgt, prefix: str) -> dict:
    """Source/target positions for each setup, checking that the claim and suffix tokens match."""
    src_ids, src_enc, src_q_start = src
    tgt_ids, tgt_enc, tgt_q_start = tgt

    comma_pos = src_enc.char_to_token(src_q_start + len(prefix) - 1)
    if tokenizer.decode([src_ids[0, comma_pos].item()]) != ",":
        raise ValueError(f"Comma is not its own token: {tokenizer.decode([src_ids[0, comma_pos].item()])!r}")
    tgt_claim_start = tgt_enc.char_to_token(tgt_q_start)
    if not torch.equal(src_ids[0, comma_pos + 1:], tgt_ids[0, tgt_claim_start:]):
        raise ValueError("Claim/suffix tokens differ between run 1 and run 2")

    start_pos = tgt_claim_start - 1
    return {
        "last": {"source": src_ids.shape[1] - 1, "target": tgt_ids.shape[1] - 1},
        "comma": {"source": comma_pos, "target": start_pos},
        "comma_target_token": tokenizer.decode([tgt_ids[0, start_pos].item()]),
    }


def single_token_ids(tokenizer, strings: list[str]) -> list[int]:
    ids = []
    for s in strings:
        encoded = tokenizer.encode(s, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"{s!r} is not a single token: {encoded}")
        ids.append(encoded[0])
    return ids


@torch.no_grad()
def run_pair(model, tokenizer, args, src_ids, tgt_ids, positions: dict, layers: list[int], yes_no_ids) -> dict:
    """Clean and patched D for one pair. The scoring function depends on --answer."""
    record = {}
    if args.answer == "yes_no":
        yes_ids, no_ids = yes_no_ids

        def score(ids):
            return yes_no_logit_diff(model, ids, yes_ids, no_ids)[0]

        for name, ids in [("source", src_ids), ("target", tgt_ids)]:
            d, top_id = yes_no_logit_diff(model, ids, yes_ids, no_ids)
            record[f"d_{name}"], record[f"{name}_top_token"] = d, tokenizer.decode([top_id])
    else:
        src_resp = greedy_generate(model, tokenizer, src_ids, args.max_new_tokens)
        tgt_resp = greedy_generate(model, tokenizer, tgt_ids, args.max_new_tokens)

        def score(ids):
            return response_logprob(model, ids, src_resp) - response_logprob(model, ids, tgt_resp)

        record["run1_answer"] = tokenizer.decode(src_resp, skip_special_tokens=True).strip()   # "My <role> said, <claim>"
        record["run2_answer"] = tokenizer.decode(tgt_resp, skip_special_tokens=True).strip()   # "<claim>", clean
        record["identical_answers"] = torch.equal(src_resp, tgt_resp)
        record["d_source"], record["d_target"] = score(src_ids), score(tgt_ids)

    src_positions = [positions[setup]["source"] for setup in SETUPS]
    with cache_residuals(model, src_positions, layer_indices=layers) as cache:
        model.model(input_ids=src_ids, use_cache=False)

    record["d_patched"] = {}
    if args.answer == "full":
        record["patched_run2_answers"] = {}   # setup -> layer -> run 2's greedy answer with that patch
    for i, setup in enumerate(SETUPS):
        tgt_pos = [positions[setup]["target"]]
        record["d_patched"][setup] = {}
        for layer_idx in layers:
            with patch_residual(model, layer_idx, tgt_pos, cache[layer_idx][i:i + 1]):
                record["d_patched"][setup][str(layer_idx)] = score(tgt_ids)
        if args.answer == "full":
            record["patched_run2_answers"][setup] = {}
            for layer_idx in args.generate_layers:
                with patch_residual(model, layer_idx, tgt_pos, cache[layer_idx][i:i + 1]):
                    resp = greedy_generate(model, tokenizer, tgt_ids, args.max_new_tokens)
                record["patched_run2_answers"][setup][str(layer_idx)] = tokenizer.decode(resp, skip_special_tokens=True).strip()
    del cache
    return record


def aggregate(values: list[float]) -> dict:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"n": 0, "mean": None, "sem": None, "median": None}
    sem = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return {"n": int(len(values)), "mean": float(values.mean()), "sem": sem, "median": float(np.median(values))}


def summarize(pairs_path: str, summary_path: str, role: str, answer: str, min_effect: float) -> dict:
    with open(pairs_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    recovery = {}   # group -> setup -> layer -> values
    num_with_effect = 0
    for record in records:
        effect = record["d_source"] - record["d_target"]
        if abs(effect) < min_effect:
            continue
        num_with_effect += 1
        for setup, by_layer in record["d_patched"].items():
            for layer, d_patched in by_layer.items():
                for group in ["all", record["framing"]]:
                    value = (d_patched - record["d_target"]) / effect
                    recovery.setdefault(group, {}).setdefault(setup, {}).setdefault(layer, []).append(value)

    summary = {
        "role": role,
        "answer": answer,
        "min_effect": min_effect,
        "num_pairs": len(records),
        "num_pairs_with_effect": num_with_effect,
        # Clean role effect, all pairs: for yes_no, > 0 means the role makes the model say "Yes" more
        "clean": {
            group: {
                key: aggregate([r[key] for r in records if group in ("all", r["framing"])])
                for key in ["d_source", "d_target"]
            } | {"effect": aggregate([r["d_source"] - r["d_target"] for r in records if group in ("all", r["framing"])])}
            for group in ["all"] + FRAMINGS
        },
        "recovery": {
            group: {
                setup: {l: aggregate(v) for l, v in sorted(by_layer.items(), key=lambda kv: int(kv[0]))}
                for setup, by_layer in recovery[group].items()
            }
            for group in ["all"] + FRAMINGS if group in recovery
        },
    }
    if answer == "full":
        summary["num_identical_answers"] = sum(r["identical_answers"] for r in records)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    return summary


def main(args):
    set_global_seed(SEED)
    min_effect = args.min_effect if args.min_effect is not None else DEFAULT_MIN_EFFECT[args.answer]

    dataset = sample_reviews(load_dataset(args.input_path), args.num_reviews, seed=args.seed)
    if args.debug:
        dataset = dataset[:1]
    print(f"Number of reviews: {len(dataset)}")

    model, tokenizer = load_model_and_tokenizer(args.model)
    num_layers = len(get_decoder_layers(model))
    layers = list(range(0, num_layers, args.layer_stride))
    if args.generate_layers is None:   # default: every 10th layer (0, 10, 20, 30 for 32-layer models)
        args.generate_layers = list(range(0, num_layers, 10))
    if args.answer == "full":
        print(f"Generating patched answers at layers: {args.generate_layers}")
    yes_no_ids = (single_token_ids(tokenizer, YES_STRINGS), single_token_ids(tokenizer, NO_STRINGS))
    max_prompt_len = get_context_length(model) - (args.max_new_tokens if args.answer == "full" else 0)
    print(f"Patching layers: {layers}")

    for role in args.roles:
        role_dir = os.path.join(args.output_dir, role)
        os.makedirs(role_dir, exist_ok=True)
        pairs_path, summary_path = os.path.join(role_dir, "pairs.jsonl"), os.path.join(role_dir, "summary.json")
        done = set()
        if os.path.exists(pairs_path):
            with open(pairs_path, "r", encoding="utf-8") as f:
                done = {json.loads(line)["pair_key"] for line in f if line.strip()}
        print(f"\n=== Role: {role} ({len(done)} pairs already done) ===")

        with open(pairs_path, "a", encoding="utf-8") as out_file:
            for example in tqdm(dataset, desc=role):
                rct_inputs = format_rct_inputs(example["Inputs"])
                for framing, claim in get_claims(example).items():
                    key = f"{example['ReviewID']}_{framing}"
                    if key in done:
                        continue
                    prefix = role_prefix(role)
                    src_question = f"{prefix} {claim}"
                    src = encode_prompt(tokenizer, args.answer, src_question, rct_inputs, model.device, args.prompt)
                    tgt = encode_prompt(tokenizer, args.answer, claim, rct_inputs, model.device, args.prompt)
                    if src[0].shape[1] > max_prompt_len:
                        print(f"Skipping {key}: prompt too long ({src[0].shape[1]} tokens)")
                        continue
                    try:
                        positions = find_positions(tokenizer, src, tgt, prefix)
                    except ValueError as e:   # e.g. the claim's first word tokenizes differently after "said,"
                        print(f"Skipping {key}: {e}")
                        continue

                    record = {
                        "pair_key": key,
                        "review_id": example["ReviewID"],
                        "role": role,
                        "framing": framing,
                        "answer": args.answer,
                        "prompt": args.prompt,
                        "source_question": src_question,
                        "target_question": claim,
                        "positions": positions,
                    }
                    record |= run_pair(model, tokenizer, args, src[0], tgt[0], positions, layers, yes_no_ids)
                    out_file.write(json.dumps(record) + "\n")
                    out_file.flush()
                gc.collect()
                torch.cuda.empty_cache()

        summary = summarize(pairs_path, summary_path, role, args.answer, min_effect)
        clean = summary["clean"]
        print(f"Pairs: {summary['num_pairs']} (|effect| >= {min_effect}: {summary['num_pairs_with_effect']})")
        for framing in FRAMINGS:
            e = clean[framing]["effect"]
            if e["n"]:
                print(f"  clean effect D(run 1) - D(run 2), {framing} claims: {e['mean']:.3f} +/- {e['sem']:.3f}")
        for setup, by_layer in summary["recovery"].get("all", {}).items():
            print(f"Mean recovery, {setup}: " + ", ".join(f"L{int(l)} {r['mean']:.2f}" for l, r in by_layer.items()))
        print(f"Saved {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch 'My <role> said, <claim>' into '<claim>' (last token and comma)")
    parser.add_argument("--answer", required=True, choices=ANSWERS, help="yes_no (single-token readout) or full (long-form answer).")
    parser.add_argument("--prompt", default="question_only", choices=PROMPTS, help="question_only ('Question: <question>', no instructions or articles) or articles (the main pipeline's prompt with article summaries).")
    parser.add_argument("--roles", nargs="+", default=ROLES, help="roles to run (any word works in 'My <role> said,').")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="HuggingFace model name.")
    parser.add_argument("--input_path", required=True, help="dataset with RCT inputs and framed questions (extracted question set).")
    parser.add_argument("--output_dir", required=True, help="outputs go to <output_dir>/<role>/.")
    parser.add_argument("--num_reviews", default=15, type=int, help="number of reviews (seeded sample; 2 pairs each per role).")
    parser.add_argument("--layer_stride", default=1, type=int, help="patch every k-th layer starting at 0.")
    parser.add_argument("--max_new_tokens", default=512, type=int, help="full only: max tokens for greedy answers.")
    parser.add_argument("--generate_layers", nargs="*", type=int, default=None, help="full only: layers at which run 2's patched answer is generated and saved (default: every 10th layer; pass the flag with no values for none).")
    parser.add_argument("--min_effect", default=None, type=float, help="pairs with |D(run 1) - D(run 2)| below this are left out of the summary (default 0.5 yes_no, 3 full).")
    parser.add_argument("--seed", default=SEED, type=int, help="seed for sampling reviews.")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, help="only run 1 review.")
    args = parser.parse_args()

    print("\nArguments:")
    for k, v in vars(args).items():
        print(f"{k:<18} {v}")
    print()
    main(args)
