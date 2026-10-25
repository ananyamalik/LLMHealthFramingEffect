"""
KV ablation of the role in the clean run: "My <role> said, <claim>" with the role's keys/values hidden, or replaced by
those of "My <source_role> said, <claim>" (default source: doctor). Same prompts and reviews as run_role_patching.py
(--answer full); every condition is a greedy full answer.

Spans of the role prefix (token positions in run 1):
    word    the role word ("friend")
    said    the tokens after it, up to and including the comma (" said,"); they have already read the role word, so
            they can carry it on even when the word itself is hidden or replaced
    prefix  "My <role> said,"

Conditions per pair:
    run1            "My <role> said, <claim>", clean
    run2            "<claim>", clean (no role)
    source          "My <source_role> said, <claim>", clean (what full replacement would give)
    hide_word       run 1, nothing after the role word can attend to it (attention mask, every layer)
    hide_prefix     run 1, nothing after "My <role> said," can attend to it
    replace_word    run 1, the role word's keys/values swapped for the source run's (--kv_layers, default all)
    replace_said    run 1, the keys/values of " said," swapped for the source run's (the role word itself is kept)

Hiding keeps the positions of the later tokens (position ids come from the cache position, not the mask), so the claim
and articles sit at the same positions as in the clean run 1. Replacement needs run 1 and the source run to tokenize
to the same length and differ only in the role word (checked).

Outputs (in --output_dir/<role>/):
    ablation.jsonl   one line per (review, framing): questions, spans, and the answer for every condition (resumable)
"""
import argparse
import gc
import json
import os

import torch
from tqdm import tqdm

from patching_utils import (
    DEFAULT_MODEL_NAME, SEED, set_global_seed, load_model_and_tokenizer, get_decoder_layers, get_context_length,
    get_stop_token_ids, format_rct_inputs, load_dataset, sample_reviews,
)
from run_role_patching import FRAMINGS, role_prefix, get_claims, encode_prompt

CONDITIONS = ["run1", "run2", "source", "hide_word", "hide_prefix", "replace_word", "replace_said"]


def role_spans(encoding, question_start: int, role: str) -> dict[str, list[int]]:
    """Token positions of the role word, of " said," and of the whole prefix, in run 1."""
    word_start = question_start + len("My ")
    word_end = word_start + len(role)
    comma = question_start + len(role_prefix(role)) - 1
    word = sorted({encoding.char_to_token(c) for c in range(word_start, word_end)} - {None})
    first = encoding.char_to_token(question_start)
    comma_pos = encoding.char_to_token(comma)
    return {
        "word": word,
        "said": list(range(word[-1] + 1, comma_pos + 1)),
        "prefix": list(range(first, comma_pos + 1)),
    }


def cache_kv(cache) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """(keys, values) per layer, each [batch, kv_heads, seq, head_dim], for old (key_cache) and new (layers) DynamicCache."""
    if hasattr(cache, "layers"):
        return [(layer.keys, layer.values) for layer in cache.layers]
    return list(zip(cache.key_cache, cache.value_cache))


@torch.no_grad()
def greedy_from(model, tokenizer, ids: torch.Tensor, max_new_tokens: int, cache=None,
                attention_mask: torch.Tensor | None = None) -> list[int]:
    """
    Greedy answer with KV caching. `ids` is the prompt, or the rest of it when `cache` already holds the keys/values
    of the tokens before it. attention_mask: [1, cached + len(ids)], 0 = hidden from every later token, or None.
    """
    stop_ids = set(get_stop_token_ids(model, tokenizer))
    answer = []
    for _ in range(max_new_tokens):
        out = model.model(input_ids=ids, past_key_values=cache, attention_mask=attention_mask, use_cache=True)
        cache = out.past_key_values
        next_token = model.lm_head(out.last_hidden_state[:, -1:, :]).argmax(-1)
        if next_token.item() in stop_ids:
            break
        answer.append(next_token.item())
        ids = next_token
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(1, 1)], dim=1)
    return answer


def hidden_mask(ids: torch.Tensor, positions: list[int]) -> torch.Tensor:
    mask = torch.ones_like(ids)
    mask[0, positions] = 0
    return mask


@torch.no_grad()
def replaced_kv_answer(model, tokenizer, src_ids: torch.Tensor, source_ids: torch.Tensor, positions: list[int],
                       kv_layers: list[int], max_new_tokens: int) -> list[int]:
    """Prefill run 1 up to the end of the role prefix, swap in the source run's keys/values at `positions`, then
    process the rest of run 1's prompt on top of that cache and answer."""
    split = max(positions) + 1
    source_cache = model.model(input_ids=source_ids[:, :split], use_cache=True).past_key_values
    cache = model.model(input_ids=src_ids[:, :split], use_cache=True).past_key_values
    source_kv = cache_kv(source_cache)
    for layer_idx, (keys, values) in enumerate(cache_kv(cache)):
        if layer_idx in kv_layers:
            keys[:, :, positions, :] = source_kv[layer_idx][0][:, :, positions, :]
            values[:, :, positions, :] = source_kv[layer_idx][1][:, :, positions, :]
    del source_cache
    return greedy_from(model, tokenizer, src_ids[:, split:], max_new_tokens, cache=cache)


def main(args):
    set_global_seed(SEED)
    dataset = sample_reviews(load_dataset(args.input_path), args.num_reviews, seed=args.seed)
    if args.review_ids:
        dataset = [ex for ex in dataset if ex["ReviewID"] in set(args.review_ids)]
    if args.max_reviews is not None:
        dataset = dataset[:args.max_reviews]
    print(f"Number of reviews: {len(dataset)}")

    model, tokenizer = load_model_and_tokenizer(args.model)
    num_layers = len(get_decoder_layers(model))
    kv_layers = args.kv_layers if args.kv_layers else list(range(num_layers))
    max_prompt_len = get_context_length(model) - args.max_new_tokens

    for role in args.roles:
        if role == args.source_role:
            print(f"Skipping role {role}: same as --source_role")
            continue
        role_dir = os.path.join(args.output_dir, role)
        os.makedirs(role_dir, exist_ok=True)
        out_path = os.path.join(role_dir, "ablation.jsonl")
        done = set()
        if os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as f:
                done = {json.loads(line)["pair_key"] for line in f if line.strip()}
        print(f"\n=== Role: {role} -> {args.source_role} ({len(done)} pairs already done) ===")

        with open(out_path, "a", encoding="utf-8") as out_file:
            for example in tqdm(dataset, desc=role):
                rct_inputs = format_rct_inputs(example["Inputs"])
                for framing, claim in get_claims(example).items():
                    key = f"{example['ReviewID']}_{framing}"
                    if framing not in args.framings or key in done:
                        continue
                    src_question = f"{role_prefix(role)} {claim}"
                    source_question = f"{role_prefix(args.source_role)} {claim}"
                    src_ids, src_enc, src_q_start = encode_prompt(tokenizer, "full", src_question, rct_inputs, model.device)
                    source_ids, _, _ = encode_prompt(tokenizer, "full", source_question, rct_inputs, model.device)
                    tgt_ids, _, _ = encode_prompt(tokenizer, "full", claim, rct_inputs, model.device)
                    if src_ids.shape[1] > max_prompt_len or source_ids.shape[1] > max_prompt_len:
                        print(f"Skipping {key}: prompt too long ({src_ids.shape[1]} tokens)")
                        continue

                    spans = role_spans(src_enc, src_q_start, role)
                    differs = (src_ids != source_ids).nonzero()[:, 1].tolist() if src_ids.shape == source_ids.shape else None
                    if differs is None or not set(differs) <= set(spans["word"]):
                        raise ValueError(f"{key}: '{role}' and '{args.source_role}' prompts differ outside the role word; replacement needs aligned tokens")

                    answers = {
                        "run1": greedy_from(model, tokenizer, src_ids, args.max_new_tokens),
                        "run2": greedy_from(model, tokenizer, tgt_ids, args.max_new_tokens),
                        "source": greedy_from(model, tokenizer, source_ids, args.max_new_tokens),
                        "hide_word": greedy_from(model, tokenizer, src_ids, args.max_new_tokens,
                                                 attention_mask=hidden_mask(src_ids, spans["word"])),
                        "hide_prefix": greedy_from(model, tokenizer, src_ids, args.max_new_tokens,
                                                   attention_mask=hidden_mask(src_ids, spans["prefix"])),
                        "replace_word": replaced_kv_answer(model, tokenizer, src_ids, source_ids, spans["word"],
                                                           kv_layers, args.max_new_tokens),
                        "replace_said": replaced_kv_answer(model, tokenizer, src_ids, source_ids, spans["said"],
                                                           kv_layers, args.max_new_tokens),
                    }
                    record = {
                        "pair_key": key,
                        "review_id": example["ReviewID"],
                        "role": role,
                        "source_role": args.source_role,
                        "framing": framing,
                        "run1_question": src_question,
                        "run2_question": claim,
                        "source_question": source_question,
                        "spans": spans,
                        "kv_layers": kv_layers,
                    }
                    for name, ids in answers.items():
                        record[f"{name}_answer"] = tokenizer.decode(ids, skip_special_tokens=True).strip()
                    out_file.write(json.dumps(record) + "\n")
                    out_file.flush()
                gc.collect()
                torch.cuda.empty_cache()
        print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hide or replace the role's keys/values in 'My <role> said, <claim>' (full answers)")
    parser.add_argument("--roles", nargs="+", default=["friend", "enemy"], help="roles to ablate.")
    parser.add_argument("--source_role", default="doctor", help="role whose keys/values are swapped in.")
    parser.add_argument("--framings", nargs="+", default=FRAMINGS, choices=FRAMINGS, help="framings to run.")
    parser.add_argument("--kv_layers", nargs="*", type=int, default=None, help="layers whose keys/values are replaced (default: all).")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="HuggingFace model name.")
    parser.add_argument("--input_path", required=True, help="dataset with RCT inputs and framed questions (extracted question set).")
    parser.add_argument("--output_dir", required=True, help="outputs go to <output_dir>/<role>/.")
    parser.add_argument("--num_reviews", default=15, type=int, help="seeded sample of reviews; keep the role-patching value so the same reviews are drawn.")
    parser.add_argument("--review_ids", nargs="*", default=None, help="only these reviews (from the sample).")
    parser.add_argument("--max_reviews", default=None, type=int, help="only the first N reviews (after --review_ids).")
    parser.add_argument("--max_new_tokens", default=512, type=int, help="max tokens per answer.")
    parser.add_argument("--seed", default=SEED, type=int, help="seed for sampling reviews.")
    args = parser.parse_args()

    print("\nArguments:")
    for k, v in vars(args).items():
        print(f"{k:<18} {v}")
    print()
    main(args)
