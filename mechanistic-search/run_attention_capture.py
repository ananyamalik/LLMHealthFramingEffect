"""
Attention during generation for role patching: what the model attends to while it writes its full answer, in the
clean runs and in the patched run 2. Same prompts, pairs and patching setups as run_role_patching.py (--answer full).

Conditions per pair:
    run1                    "My <role> said, <claim>", clean
    run2                    "<claim>", clean
    patched_<setup>_L<l>    run 2 with run 1's residual stream written in at one layer (setups: last, comma)

Each generation is greedy with KV caching. The prompt minus its last token is prefilled; the last prompt token and
every generated token are then fed one at a time with output_attentions, so step 0 is the attention of the last prompt
token (which predicts the first answer token) and step t > 0 is the attention of answer token t - 1. Attention weights
need the eager implementation, so answers can differ slightly from the sdpa answers in pairs.jsonl.

Every prompt token is assigned to one segment:
    first_token   position 0 (usually an attention sink)
    template      chat-template tokens outside the user message
    instructions  the rest of the user message (task text, "**QUESTION**: ", "**ARTICLE SUMMARIES**: ")
    role          "My <role> said," (run 1 only)
    claim         the claim
    articles      the article summaries
Two more columns: generated (answer tokens so far) and comma_pos, the comma setup's patch slot (run 1: the "," of
"My <role> said,"; run 2: the token before the claim). comma_pos overlaps the other segments.

Outputs (in --output_dir/<role>/):
    attention.jsonl                     one line per (pair, condition): answer, number of steps, columns, file
    attention/<pair>__<condition>.npz   segment_mass [steps, layers, columns] (or [steps, layers, heads, columns] with
                                        --per_head), averaged over heads; token_mass [layers, prompt_len], the attention
                                        to each prompt token averaged over heads and steps; prompt_tokens; segment_ids
"""
import argparse
import gc
import json
import os
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
from tqdm import tqdm

from patching_utils import (
    DEFAULT_MODEL_NAME, SEED, render_prompt, set_global_seed, load_model_and_tokenizer, get_context_length,
    get_stop_token_ids, format_rct_inputs, load_dataset, sample_reviews, cache_residuals,
)
from run_role_patching import ROLES, FRAMINGS, SETUPS, TEMPLATES, role_prefix, get_claims, encode_prompt, find_positions

ARTICLES_MARKER = "**ARTICLE SUMMARIES**: "
SEGMENTS = ["first_token", "template", "instructions", "role", "claim", "articles"]
COLUMNS = SEGMENTS + ["generated", "comma_pos"]


def char_span_to_tokens(encoding, start: int, end: int) -> tuple[int, int]:
    """Token range [first, last + 1) covering characters [start, end), skipping characters with no token."""
    first = next(encoding.char_to_token(c) for c in range(start, end) if encoding.char_to_token(c) is not None)
    last = next(encoding.char_to_token(c) for c in range(end - 1, start - 1, -1) if encoding.char_to_token(c) is not None)
    return first, last + 1


def segment_ids(tokenizer, question: str, claim: str, rct_inputs: list[dict], encoding, question_start: int,
                prompt_len: int) -> np.ndarray:
    """Index into SEGMENTS for every prompt token. The prompt is built the same way as in encode_prompt."""
    template_name, template_dir = TEMPLATES["full"]
    user_text = render_prompt(template_name, template_dir=template_dir, question=question, abstracts=rct_inputs).strip()
    text = tokenizer.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True)

    user_start = text.index(user_text)
    user_end = user_start + len(user_text)
    claim_start = question_start + len(question) - len(claim)
    if text[claim_start:claim_start + len(claim)] != claim:
        raise ValueError("Claim not found at the expected position")
    articles_start = text.index(ARTICLES_MARKER, question_start) + len(ARTICLES_MARKER)

    seg = np.full(prompt_len, SEGMENTS.index("template"), dtype=np.int64)
    spans = [("instructions", user_start, user_end), ("articles", articles_start, user_end),
             ("claim", claim_start, claim_start + len(claim))]
    if claim_start > question_start:   # run 1: "My <role> said," (the space before the claim goes with the claim's first token)
        spans.append(("role", question_start, claim_start - 1))
    for name, start, end in spans:   # later spans overwrite earlier ones
        first, last = char_span_to_tokens(encoding, start, end)
        seg[first:last] = SEGMENTS.index(name)
    seg[0] = SEGMENTS.index("first_token")
    return seg


@contextmanager
def patch_residual_at(model, layer_idx: int, position: int, patch_value: torch.Tensor, state: dict):
    """
    Like patching_utils.patch_residual, but for a decode loop that feeds the prompt in pieces: state["offset"] is the
    absolute position of the first token of the current forward pass, so the patch lands wherever `position` falls
    (the prefill for the comma setup, the first single-token step for the last setup).
    """
    layer = model.model.layers[layer_idx]

    def hook(module, inputs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        rel = position - state["offset"]
        if 0 <= rel < hidden_states.shape[1]:
            hidden_states[0, rel, :] = patch_value.to(device=hidden_states.device, dtype=hidden_states.dtype)
        return output

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def generate_with_attention(model, tokenizer, input_ids: torch.Tensor, seg: np.ndarray, comma_pos: int,
                            max_new_tokens: int, per_head: bool, patch: tuple | None = None):
    """
    Greedy answer plus attention per step. patch = (layer_idx, target_position, value) or None.
    Returns (answer ids, segment_mass [steps, layers, (heads,) columns], token_mass [layers, prompt_len]).
    """
    prompt_len = input_ids.shape[1]
    stop_ids = set(get_stop_token_ids(model, tokenizer))
    seg_index = torch.as_tensor(seg, device=input_ids.device)
    state = {"offset": 0}
    steps, answer = [], []
    token_mass = None

    with patch_residual_at(model, *patch, state) if patch else nullcontext():
        past = model.model(input_ids=input_ids[:, :-1], use_cache=True).past_key_values
        state["offset"] = prompt_len - 1
        next_ids = input_ids[:, -1:]
        for _ in range(max_new_tokens):
            out = model.model(input_ids=next_ids, past_key_values=past, use_cache=True, output_attentions=True)
            past = out.past_key_values
            if not out.attentions or out.attentions[0] is None:
                raise RuntimeError("No attention weights returned; load the model with attn_implementation='eager'")

            attn = torch.stack([a[0, :, -1, :] for a in out.attentions]).float()   # [layers, heads, kv_len]
            prompt_attn = attn[..., :prompt_len]
            seg_mass = torch.zeros(*attn.shape[:2], len(SEGMENTS), device=attn.device).index_add_(2, seg_index.to(attn.device), prompt_attn)
            mass = torch.cat([seg_mass, attn[..., prompt_len:].sum(-1, keepdim=True), attn[..., comma_pos:comma_pos + 1]], dim=-1)
            steps.append((mass if per_head else mass.mean(1)).cpu())
            head_mean = prompt_attn.mean(1)
            token_mass = head_mean if token_mass is None else token_mass + head_mean

            next_token = model.lm_head(out.last_hidden_state[:, -1:, :]).argmax(-1)
            state["offset"] += next_ids.shape[1]
            if next_token.item() in stop_ids:
                break
            answer.append(next_token.item())
            next_ids = next_token
    del past
    return answer, torch.stack(steps).numpy(), (token_mass / len(steps)).cpu().numpy()


def main(args):
    set_global_seed(SEED)
    dataset = sample_reviews(load_dataset(args.input_path), args.num_reviews, seed=args.seed)
    if args.review_ids:
        missing = set(args.review_ids) - {ex["ReviewID"] for ex in dataset}
        if missing:
            print(f"WARNING: not in the sampled reviews (raise --num_reviews?): {sorted(missing)}")
        dataset = [ex for ex in dataset if ex["ReviewID"] in set(args.review_ids)]
    if args.max_reviews is not None:
        dataset = dataset[:args.max_reviews]
    print(f"Number of reviews: {len(dataset)}")

    model, tokenizer = load_model_and_tokenizer(args.model, attn_implementation="eager")
    max_prompt_len = get_context_length(model) - args.max_new_tokens

    for role in args.roles:
        role_dir = os.path.join(args.output_dir, role)
        attn_dir = os.path.join(role_dir, "attention")
        os.makedirs(attn_dir, exist_ok=True)
        meta_path = os.path.join(role_dir, "attention.jsonl")
        done = set()
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                done = {(r["pair_key"], r["condition"]) for r in map(json.loads, filter(str.strip, f))}
        print(f"\n=== Role: {role} ({len(done)} generations already done) ===")

        with open(meta_path, "a", encoding="utf-8") as meta_file:
            for example in tqdm(dataset, desc=role):
                rct_inputs = format_rct_inputs(example["Inputs"])
                for framing, claim in get_claims(example).items():
                    if framing not in args.framings:
                        continue
                    key = f"{example['ReviewID']}_{framing}"
                    prefix = role_prefix(role)
                    src_question = f"{prefix} {claim}"
                    src = encode_prompt(tokenizer, "full", src_question, rct_inputs, model.device)
                    tgt = encode_prompt(tokenizer, "full", claim, rct_inputs, model.device)
                    if src[0].shape[1] > max_prompt_len:
                        print(f"Skipping {key}: prompt too long ({src[0].shape[1]} tokens)")
                        continue
                    positions = find_positions(tokenizer, src, tgt, prefix)
                    src_seg = segment_ids(tokenizer, src_question, claim, rct_inputs, src[1], src[2], src[0].shape[1])
                    tgt_seg = segment_ids(tokenizer, claim, claim, rct_inputs, tgt[1], tgt[2], tgt[0].shape[1])

                    # (condition, prompt ids, segments, comma slot, patch)
                    conditions = [("run1", src[0], src_seg, positions["comma"]["source"], None),
                                  ("run2", tgt[0], tgt_seg, positions["comma"]["target"], None)]
                    patched = [(f"patched_{setup}_L{layer}", i, setup, layer)
                               for i, setup in enumerate(args.setups) for layer in args.layers]
                    if any((key, name) not in done for name, *_ in patched):
                        with cache_residuals(model, [positions[s]["source"] for s in args.setups], layer_indices=args.layers) as cache:
                            model.model(input_ids=src[0], use_cache=False)
                        for name, i, setup, layer in patched:
                            patch = (layer, positions[setup]["target"], cache[layer][i])
                            conditions.append((name, tgt[0], tgt_seg, positions["comma"]["target"], patch))

                    for name, ids, seg, comma_pos, patch in conditions:
                        if (key, name) in done:
                            continue
                        answer, segment_mass, token_mass = generate_with_attention(
                            model, tokenizer, ids, seg, comma_pos, args.max_new_tokens, args.per_head, patch,
                        )
                        file_name = f"{key}__{name}.npz"
                        np.savez_compressed(
                            os.path.join(attn_dir, file_name),
                            segment_mass=segment_mass.astype(np.float16),
                            token_mass=token_mass.astype(np.float16),
                            prompt_tokens=np.array(tokenizer.convert_ids_to_tokens(ids[0].tolist())),
                            segment_ids=seg,
                        )
                        meta_file.write(json.dumps({
                            "pair_key": key,
                            "review_id": example["ReviewID"],
                            "role": role,
                            "framing": framing,
                            "condition": name,
                            "setup": patch and name.split("_")[1],
                            "layer": patch and patch[0],
                            "answer": tokenizer.decode(answer, skip_special_tokens=True).strip(),
                            "n_steps": int(segment_mass.shape[0]),
                            "prompt_len": int(ids.shape[1]),
                            "columns": COLUMNS,
                            "segment_token_counts": {s: int((seg == j).sum()) for j, s in enumerate(SEGMENTS)},
                            "positions": positions,
                            "file": os.path.join("attention", file_name),
                        }) + "\n")
                        meta_file.flush()
                gc.collect()
                torch.cuda.empty_cache()
        print(f"Saved {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attention per generation step, clean and patched (role patching, full answers)")
    parser.add_argument("--roles", nargs="+", default=ROLES, help="roles to run.")
    parser.add_argument("--framings", nargs="+", default=FRAMINGS, choices=FRAMINGS, help="framings to run.")
    parser.add_argument("--setups", nargs="+", default=SETUPS, choices=SETUPS, help="patching setups.")
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 10, 20, 30], help="layers to patch (one patched generation each).")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="HuggingFace model name.")
    parser.add_argument("--input_path", required=True, help="dataset with RCT inputs and framed questions (extracted question set).")
    parser.add_argument("--output_dir", required=True, help="outputs go to <output_dir>/<role>/.")
    parser.add_argument("--num_reviews", default=15, type=int, help="seeded sample of reviews; keep the role-patching value so the same reviews are drawn.")
    parser.add_argument("--review_ids", nargs="*", default=None, help="only these reviews (from the sample).")
    parser.add_argument("--max_reviews", default=None, type=int, help="only the first N reviews (after --review_ids).")
    parser.add_argument("--max_new_tokens", default=512, type=int, help="max tokens per answer.")
    parser.add_argument("--per_head", action=argparse.BooleanOptionalAction, help="keep the head dimension in segment_mass.")
    parser.add_argument("--seed", default=SEED, type=int, help="seed for sampling reviews.")
    args = parser.parse_args()

    print("\nArguments:")
    for k, v in vars(args).items():
        print(f"{k:<18} {v}")
    print()
    main(args)
