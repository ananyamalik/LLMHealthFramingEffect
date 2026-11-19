"""
Framing x role residual streams of a small local reasoning model, on the prompt and through the whole generation.

Same design as mechanistic-search/run_interaction_vector.py and role_framing_vectors.py, on a model that thinks
before it answers (default Qwen/Qwen3-0.6B with thinking on; DeepSeek-R1-Distill-Qwen-1.5B and Qwen3-1.7B also work).

Prompts (identical to mechanistic-search's question-only prompts): for every review, both framings x (bare claim +
"My <role> said,"):
    Question: [My <role> said, ]<intervention> will|won't work for <condition>. What do you think?
The model writes greedily: a reasoning trace (<think> ... </think>), then its answer. If the trace has not closed
after --max_think_tokens, "</think>" is forced (budget forcing, flagged as `forced_think_end`), and the answer gets up to
--max_answer_tokens more.

Residual stream = every decoder layer's output (as in patching_utils.cache_residuals), recorded with forward hooks
during generation itself (KV-cached decoding): the prefill pass gives every prompt token, then each decode step gives
the token that was just fed in. gen_states[:, i] is the residual at generated token i, the position that predicts
token i + 1; the last prompt token's residual predicts generated token 0.

Also per question (no thinking, a single forward pass each):
    d_nothink   D = log p(Yes) - log p(No) for the yes/no prompt (question_only_yes_no) with thinking switched off;
                its last-token residual is the `yes_no` site (as in get_diffmeans_directions.py)
    d_after     D for the yes/no question asked as a follow-up turn after the model's own final answer (thinking off)

Sites (each [layers, d_model], saved per question in sites/<key>.pt, what reasoning_vectors.py fits vectors on):
    prompt:      intervention, verdict, claim_end, question_end, last   (run_belief_probe.py's anchors), yes_no
    generation:  think_mean (mean over the reasoning tokens), think_end (the </think> token), answer_first (first
                 token after </think> and its whitespace), answer_mean, answer_last, gen_mean (all generated tokens),
                 think_bin_<k> (--think_bins equal slices of the reasoning trace, mean in each),
                 answer_bin_<k> (--answer_bins slices of the answer)

Outputs (--output_dir, default reasoning_anchors/outputs/<model_tag>/):
    generations/generations.jsonl   one line per question: ids, prompt text, the raw generation (with <think> tags),
                                    reasoning, answer, token counts and spans, anchor positions, D's
                                    (appended as it goes; a rerun skips questions already in it)
    generations/generations.csv     the same without positions, one row per question (rewritten after every question)
    activations/sites/<key>.pt      {site: [layers, d_model] float16}
    activations/trajectories/<key>.pt
                                    {"prompt_states": [layers, prompt tokens, d] fp16, "gen_states": [layers, gen
                                    tokens, d] fp16, "prompt_ids", "gen_ids", "think_start", "think_end",
                                    "answer_start", "layers"}
                                    (--traj_layers to keep fewer layers; --no_trajectories to keep only sites)
    activations/answer_dir.pt       (W_U[Yes] - W_U[No]) * final-norm gain [d_model] (logit-lens yes/no direction)
    meta.json                       model, layers, d_model, special tokens, yes/no ids, settings

    python code/extract_reasoning_states.py                      # 16 reviews x 8 questions (~4-5 h on a laptop CPU)
    python code/extract_reasoning_states.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --model_tag r1-distill-qwen-1.5B
"""
import argparse
import csv
import json
import os
import sys
import time

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))     # reasoning_anchors/code
PROJECT = os.path.dirname(HERE)                         # reasoning_anchors
REPO = os.path.dirname(PROJECT)
sys.path.insert(0, os.path.join(REPO, "mechanistic-search"))
from patching_utils import SEED, render_prompt, load_dataset, sample_reviews, set_global_seed  # noqa: E402
from run_role_patching import FRAMINGS, ROLES, YES_STRINGS, NO_STRINGS, role_prefix, get_claims  # noqa: E402
from run_belief_probe import NO_ROLE, ANCHORS, anchor_chars  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_INPUT = os.path.join(REPO, "code/outputs/questions/qwen3_thinking-4B/extracted/"
                                   "cochrane_review_data_final_with_questions.json")
TEMPLATE_DIR = "../mechanistic-search/prompts"   # relative to code/, where render_prompt resolves it
QUESTION_MARKER = "Question: "
YES_NO_FOLLOW_UP = 'Does the treatment in the question work? Answer with only "Yes" or "No".'
DEVICE = "cpu"
PROMPT_SITES = ANCHORS + ["yes_no"]
GEN_SITES = ["think_mean", "think_end", "answer_first", "answer_mean", "answer_last", "gen_mean"]


# Model and prompts
def load_model(name: str, dtype: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype)).to(device)
    model.eval()
    return model, tokenizer


def chat_text(tokenizer, messages: list[dict], thinking: bool) -> str:
    """Chat-template text ending with the assistant turn's start. With thinking off, the turn starts with an empty
    reasoning block, so the next token is the answer (Qwen3: enable_thinking=False; R1-style templates that always open
    <think> get it closed here)."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    if not thinking and text.rstrip().endswith("<think>"):
        text = text.rstrip() + "\n\n</think>\n\n"
    return text


def encode(tokenizer, text: str):
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return torch.tensor([enc["input_ids"]], device=DEVICE), enc


def question_text(claim: str, role: str) -> str:
    return claim if role == NO_ROLE else f"{role_prefix(role)} {claim}"


def single_ids(tokenizer, strings: list[str]) -> list[int]:
    out = []
    for s in strings:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"{s!r} is not a single token: {ids}")
        out.append(ids[0])
    return out


def special_id(tokenizer, token: str) -> int:
    ids = tokenizer.encode(token, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{token!r} is not a single token in this tokenizer ({ids}); is this a reasoning model?")
    return ids[0]


def stop_ids(model, tokenizer) -> set[int]:
    ids = {tokenizer.eos_token_id}
    eos = model.generation_config.eos_token_id
    ids.update(eos if isinstance(eos, list) else [eos])
    vocab = tokenizer.get_vocab()
    ids.update(vocab[t] for t in ["<|im_end|>", "<|eot_id|>", "<｜end▁of▁sentence｜>"] if t in vocab)
    return {i for i in ids if i is not None}


# Recording
class ResidualRecorder:
    """Forward hooks on every decoder layer; each forward pass appends that pass's residuals [layers, seq, d]."""

    def __init__(self, model):
        self.layers = model.model.layers
        self.current, self.passes, self.handles = {}, [], []

    def __enter__(self):
        for i, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_hook(self._hook(i)))
        return self

    def _hook(self, i):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            self.current[i] = h[0].detach().to("cpu", torch.float16)
        return hook

    def commit(self) -> None:
        self.passes.append(torch.stack([self.current[i] for i in range(len(self.layers))]))
        self.current = {}

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def generate_with_states(model, prompt_ids: torch.Tensor, stops: set[int], think_end_ids: list[int],
                         think_end_id: int, max_think: int, max_answer: int) -> dict:
    """Greedy decoding with KV cache and per-step residual recording. Forces </think> after max_think reasoning tokens."""
    with ResidualRecorder(model) as rec:
        out = model(input_ids=prompt_ids, use_cache=True)
        rec.commit()
        past, logits = out.past_key_values, out.logits[0, -1]
        gen, forced, think_end = [], False, None
        pending = []   # tokens to feed regardless of the logits (forced </think>)
        while True:
            if pending:
                next_id = pending.pop(0)
            else:
                next_id = int(logits.argmax())
                if think_end is None and next_id != think_end_id and len(gen) >= max_think:
                    pending, forced = list(think_end_ids), True
                    next_id = pending.pop(0)
            gen.append(next_id)
            if next_id == think_end_id and think_end is None:
                think_end = len(gen) - 1
            n_answer = len(gen) - (think_end + 1) if think_end is not None else 0
            if next_id in stops or n_answer >= max_answer:
                break
            out = model(input_ids=torch.tensor([[next_id]], device=DEVICE), past_key_values=past, use_cache=True)
            rec.commit()
            past, logits = out.past_key_values, out.logits[0, -1]
        prompt_states = rec.passes[0]                                    # [layers, prompt, d]
        gen_states = torch.cat(rec.passes[1:], 1) if len(rec.passes) > 1 else prompt_states[:, :0]
    # gen_states has one column per generated token that was fed back (all but the final stop token)
    return {"gen_ids": gen, "prompt_states": prompt_states, "gen_states": gen_states,
            "think_end": think_end, "forced_think_end": forced}


@torch.no_grad()
def yes_no_d(model, ids: torch.Tensor, yes_ids, no_ids, keep_states: bool = False):
    out = model(input_ids=ids, use_cache=False, output_hidden_states=keep_states)
    log_probs = torch.log_softmax(out.logits[0, -1].float(), -1)
    d = float(torch.logsumexp(log_probs[yes_ids], 0) - torch.logsumexp(log_probs[no_ids], 0))
    states = torch.stack([h[0, -1] for h in out.hidden_states[1:]]).half().cpu() if keep_states else None
    return d, states


def bins(x: torch.Tensor, k: int) -> list[torch.Tensor]:
    """k equal slices along dim 1 of [layers, tokens, d], mean of each (a short span repeats tokens)."""
    n = x.shape[1]
    out = []
    for b in range(k):
        lo = min(int(b * n / k), n - 1)
        hi = max(lo + 1, int((b + 1) * n / k))
        out.append(x[:, lo:hi].float().mean(1).half())
    return out


def gen_sites(gen_states: torch.Tensor, think_start: int, think_end: int | None, answer_start: int | None,
              args) -> dict[str, torch.Tensor]:
    """Generation sites from gen_states [layers, n, d]. think span = (think_start, think_end) exclusive of the tags."""
    n = gen_states.shape[1]
    sites = {}
    if n == 0:
        return sites
    sites["gen_mean"] = gen_states.float().mean(1).half()
    t_end = think_end if think_end is not None else n
    think = gen_states[:, think_start:min(t_end, n)]
    if think.shape[1] > 0:
        sites["think_mean"] = think.float().mean(1).half()
        for k, v in enumerate(bins(think, args.think_bins)):
            sites[f"think_bin_{k}"] = v
    if think_end is not None and think_end < n:
        sites["think_end"] = gen_states[:, think_end]
    if answer_start is not None and answer_start < n:
        answer = gen_states[:, answer_start:]
        sites["answer_first"] = answer[:, 0]
        sites["answer_mean"] = answer.float().mean(1).half()
        sites["answer_last"] = answer[:, -1]
        for k, v in enumerate(bins(answer, args.answer_bins)):
            sites[f"answer_bin_{k}"] = v
    return sites


# Main loop
def run_question(model, tokenizer, review_id, framing, role, claim, ctx, args) -> tuple[dict, dict, dict]:
    question = question_text(claim, role)
    user = render_prompt("question_only", template_dir=TEMPLATE_DIR, question=question)
    text = chat_text(tokenizer, [{"role": "user", "content": user}], thinking=True)
    ids, enc = encode(tokenizer, text)
    q_start = text.index(QUESTION_MARKER + question) + len(QUESTION_MARKER)
    claim_start = q_start + len(question) - len(claim)
    positions = {a: enc.char_to_token(claim_start + c) for a, c in anchor_chars(claim).items()}
    positions["last"] = ids.shape[1] - 1
    if role != NO_ROLE:
        positions["role_start"] = enc.char_to_token(q_start)
    positions["claim_start"] = enc.char_to_token(claim_start)

    t0 = time.time()
    g = generate_with_states(model, ids, ctx["stops"], ctx["think_end_ids"], ctx["think_end_id"],
                             args.max_think_tokens, args.max_answer_tokens)
    seconds = time.time() - t0
    gen = g["gen_ids"]

    # Spans in generated-token indices. The template may already contain <think> (R1-style), else the model writes it.
    think_start = 1 if gen and gen[0] == ctx["think_start_id"] else 0
    think_end = g["think_end"]
    answer_start = None
    if think_end is not None:
        answer_start = think_end + 1
        while answer_start < len(gen) and tokenizer.decode([gen[answer_start]]).strip() == "":
            answer_start += 1
        if answer_start >= len(gen) or gen[answer_start] in ctx["stops"]:
            answer_start = None
    reasoning = tokenizer.decode(gen[think_start:think_end if think_end is not None else len(gen)],
                                 skip_special_tokens=True).strip()
    answer = tokenizer.decode(gen[answer_start:], skip_special_tokens=True).strip() if answer_start is not None else ""

    sites = {a: g["prompt_states"][:, positions[a]] for a in ANCHORS}
    sites |= gen_sites(g["gen_states"], think_start, think_end, answer_start, args)
    sites = {k: v.clone() for k, v in sites.items()}   # views would save their whole parent tensor

    yn_user = render_prompt("question_only_yes_no", template_dir=TEMPLATE_DIR, question=question)
    yn_ids, _ = encode(tokenizer, chat_text(tokenizer, [{"role": "user", "content": yn_user}], thinking=False))
    d_nothink, sites["yes_no"] = yes_no_d(model, yn_ids, ctx["yes_ids"], ctx["no_ids"], keep_states=True)
    follow = [{"role": "user", "content": user}, {"role": "assistant", "content": answer or reasoning[-2000:]},
              {"role": "user", "content": YES_NO_FOLLOW_UP}]
    fu_ids, _ = encode(tokenizer, chat_text(tokenizer, follow, thinking=False))
    d_after, _ = yes_no_d(model, fu_ids, ctx["yes_ids"], ctx["no_ids"])

    key = f"{review_id}_{framing}_{role}"
    row = {"key": key, "review_id": review_id, "framing": framing, "role": role, "question": question,
           "prompt_text": text, "generation": tokenizer.decode(gen, skip_special_tokens=False),
           "reasoning": reasoning, "answer": answer, "n_prompt": ids.shape[1], "n_gen": len(gen),
           "n_think": (think_end if think_end is not None else len(gen)) - think_start,
           "n_answer": len(gen) - answer_start if answer_start is not None else 0,
           "think_start": think_start, "think_end": think_end, "answer_start": answer_start,
           "forced_think_end": g["forced_think_end"], "ended": bool(gen) and gen[-1] in ctx["stops"],
           "positions": positions, "d_nothink": d_nothink, "d_after": d_after, "seconds": round(seconds, 1),
           "tok_per_s": round(len(gen) / max(seconds, 1e-6), 2), "sites": sorted(sites)}
    traj = {"prompt_states": g["prompt_states"][args.traj_layers], "gen_states": g["gen_states"][args.traj_layers],
            "prompt_ids": ids[0].tolist(), "gen_ids": gen, "think_start": think_start, "think_end": think_end,
            "answer_start": answer_start, "layers": args.traj_layers}
    return row, sites, traj


CSV_FIELDS = ["key", "review_id", "framing", "role", "question", "reasoning", "answer", "n_think", "n_answer",
              "forced_think_end", "ended", "d_nothink", "d_after", "seconds", "tok_per_s"]


def write_generations_csv(jsonl_path: str, csv_path: str) -> None:
    with open(jsonl_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main(args):
    set_global_seed(args.seed)
    torch.set_num_threads(args.threads or torch.get_num_threads())
    out_dir = args.output_dir or os.path.join(PROJECT, "outputs", args.model_tag)
    act_dir, gen_dir = os.path.join(out_dir, "activations"), os.path.join(out_dir, "generations")
    for sub in [os.path.join(act_dir, "sites"), os.path.join(act_dir, "trajectories"), gen_dir]:
        os.makedirs(sub, exist_ok=True)

    dataset = sample_reviews(load_dataset(args.input_path), args.num_reviews, seed=args.seed)
    if args.review_ids:
        dataset = [e for e in load_dataset(args.input_path) if e["ReviewID"] in set(args.review_ids)]

    global DEVICE
    DEVICE = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args.model, args.dtype, DEVICE)
    num_layers = len(model.model.layers)
    args.traj_layers = args.traj_layers or list(range(num_layers))
    ctx = {"stops": stop_ids(model, tokenizer), "think_start_id": special_id(tokenizer, "<think>"),
           "think_end_id": special_id(tokenizer, "</think>"),
           "think_end_ids": tokenizer.encode("\n</think>\n\n", add_special_tokens=False),
           "yes_ids": single_ids(tokenizer, YES_STRINGS), "no_ids": single_ids(tokenizer, NO_STRINGS)}
    w_u = model.lm_head.weight.float()
    answer_dir = (w_u[ctx["yes_ids"]].mean(0) - w_u[ctx["no_ids"]].mean(0)) * model.model.norm.weight.float()
    torch.save(answer_dir.detach().cpu(), os.path.join(act_dir, "answer_dir.pt"))
    meta = {"model": args.model, "model_tag": args.model_tag, "num_layers": num_layers,
            "d_model": model.config.hidden_size, "roles": args.roles, "framings": FRAMINGS,
            "prompt_sites": PROMPT_SITES, "gen_sites": GEN_SITES, "think_bins": args.think_bins,
            "answer_bins": args.answer_bins, "max_think_tokens": args.max_think_tokens,
            "max_answer_tokens": args.max_answer_tokens, "traj_layers": args.traj_layers, "dtype": args.dtype,
            "yes_ids": ctx["yes_ids"], "no_ids": ctx["no_ids"], "think_end_id": ctx["think_end_id"],
            "review_ids": [e["ReviewID"] for e in dataset], "seed": args.seed}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    rows_path = os.path.join(gen_dir, "generations.jsonl")
    done = set()
    if os.path.exists(rows_path):
        with open(rows_path) as f:
            done = {json.loads(line)["key"] for line in f if line.strip()}
        print(f"Resuming: {len(done)} questions already in {rows_path}")

    todo = [(e, f, r) for e in dataset for f in FRAMINGS for r in [NO_ROLE] + args.roles
            if f"{e['ReviewID']}_{f}_{r}" not in done]
    print(f"{model.config.num_hidden_layers} layers, d_model {model.config.hidden_size}; "
          f"{len(dataset)} reviews, {len(todo)} questions to run")
    bar = tqdm(todo, desc="questions")
    for example, framing, role in bar:
        claim = get_claims(example)[framing]
        row, sites, traj = run_question(model, tokenizer, example["ReviewID"], framing, role, claim, ctx, args)
        torch.save(sites, os.path.join(act_dir, "sites", f"{row['key']}.pt"))
        if not args.no_trajectories:
            torch.save(traj, os.path.join(act_dir, "trajectories", f"{row['key']}.pt"))
        with open(rows_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        write_generations_csv(rows_path, os.path.join(gen_dir, "generations.csv"))
        bar.set_postfix(n_gen=row["n_gen"], tok_s=row["tok_per_s"], forced=row["forced_think_end"],
                        d_after=f"{row['d_after']:+.1f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Prompt and generation residuals of a reasoning model, framing x role")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--model_tag", default="qwen3-0.6B")
    p.add_argument("--input_path", default=DEFAULT_INPUT)
    p.add_argument("--output_dir", default=None, help="default: reasoning_anchors/outputs/<model_tag>")
    p.add_argument("--num_reviews", default=16, type=int, help="seeded sample of reviews (-1: all 368)")
    p.add_argument("--review_ids", nargs="*", default=None, help="run these reviews instead of the sample")
    p.add_argument("--roles", nargs="+", default=ROLES)
    p.add_argument("--max_think_tokens", default=512, type=int, help="reasoning budget before </think> is forced")
    p.add_argument("--max_answer_tokens", default=192, type=int)
    p.add_argument("--think_bins", default=10, type=int)
    p.add_argument("--answer_bins", default=4, type=int)
    p.add_argument("--traj_layers", nargs="*", type=int, default=None, help="layers kept in trajectories/ (default all)")
    p.add_argument("--no_trajectories", action="store_true", help="save only the sites, not every token")
    p.add_argument("--dtype", default="float32", help="float32 on CPU; bfloat16 on a GPU")
    p.add_argument("--device", default="auto", help="auto: cuda if available, else cpu")
    p.add_argument("--threads", default=0, type=int)
    p.add_argument("--seed", default=SEED, type=int)
    main(p.parse_args())
