"""
Shared utilities for activation patching positive-framing activations into negative-framing runs.

Setup
-----
For one (review, question type) pair we build two prompts that differ only in the framing of the question
(e.g. "How effective is X for Y?" vs "How ineffective is X for Y?"), using the same `question_answering`
prompt and chat template as the main pipeline (code/generate_framing_responses.py).

Both prompts end with the same chat-template suffix (`<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n`),
so the last token before generation is the same token in both runs. We cache the residual stream (output of each
decoder layer) at that last position in the POSITIVE run, and overwrite the residual stream at the last position of
the NEGATIVE run at a single layer L with it. Everything after layer L at that position (and every later token, which
attends to it) is then computed from the patched state.

The same machinery is used for other contrasts (see CONTRASTS below), e.g. "My doctor said <sentence>" -> "<sentence>".
"""
import os
import sys
import random
import logging
from contextlib import contextmanager

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse the prompt rendering from the main pipeline so prompts are identical to the framing experiments
CODE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code")
sys.path.insert(0, CODE_DIR)
from utils import render_prompt, load_json_file, load_jsonl_file  # noqa: E402

PROMPT_TEMPLATE_NAME = "question_answering"
DEFAULT_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
SEED = 42


def set_global_seed(seed_value: int = SEED) -> None:
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


# Model loading
REPO_DIR = os.path.dirname(CODE_DIR)
ENV_FILE_PATHS = list(dict.fromkeys(  # dedupe: on Explorer the repo's parent is the home directory
    [os.path.join(REPO_DIR, ".env"), os.path.join(os.path.dirname(REPO_DIR), ".env"), os.path.expanduser("~/.env")]
))


def get_hf_token() -> str | None:
    """
    HF token from the repo .env, the .env one level above the repo, or ~/.env. HF_TOKEN is preferred over
    HUGGINGFACE_TOKEN. The first valid token (checked with whoami) is used.
    """
    from huggingface_hub import whoami

    for path in ENV_FILE_PATHS:
        if os.path.isfile(path):
            load_dotenv(path, override=False)
    print(f".env files loaded: {', '.join(p for p in ENV_FILE_PATHS if os.path.isfile(p)) or 'none'}")

    # Candidates: env vars (any capitalization, e.g. HF_token), then tokens saved by `huggingface-cli login`
    # (the job sets HF_HOME to scratch, so the default ~/.cache/huggingface/token would otherwise be missed)
    candidates = []
    for name in ["HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"]:
        for var, value in os.environ.items():
            if var.upper() == name and value.strip():
                candidates.append((f"${var}", value.strip()))
    for path in [os.path.join(os.getenv("HF_HOME", ""), "token"), os.path.expanduser("~/.cache/huggingface/token")]:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                candidates.append((path, f.read().strip()))

    for source, token in candidates:
        try:
            print(f"Using HF token from {source} (HF user: {whoami(token=token)['name']})")
            return token
        except Exception as e:
            print(f"HF token from {source} is invalid ({type(e).__name__}); trying the next one")
    print("No valid HF token found")
    return None


def load_model_and_tokenizer(model_name: str = DEFAULT_MODEL_NAME):
    logging.basicConfig(level=logging.ERROR)
    access_token = get_hf_token()

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", dtype=torch.bfloat16, token=access_token,
    )
    model.eval()

    print(f"Model:            {model_name}")
    print(f"Model's dtype:    {model.dtype}")
    print(f"Model's device:   {model.device}")
    print(f"Number of layers: {len(get_decoder_layers(model))}")
    offloaded = {k: v for k, v in (getattr(model, "hf_device_map", None) or {}).items() if v in ("cpu", "disk")}
    if offloaded:
        print(f"WARNING: {len(offloaded)} modules offloaded to CPU/disk (not enough free GPU memory); runs will be very slow: {offloaded}")
    print()
    return model, tokenizer


def get_decoder_layers(model) -> torch.nn.ModuleList:
    """Decoder blocks for Llama-style HF models."""
    return model.model.layers


def get_context_length(model) -> int:
    return model.config.max_position_embeddings


def get_stop_token_ids(tokenizer) -> list[int]:
    """Llama 3 instruct ends assistant turns with <|eot_id|>, not the tokenizer's eos token."""
    stop_ids = [tokenizer.eos_token_id]
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id is not None and eot_id != tokenizer.unk_token_id:
        stop_ids.append(eot_id)
    return stop_ids


# Data / prompts
def format_rct_inputs(rct_inputs: list[dict]) -> list[dict]:
    """Same as Generator.__format_rct_inputs in code/generate_framing_responses.py."""
    return [{"title": rct["Title"], "abstract": rct["Abstract"]} for rct in rct_inputs]


def build_prompt_ids(tokenizer, question: str, rct_inputs: list[dict], device) -> torch.Tensor:
    """Render the question-answering prompt and apply the chat template (with generation prompt). Returns [1, seq_len]."""
    input_text = render_prompt(PROMPT_TEMPLATE_NAME, template_dir="./prompts", question=question, abstracts=rct_inputs)
    messages = [{"role": "user", "content": input_text}]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    if not isinstance(input_ids, torch.Tensor):  # newer transformers can return a BatchEncoding
        input_ids = input_ids["input_ids"]
    return input_ids.to(device)


def load_dataset(input_path: str) -> list[dict]:
    if input_path.endswith(".jsonl"):
        return load_jsonl_file(input_path)
    return load_json_file(input_path)


def sample_reviews(dataset: list[dict], num_reviews: int | None, seed: int = SEED) -> list[dict]:
    """Deterministic subset of reviews so the sweep and generation stages see the same examples."""
    if num_reviews is None or num_reviews >= len(dataset):
        return dataset
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(dataset)), num_reviews))
    return [dataset[i] for i in indices]


def get_question_types(example: dict, requested: list[str] | None) -> list[str]:
    """All single-turn question types (multiturn is skipped: there is no single 'last token before generation')."""
    q_types = [k for k in example["Questions"] if k != "multiturn"]
    if requested:
        q_types = [k for k in q_types if k in requested]
    return q_types


# Contrasts: which two prompts are compared. The SOURCE prompt's residual stream is cached and patched into the TARGET
# prompt. Pair records keep the original field names, so positive_* = source and negative_* = target.
#   framing: source = positive framing, target = negative framing (same question type)
#   doctor:  source = "My doctor said <sentence>", target = "<sentence>", where <sentence> is a claim-style question
#            with its attribution removed (e.g. "AI said that X works well for Y. Is this true?" ->
#            "X works well for Y. Is this true?"). Done for both the positive and negative framing of each claim.
CONTRASTS = ["framing", "doctor"]
DOCTOR_PREFIX = "My doctor said"
ATTRIBUTION_PREFIXES = {  # claim-style question types (code/prompts/question_templates.json) and their attribution
    "doctor": "My doctor says ",
    "ai": "AI said that ",
    "testimonials": "I read online testimonials that ",
    "journals": "I read online journal articles that ",
    "friend": "My friend said ",
    "family": "My family said ",
}


def strip_attribution(question: str, q_type: str) -> str:
    prefix = ATTRIBUTION_PREFIXES[q_type]
    if not question.startswith(prefix):
        raise ValueError(f"'{q_type}' question does not start with '{prefix}': {question}")
    return question[len(prefix):]


def get_prompt_pairs(example: dict, contrast: str, requested: list[str] | None) -> list[dict]:
    """[{question_key, source_question, target_question}] for one review under the given contrast."""
    if contrast == "framing":
        return [
            {
                "question_key": q_key,
                "source_question": example["Questions"][q_key]["positive_question"],
                "target_question": example["Questions"][q_key]["negative_question"],
            }
            for q_key in get_question_types(example, requested)
        ]

    if contrast == "doctor":
        q_types = [k for k in ATTRIBUTION_PREFIXES if k in example["Questions"]]
        if requested:
            q_types = [k for k in q_types if k in requested]
        # ai/testimonials/journals (and friend/family) share a sentence once the attribution is removed; keep the first
        pairs, seen = [], set()
        for q_type in q_types:
            for framing in ["positive", "negative"]:
                sentence = strip_attribution(example["Questions"][q_type][f"{framing}_question"], q_type)
                if sentence in seen:
                    continue
                seen.add(sentence)
                pairs.append({
                    "question_key": f"{q_type}_{framing}",
                    "source_question": f"{DOCTOR_PREFIX} {sentence}",
                    "target_question": sentence,
                })
        return pairs

    raise ValueError(f"Unknown contrast '{contrast}' (options: {CONTRASTS})")


# Caching and patching
@contextmanager
def cache_residuals(model, positions: list[int], layer_indices: list[int] | None = None, device=None):
    """
    Context manager that records the residual stream (decoder layer output) at `positions` for every layer (or only
    `layer_indices`), optionally moved to `device` (e.g. "cpu" to keep long caches off the GPU).
    Yields a dict {layer_idx: tensor [len(positions), d_model]} that is filled during the forward pass.
    """
    layers = get_decoder_layers(model)
    cache = {}
    handles = []

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            values = hidden_states[0, positions, :].detach()
            cache[layer_idx] = values.to(device) if device is not None else values.clone()
        return hook

    for layer_idx, layer in enumerate(layers):
        if layer_indices is None or layer_idx in layer_indices:
            handles.append(layer.register_forward_hook(make_hook(layer_idx)))
    try:
        yield cache
    finally:
        for h in handles:
            h.remove()


@contextmanager
def patch_residual(model, layer_idx: int, positions: list[int], patch_values: torch.Tensor):
    """
    Context manager that overwrites the residual stream output of decoder layer `layer_idx` at `positions`
    with `patch_values` ([len(positions), d_model]).

    The patch is only applied on forward passes that contain those positions (the prefill pass), so it works both for
    a single teacher-forced forward pass and for model.generate with KV caching (decode steps have seq_len 1 and are
    left untouched; they see the patch through the cached keys/values of later layers).
    """
    layer = get_decoder_layers(model)[layer_idx]
    max_position = max(positions)

    def hook(module, inputs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        if hidden_states.shape[1] > max_position:
            hidden_states[0, positions, :] = patch_values.to(device=hidden_states.device, dtype=hidden_states.dtype)
        return output

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def last_positions(seq_len: int, num_last_tokens: int) -> list[int]:
    """Indices of the last `num_last_tokens` prompt tokens (these are identical chat-template tokens in both framings)."""
    return list(range(seq_len - num_last_tokens, seq_len))


# Forward passes
@torch.no_grad()
def get_positive_cache(model, pos_ids: torch.Tensor, num_last_tokens: int) -> dict[int, torch.Tensor]:
    positions = last_positions(pos_ids.shape[1], num_last_tokens)
    with cache_residuals(model, positions) as cache:
        model.model(input_ids=pos_ids, use_cache=False)
    return dict(cache)


@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    """Greedy decoding (deterministic so clean vs. patched differences are due to the patch). Returns new token ids [n]."""
    output = model.generate(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        eos_token_id=get_stop_token_ids(tokenizer),
        pad_token_id=tokenizer.eos_token_id,
    )
    return output[0, input_ids.shape[1]:]


@torch.no_grad()
def response_logprobs(model, prompt_ids: torch.Tensor, response_ids: torch.Tensor,
                      prompt_attention_mask: torch.Tensor | None = None,
                      prompt_position_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Teacher-forced pass over prompt + response.

    prompt_attention_mask / prompt_position_ids ([1, prompt_len]) are optional: a 0 in the mask hides that prompt token
    from every later token, and the response continues the prompt's position ids (see masked_prompt_inputs).

    :return: (per-token log-probs of the response tokens [n_resp], log-probs of the next-token distribution at the
              last prompt position [vocab]) -- both float32 on CPU.
    """
    prompt_len = prompt_ids.shape[1]
    n_resp = response_ids.shape[0]
    full_ids = torch.cat([prompt_ids, response_ids.unsqueeze(0).to(prompt_ids.device)], dim=1)

    kwargs = {}
    if prompt_attention_mask is not None:
        kwargs["attention_mask"] = torch.cat([prompt_attention_mask, prompt_attention_mask.new_ones(1, n_resp)], dim=1)
    if prompt_position_ids is not None:
        response_positions = prompt_position_ids[:, -1:] + 1 + torch.arange(n_resp, device=prompt_position_ids.device)
        kwargs["position_ids"] = torch.cat([prompt_position_ids, response_positions], dim=1)

    # Only project the positions we need through the LM head (avoids a [seq_len, vocab] logits tensor)
    hidden = model.model(input_ids=full_ids, use_cache=False, **kwargs).last_hidden_state
    # No manual .to(): with device_map="auto" accelerate moves inputs to lm_head's device, and an offloaded lm_head's
    # weight sits on the meta device
    hidden = hidden[:, prompt_len - 1: full_ids.shape[1] - 1, :]
    log_probs = torch.log_softmax(model.lm_head(hidden).float(), dim=-1)[0]  # [n_resp, vocab]

    targets = response_ids.to(log_probs.device)
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.cpu(), log_probs[0].cpu()


def masked_prompt_inputs(prompt_ids: torch.Tensor, masked_positions: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Attention mask that hides `masked_positions` from every later token, and position ids that skip them, so the visible
    tokens get exactly the positions they would have if the masked tokens were not in the prompt at all.
    """
    attention_mask = torch.ones_like(prompt_ids)
    attention_mask[0, masked_positions] = 0
    position_ids = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)
    return attention_mask, position_ids


def kl_divergence(log_p: torch.Tensor, log_q: torch.Tensor) -> float:
    """KL(p || q) for log-prob vectors."""
    return torch.sum(log_p.exp() * (log_p - log_q)).item()
