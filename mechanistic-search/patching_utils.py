"""
Shared utilities for activation patching: model loading, prompt rendering, residual-stream caching/patching, and
greedy / teacher-forced passes. The older, larger version used by the archived experiments is in archive/.
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


def load_model_and_tokenizer(model_name: str = DEFAULT_MODEL_NAME, attn_implementation: str | None = None):
    """attn_implementation: None for the transformers default (sdpa), "eager" to get attention weights back."""
    logging.basicConfig(level=logging.ERROR)
    access_token = get_hf_token()

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    tokenizer.pad_token = tokenizer.eos_token

    # bfloat16 needs compute capability 8.0+ (A100 and newer); older GPUs (V100, T4) only emulate it, so use float16
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    extra_kwargs = {"attn_implementation": attn_implementation} if attn_implementation else {}
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", dtype=torch.bfloat16 if use_bf16 else torch.float16, token=access_token,
        **extra_kwargs,
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
    """Decoder blocks for Llama-style HF models (Llama, OLMo 2/3)."""
    return model.model.layers


def get_context_length(model) -> int:
    return model.config.max_position_embeddings


def get_stop_token_ids(model, tokenizer) -> list[int]:
    """
    Ids that end an assistant turn: the tokenizer's and generation config's eos, plus the chat end-of-turn token
    (Llama 3 ends turns with <|eot_id|> and OLMo 3 with <|im_end|>, neither of which is the tokenizer's eos token).
    """
    stop_ids = {tokenizer.eos_token_id}
    config_eos = model.generation_config.eos_token_id
    stop_ids.update(config_eos if isinstance(config_eos, list) else [config_eos])
    vocab = tokenizer.get_vocab()
    stop_ids.update(vocab[t] for t in ["<|eot_id|>", "<|im_end|>"] if t in vocab)
    return sorted(i for i in stop_ids if i is not None)


# Data
def format_rct_inputs(rct_inputs: list[dict]) -> list[dict]:
    """Same as Generator.__format_rct_inputs in code/generate_framing_responses.py."""
    return [{"title": rct["Title"], "abstract": rct["Abstract"]} for rct in rct_inputs]


def load_dataset(input_path: str) -> list[dict]:
    if input_path.endswith(".jsonl"):
        return load_jsonl_file(input_path)
    return load_json_file(input_path)


def sample_reviews(dataset: list[dict], num_reviews: int | None, seed: int = SEED) -> list[dict]:
    """Deterministic subset of reviews, so every run sees the same examples."""
    if num_reviews is None or num_reviews < 0 or num_reviews >= len(dataset):
        return dataset
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(dataset)), num_reviews))
    return [dataset[i] for i in indices]


# Caching and patching
@contextmanager
def cache_residuals(model, positions: list[int], layer_indices: list[int] | None = None):
    """
    Records the residual stream (decoder layer output) at `positions` for every layer (or only `layer_indices`).
    Yields a dict {layer_idx: tensor [len(positions), d_model]} that is filled during the forward pass.
    """
    layers = get_decoder_layers(model)
    cache = {}
    handles = []

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            cache[layer_idx] = hidden_states[0, positions, :].detach().clone()
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
    Overwrites the residual stream output of decoder layer `layer_idx` at `positions` with `patch_values`
    ([len(positions), d_model]).

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


# Forward passes
@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    """Greedy decoding (deterministic, so clean vs. patched differences are due to the patch). Returns new token ids [n]."""
    output = model.generate(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        eos_token_id=get_stop_token_ids(model, tokenizer),
        pad_token_id=tokenizer.eos_token_id,
    )
    return output[0, input_ids.shape[1]:]


@torch.no_grad()
def patched_generate_every_token(model, tokenizer, src_ids: torch.Tensor, tgt_ids: torch.Tensor, layer_idx: int,
                                 src_positions: list[int], tgt_positions: list[int], max_new_tokens: int) -> torch.Tensor:
    """
    Greedy decoding of the target prompt with the source run's residual stream written in at decoder layer `layer_idx`:
    at `tgt_positions` of the prompt (from `src_positions`) and at every generated token. The source prompt is decoded
    in lockstep on the target's generated tokens, so each new token's patch comes from the source run at that same token.
    Returns new token ids [n] (including the stop token, like greedy_generate).
    """
    layer = get_decoder_layers(model)[layer_idx]
    stop_ids = set(get_stop_token_ids(model, tokenizer))
    state = {"mode": None, "positions": None, "values": None}

    def hook(module, inputs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        if state["mode"] == "record":
            state["values"] = hidden_states[0, state["positions"], :].detach().clone()
        elif state["mode"] == "patch":
            hidden_states[0, state["positions"], :] = state["values"].to(device=hidden_states.device, dtype=hidden_states.dtype)
        return output

    def step(input_ids, past_key_values, mode, positions):
        state["mode"], state["positions"] = mode, positions
        return model.model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)

    handle = layer.register_forward_hook(hook)
    try:
        src_out = step(src_ids, None, "record", src_positions)   # source first: its values are patched into the target
        tgt_out = step(tgt_ids, None, "patch", tgt_positions)
        new_ids = []
        for _ in range(max_new_tokens):
            next_id = int(model.lm_head(tgt_out.last_hidden_state[:, -1, :]).argmax(dim=-1))
            new_ids.append(next_id)
            if next_id in stop_ids:
                break
            token = torch.tensor([[next_id]], device=tgt_ids.device)
            src_out = step(token, src_out.past_key_values, "record", [0])
            tgt_out = step(token, tgt_out.past_key_values, "patch", [0])
    finally:
        handle.remove()
    return torch.tensor(new_ids, device=tgt_ids.device)


@torch.no_grad()
def response_logprob(model, prompt_ids: torch.Tensor, response_ids: torch.Tensor) -> float:
    """Teacher-forced log p(response | prompt), summed over the response tokens."""
    prompt_len = prompt_ids.shape[1]
    full_ids = torch.cat([prompt_ids, response_ids.unsqueeze(0).to(prompt_ids.device)], dim=1)
    # Only project the positions we need through the LM head (avoids a [seq_len, vocab] logits tensor)
    hidden = model.model(input_ids=full_ids, use_cache=False).last_hidden_state
    hidden = hidden[:, prompt_len - 1: full_ids.shape[1] - 1, :]
    log_probs = torch.log_softmax(model.lm_head(hidden).float(), dim=-1)[0]  # [n_resp, vocab]
    targets = response_ids.to(log_probs.device)
    return log_probs.gather(-1, targets.unsqueeze(-1)).sum().item()


@torch.no_grad()
def yes_no_logit_diff(model, prompt_ids: torch.Tensor, yes_ids: list[int], no_ids: list[int]) -> tuple[float, int]:
    """(log p(Yes) - log p(No), top-1 token id) at the last prompt position, with the listed token variants pooled."""
    hidden = model.model(input_ids=prompt_ids, use_cache=False).last_hidden_state[:, -1:, :]
    log_probs = torch.log_softmax(model.lm_head(hidden).float(), dim=-1)[0, 0]
    d = (torch.logsumexp(log_probs[yes_ids], 0) - torch.logsumexp(log_probs[no_ids], 0)).item()
    return d, int(log_probs.argmax())
