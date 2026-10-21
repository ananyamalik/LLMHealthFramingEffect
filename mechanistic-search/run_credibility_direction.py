"""
Credibility direction: does one residual-stream direction at the source token order sources by how much the model
agrees with them, and does adding it to "My friend said, ..." runs raise agreement steadily (dose-response) for both
claim polarities?

Prompts: the Yes/No readout prompt (prompts/question_answering_yes_no.jinja2, same as run_yes_no_patching.py) with the
claim attributed to one of SOURCES, in each of FRAMES:
    said:       "My doctor said, <claim>"      "A stranger said, <claim>"     "An AI said, <claim>"
    according:  "According to my doctor, <claim>"
plus the bare "<claim>" (no source). The claims are the doctor contrast's claims (patching_utils.get_prompt_pairs with
contrast "doctor"): 3 distinct sentences per review, each in a positive ("will work") and negative ("won't work")
polarity. The "said" frame with "my doctor" / "my friend" is exactly the doctor_friend Yes/No prompt.

Agreement (signed so it means "agrees with the source" for both polarities):
    D(prompt) = log p(Yes) - log p(No) at the last prompt position
    A(prompt) = D for positive claims, -D for negative claims
    shift(source) = A(source prompt) - A(bare claim)

Stages (--stages, run in order):
    behavior   shift per source / frame / polarity, plus the residual stream at the source token (last token of the
               source phrase, e.g. " doctor", " AI") at every layer. The source token comes before the claim, so with
               causal attention its activation is the same for every claim and review; it is averaged over a few
               prompts and the largest deviation is logged as a check.
    direction  per layer, with activations centered over sources within each frame:
                   doctor_minus_friend  x_doctor - x_friend
                   pls_all              sum_s (b_s - mean b) (x_s - mean x), b = behavioral shift (1-component PLS)
                   pls_heldout          pls_all fit without doctor and friend (the steering default: both ends of the
                                        steering test are out of sample)
                   pc1                  first principal component of the source activations (no behavior used; sign
                                        set so doctor projects above friend)
               and tests whether the projection orders the sources like their behavioral shift (Spearman rho):
                   rho_all             all sources
                   rho_without_doctor  without doctor (is it more than "doctor vs everything else"?)
                   rho_unseen          sources the direction was not fit on (doctor_minus_friend: the other 6)
                   rho_loo             pls_all, leave one source out: predict the held-out source's shift from its
                                       projection on a direction fit to the others (permutation p over shuffled shifts)
                   rho_cross_frame     pls fit on one frame, tested on the other
               The credibility coordinate (proj_s - proj_friend) / (proj_doctor - proj_friend) puts each source on a
               friend = 0, doctor = 1 scale; "a stranger" should fall at or below friend.
    steer      adds alpha * gap_L * v_L to the residual stream at the friend prompt's source token at layer L, where
               v_L is the unit direction and gap_L = (x_doctor - x_friend) . v_L, so alpha = 1 moves friend's
               projection to doctor's. Matched-norm random directions are the control. A causal, claim-independent
               credibility representation predicts A rising steadily with alpha for both polarities, and not for the
               random directions.

Outputs (in --output_dir):
    credibility_behavior.jsonl         one line per claim: readouts for the bare claim and every source x frame
    source_activations.pt              [frames, sources, layers, d_model] activations at the source token
    credibility_behavior_summary.json
    credibility_directions.pt          unit directions [layers, d_model] and doctor-friend gaps per direction
    credibility_direction_summary.json
    credibility_steering.jsonl         one line per (claim, steered source, frame): D for every direction/layer/alpha
    credibility_steering_summary.json
    credibility_behavior.png, credibility_direction.png, credibility_steering.png
"""
import argparse
import itertools
import json
import os
import gc

import numpy as np
import torch
from tqdm import tqdm

from patching_utils import (
    DEFAULT_MODEL_NAME, SEED, set_global_seed, load_model_and_tokenizer, get_decoder_layers, format_rct_inputs,
    build_prompt_ids, load_dataset, sample_reviews, get_prompt_pairs, cache_residuals, steer_residual,
)
from run_layer_sweep import aggregate
from run_comma_patching import QUESTION_MARKER
from run_yes_no_patching import TEMPLATE_NAME, TEMPLATE_DIR, YES_STRINGS, NO_STRINGS, single_token_ids, readout

from utils import render_prompt  # noqa: E402  (code/ is on sys.path via patching_utils)

BEHAVIOR_FILE_NAME = "credibility_behavior.jsonl"
ACTIVATIONS_FILE_NAME = "source_activations.pt"
BEHAVIOR_SUMMARY_FILE_NAME = "credibility_behavior_summary.json"
DIRECTIONS_FILE_NAME = "credibility_directions.pt"
DIRECTION_SUMMARY_FILE_NAME = "credibility_direction_summary.json"
STEERING_FILE_NAME = "credibility_steering.jsonl"
STEERING_SUMMARY_FILE_NAME = "credibility_steering_summary.json"
STAGES = ["behavior", "direction", "steer"]

SOURCES = ["doctor", "pharmacist", "nurse", "friend", "neighbor", "stranger", "website", "ai"]
# natural: how people usually say it (personal relationships take "my"); indefinite: the same determiner for every
# source, to check that the ordering is not just "my" vs "a"
SOURCE_PHRASES = {
    "natural": {
        "doctor": "my doctor", "pharmacist": "my pharmacist", "nurse": "my nurse", "friend": "my friend",
        "neighbor": "my neighbor", "stranger": "a stranger", "website": "a website", "ai": "an AI",
    },
    "indefinite": {
        "doctor": "a doctor", "pharmacist": "a pharmacist", "nurse": "a nurse", "friend": "a friend",
        "neighbor": "a neighbor", "stranger": "a stranger", "website": "a website", "ai": "an AI",
    },
}
FRAMES = {  # text before and after the source phrase; the claim follows
    "said": ("", " said, "),
    "according": ("According to ", ", "),
}
NO_SOURCE = "none"
HIGH_END, LOW_END = "doctor", "friend"   # endpoints of the credibility coordinate and of the steering scale
DIRECTION_NAMES = ["doctor_minus_friend", "pls_all", "pls_heldout", "pc1"]


# Prompts
def get_claims(example: dict) -> list[dict]:
    """[{claim_key, polarity, claim}] -- the doctor contrast's bare sentences (3 distinct claims x 2 polarities)."""
    return [
        {"claim_key": p["question_key"], "polarity": p["question_key"].rsplit("_", 1)[1], "claim": p["target_question"]}
        for p in get_prompt_pairs(example, "doctor", None)
    ]


def attributed_question(claim: str, source: str, frame: str, determiner: str) -> tuple[str, tuple[int, int]]:
    """The question with the claim attributed to `source`, and the character span of the source phrase in it."""
    if source == NO_SOURCE:
        return claim, None
    before, after = FRAMES[frame]
    phrase = SOURCE_PHRASES[determiner][source]
    if not before:
        phrase = phrase[0].upper() + phrase[1:]
    return f"{before}{phrase}{after}{claim}", (len(before), len(before) + len(phrase))


def prompt_key(source: str, frame: str) -> str:
    return NO_SOURCE if source == NO_SOURCE else f"{frame}:{source}"


def encode_with_source(tokenizer, question: str, source_span: tuple[int, int] | None, rct_inputs: list[dict], device):
    """
    Prompt ids and the source phrase's token positions ({first, last, last_token, phrase_tokens, straddles}). The
    offset tokenization is checked against build_prompt_ids so positions match the runs exactly.
    """
    input_ids = build_prompt_ids(tokenizer, question, rct_inputs, device, template_name=TEMPLATE_NAME, template_dir=TEMPLATE_DIR)
    if source_span is None:
        return input_ids, None

    input_text = render_prompt(TEMPLATE_NAME, template_dir=TEMPLATE_DIR, question=question, abstracts=rct_inputs)
    text = tokenizer.apply_chat_template([{"role": "user", "content": input_text}], tokenize=False, add_generation_prompt=True)
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if encoding["input_ids"] != input_ids[0].tolist():
        raise ValueError("Offset tokenization does not match the chat-template ids")

    marker_index = text.find(QUESTION_MARKER + question)
    if marker_index == -1:
        raise ValueError(f"Question not found in the rendered prompt: {question}")
    start = marker_index + len(QUESTION_MARKER) + source_span[0]
    end = marker_index + len(QUESTION_MARKER) + source_span[1]
    first, last = encoding.char_to_token(start), encoding.char_to_token(end - 1)
    ids = encoding["input_ids"]
    return input_ids, {
        "first": first,
        "last": last,
        "last_token": tokenizer.decode([ids[last]]),
        "phrase_tokens": tokenizer.decode(ids[first:last + 1]),
        "straddles": encoding.token_to_chars(last).end > end,   # last token runs into the text after the phrase
    }


def agreement(d: float, polarity: str) -> float:
    return d if polarity == "positive" else -d


def load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_done(path: str) -> set[str]:
    return {r["pair_key"] for r in load_jsonl(path)}


# Stage 1: behavior + activations
def run_behavior(model, tokenizer, dataset, args, yes_ids, no_ids, behavior_path: str) -> None:
    done = load_done(behavior_path)
    print(f"Behavior: claims already done (resuming): {len(done)}")
    prompt_specs = [(NO_SOURCE, None)] + [(s, f) for f in args.frames for s in args.sources]

    with open(behavior_path, "a", encoding="utf-8") as out_file:
        for example in tqdm(dataset, desc="Behavior: reviews"):
            rct_inputs = format_rct_inputs(example["Inputs"])
            for claim in get_claims(example):
                key = f"{example['ReviewID']}_{claim['claim_key']}"
                if key in done:
                    continue
                readouts = {}
                for source, frame in prompt_specs:
                    question, _ = attributed_question(claim["claim"], source, frame, args.determiner)
                    ids = build_prompt_ids(tokenizer, question, rct_inputs, model.device, template_name=TEMPLATE_NAME, template_dir=TEMPLATE_DIR)
                    r = readout(model, ids, yes_ids, no_ids)
                    r["agreement"] = agreement(r["d"], claim["polarity"])
                    readouts[prompt_key(source, frame)] = r
                record = {
                    "pair_key": key,
                    "review_id": example["ReviewID"],
                    "claim_key": claim["claim_key"],
                    "polarity": claim["polarity"],
                    "claim": claim["claim"],
                    "readouts": readouts,
                }
                out_file.write(json.dumps(record) + "\n")
                out_file.flush()
            gc.collect()
            torch.cuda.empty_cache()


@torch.no_grad()
def run_activations(model, tokenizer, dataset, args, activations_path: str) -> None:
    """Residual stream at the source token for every source x frame, averaged over --num_activation_prompts claims."""
    if os.path.exists(activations_path):
        print(f"Activations already saved: {activations_path}")
        return
    n_layers = len(get_decoder_layers(model))
    claims = [(example, c) for example in dataset for c in get_claims(example)][:args.num_activation_prompts]
    acts = torch.zeros(len(args.frames), len(args.sources), n_layers, model.config.hidden_size)
    positions, max_rel_deviation = {}, 0.0

    for fi, frame in enumerate(args.frames):
        for si, source in enumerate(tqdm(args.sources, desc=f"Activations ({frame})")):
            samples = []
            for example, claim in claims:
                question, span = attributed_question(claim["claim"], source, frame, args.determiner)
                ids, pos = encode_with_source(tokenizer, question, span, format_rct_inputs(example["Inputs"]), model.device)
                if pos["straddles"]:
                    print(f"WARNING: last source token {pos['last_token']!r} runs past the phrase in: {question[:60]}")
                with cache_residuals(model, [pos["last"]], device="cpu") as cache:
                    model.model(input_ids=ids, use_cache=False)
                samples.append(torch.stack([cache[l][0].float() for l in range(n_layers)]))   # [layers, d]
                positions[prompt_key(source, frame)] = pos
            samples = torch.stack(samples)
            mean = samples.mean(0)
            deviation = ((samples - mean).norm(dim=-1) / mean.norm(dim=-1).clamp(min=1e-6)).max().item()
            max_rel_deviation = max(max_rel_deviation, deviation)
            acts[fi, si] = mean

    print(f"Largest relative deviation of a source-token activation across claims: {max_rel_deviation:.2e} "
          f"(~0 expected: the source token comes before the claim)")
    torch.save({
        "frames": args.frames, "sources": args.sources, "determiner": args.determiner, "acts": acts,
        "positions": positions, "num_prompts": len(claims), "max_rel_deviation": max_rel_deviation,
    }, activations_path)
    print(f"Saved {activations_path}")


def summarize_behavior(behavior_path: str, summary_path: str, sources: list[str], frames: list[str]) -> dict:
    records = load_jsonl(behavior_path)
    shifts = {}   # frame ("pooled" too) -> source -> polarity ("all" too) -> values
    agreements = {}
    for r in records:
        a_bare = r["readouts"][NO_SOURCE]["agreement"]
        agreements.setdefault(NO_SOURCE, {}).setdefault(r["polarity"], []).append(a_bare)
        for frame in frames:
            for source in sources:
                a = r["readouts"][prompt_key(source, frame)]["agreement"]
                for f in [frame, "pooled"]:
                    for pol in ["all", r["polarity"]]:
                        shifts.setdefault(f, {}).setdefault(source, {}).setdefault(pol, []).append(a - a_bare)
                agreements.setdefault(prompt_key(source, frame), {}).setdefault(r["polarity"], []).append(a)

    # behavior matrix used to fit / test directions: mean shift over all claims, per frame
    b = np.array([[np.mean(shifts[f][s]["all"]) for s in sources] for f in frames])
    pos_mean = [np.mean(shifts["pooled"][s]["positive"]) for s in sources]
    neg_mean = [np.mean(shifts["pooled"][s]["negative"]) for s in sources]
    rho, p = spearman_with_p(pos_mean, neg_mean)
    summary = {
        "num_claims": len(records),
        "sources": sources,
        "frames": frames,
        "shift_vs_bare": {
            f: {s: {pol: aggregate(v) for pol, v in by_pol.items()} for s, by_pol in by_source.items()}
            for f, by_source in shifts.items()
        },
        "agreement": {k: {pol: aggregate(v) for pol, v in by_pol.items()} for k, by_pol in agreements.items()},
        "behavior_matrix": {"frames": frames, "sources": sources, "mean_shift": b.tolist()},
        "polarity_consistency": {"rho_positive_vs_negative_source_order": rho, "p_one_sided": p},
        "source_order_by_shift": [sources[i] for i in np.argsort(-b.mean(0))],
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    return summary


# Stage 2: directions
def rankdata(x) -> np.ndarray:
    """Ranks with ties averaged."""
    x = np.asarray(x, dtype=float)
    ranks = np.empty(len(x))
    ranks[np.argsort(x, kind="mergesort")] = np.arange(len(x))
    for value in np.unique(x):
        ranks[x == value] = ranks[x == value].mean()
    return ranks


def row_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    return (a * b).sum(-1) / np.sqrt((a ** 2).sum(-1) * (b ** 2).sum(-1))


def spearman(x, y) -> float | None:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(row_corr(rankdata(x), rankdata(y)))


_PERMUTATIONS = {}


def permutations_of(n: int, num_random: int = 20000, seed: int = SEED) -> np.ndarray:
    """All permutations of range(n) for n <= 8 (40320), otherwise a fixed random sample."""
    if n not in _PERMUTATIONS:
        if n <= 8:
            _PERMUTATIONS[n] = np.array(list(itertools.permutations(range(n))))
        else:
            rng = np.random.default_rng(seed)
            _PERMUTATIONS[n] = np.array([rng.permutation(n) for _ in range(num_random)])
    return _PERMUTATIONS[n]


def spearman_with_p(x, y) -> tuple[float | None, float | None]:
    """Spearman rho and its one-sided (rho > 0) permutation p-value."""
    rho = spearman(x, y)
    if rho is None:
        return None, None
    rx, ry = rankdata(x), rankdata(y)
    null = row_corr(rx[None, :], ry[permutations_of(len(ry))])
    return rho, float(np.mean(null >= rho - 1e-12))


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True).clip(min=1e-12)


def pls_direction(X: np.ndarray, b: np.ndarray, idx: list[int]) -> np.ndarray:
    """1-component PLS direction from sources `idx`: X [frames, sources, d], b [..., frames, sources] -> [..., d]."""
    Xs = X[:, idx] - X[:, idx].mean(1, keepdims=True)
    bs = b[..., idx] - b[..., idx].mean(-1, keepdims=True)
    return unit(np.einsum("...fs,fsd->...d", bs, Xs))


def loo_predictions(X: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Leave-one-source-out predicted shift for every source. b is [P, frames, sources] (P behavior assignments, e.g.
    permutations); returns [P, sources]. For each held-out source: fit pls on the others, regress their shift on their
    projection, predict the held-out source from its projection (averaged over frames).
    """
    num_sources = X.shape[1]
    preds = np.zeros((b.shape[0], num_sources))
    for s in range(num_sources):
        train = [i for i in range(num_sources) if i != s]
        mu_x, mu_b = X[:, train].mean(1), b[..., train].mean(-1)                        # [F, d], [P, F]
        v = pls_direction(X, b, train)                                                  # [P, d]
        proj_train = np.einsum("fsd,pd->pfs", X[:, train] - mu_x[:, None], v)           # [P, F, S-1]
        b_train = b[..., train] - mu_b[..., None]
        beta = (proj_train * b_train).sum((1, 2)) / (proj_train ** 2).sum((1, 2)).clip(min=1e-12)
        proj_s = np.einsum("fd,pd->pf", X[:, s] - mu_x, v)                              # [P, F]
        preds[:, s] = (mu_b + beta[:, None] * proj_s).mean(1)
    return preds


def compute_directions(X: np.ndarray, b: np.ndarray, sources: list[str]) -> dict[str, np.ndarray]:
    """Unit directions [d] for one layer. X [frames, sources, d], b [frames, sources]."""
    hi, lo = sources.index(HIGH_END), sources.index(LOW_END)
    Xc = X - X.mean(1, keepdims=True)
    directions = {
        "doctor_minus_friend": unit((X[:, hi] - X[:, lo]).mean(0)),
        "pls_all": pls_direction(X, b, list(range(len(sources)))),
        "pls_heldout": pls_direction(X, b, [i for i in range(len(sources)) if i not in (hi, lo)]),
    }
    pc1 = np.linalg.svd(Xc.reshape(-1, Xc.shape[-1]), full_matrices=False)[2][0]
    directions["pc1"] = pc1 if (Xc[:, hi] - Xc[:, lo]).mean(0) @ pc1 >= 0 else -pc1
    return directions


def direction_metrics(X: np.ndarray, b: np.ndarray, v: np.ndarray, sources: list[str], fit_sources: list[str]) -> dict:
    hi, lo = sources.index(HIGH_END), sources.index(LOW_END)
    proj = ((X - X.mean(1, keepdims=True)) @ v).mean(0)     # [sources]
    b_mean = b.mean(0)
    gap = proj[hi] - proj[lo]
    without_doctor = [i for i in range(len(sources)) if i != hi]
    unseen = [i for i, s in enumerate(sources) if s not in fit_sources]
    rho_all, p_all = spearman_with_p(proj, b_mean)
    rho_wd, p_wd = spearman_with_p(proj[without_doctor], b_mean[without_doctor])
    rho_unseen, p_unseen = spearman_with_p(proj[unseen], b_mean[unseen]) if len(unseen) >= 3 else (None, None)
    return {
        "rho_all": rho_all, "p_all": p_all,
        "rho_without_doctor": rho_wd, "p_without_doctor": p_wd,
        "rho_unseen": rho_unseen, "p_unseen": p_unseen, "unseen_sources": [sources[i] for i in unseen],
        "doctor_friend_gap": float(gap),
        "projection": dict(zip(sources, proj.tolist())),
        "credibility_coordinate": dict(zip(sources, ((proj - proj[lo]) / gap).tolist())) if gap != 0 else None,
    }


def run_directions(activations_path: str, behavior_summary: dict, directions_path: str, summary_path: str,
                   num_permutations: int, seed: int) -> dict:
    saved = torch.load(activations_path)
    sources, frames = saved["sources"], saved["frames"]
    if sources != behavior_summary["sources"] or frames != behavior_summary["frames"]:
        raise ValueError("Activation and behavior files use different sources/frames; rerun with the same settings")
    acts = saved["acts"].double().numpy()                                  # [F, S, L, d]
    b = np.array(behavior_summary["behavior_matrix"]["mean_shift"])        # [F, S]
    n_layers = acts.shape[2]
    rng = np.random.default_rng(seed)
    perms = np.array([rng.permutation(len(sources)) for _ in range(num_permutations)])
    b_perm = b[:, perms].transpose(1, 0, 2)                                # [P, F, S], same shuffle for every frame
    fit_sources = {
        "doctor_minus_friend": [HIGH_END, LOW_END],
        "pls_all": sources,
        "pls_heldout": [s for s in sources if s not in (HIGH_END, LOW_END)],
        "pc1": [],
    }

    by_layer = {}
    unit_directions = {name: np.zeros((n_layers, acts.shape[-1])) for name in DIRECTION_NAMES}
    for layer in tqdm(range(n_layers), desc="Directions: layers"):
        X = acts[:, :, layer]
        directions = compute_directions(X, b, sources)
        layer_summary = {}
        for name, v in directions.items():
            unit_directions[name][layer] = v
            layer_summary[name] = direction_metrics(X, b, v, sources, fit_sources[name])

        preds = loo_predictions(X, b[None])[0]
        rho_loo = spearman(preds, b.mean(0))
        null_preds = loo_predictions(X, b_perm)
        null = row_corr(np.argsort(np.argsort(null_preds, 1), 1).astype(float), np.argsort(np.argsort(b_perm.mean(1), 1), 1).astype(float))
        layer_summary["pls_all"]["rho_loo"] = rho_loo
        layer_summary["pls_all"]["p_loo"] = float(np.mean(null >= rho_loo - 1e-12)) if rho_loo is not None else None
        layer_summary["pls_all"]["loo_predicted_shift"] = dict(zip(sources, preds.tolist()))

        if len(frames) > 1:   # fit on one frame, test on another
            cross = []
            for fi, fj in itertools.permutations(range(len(frames)), 2):
                v = pls_direction(X[fi:fi + 1], b[fi:fi + 1], list(range(len(sources))))
                cross.append(spearman(X[fj] @ v, b[fj]))
            layer_summary["pls_all"]["rho_cross_frame"] = float(np.mean([c for c in cross if c is not None]))
        by_layer[str(layer)] = layer_summary

    torch.save({
        "sources": sources, "frames": frames,
        "directions": {name: torch.tensor(v, dtype=torch.float32) for name, v in unit_directions.items()},
        "doctor_friend_gap": {
            name: torch.tensor([by_layer[str(l)][name]["doctor_friend_gap"] for l in range(n_layers)], dtype=torch.float32)
            for name in DIRECTION_NAMES
        },
        # residual norm at the friend source token, to put steering magnitudes in context
        "friend_norm": torch.tensor(np.linalg.norm(acts[:, sources.index(LOW_END)], axis=-1).mean(0), dtype=torch.float32),
    }, directions_path)

    def best(name, metric):
        values = [(by_layer[str(l)][name][metric], l) for l in range(n_layers) if by_layer[str(l)][name][metric] is not None]
        return max(values)[1] if values else None

    summary = {
        "sources": sources, "frames": frames, "behavior_shift": dict(zip(sources, b.mean(0).tolist())),
        "num_permutations_loo": num_permutations,
        "max_rel_deviation_across_claims": saved["max_rel_deviation"],
        "best_layer": {
            "pls_all_rho_loo": best("pls_all", "rho_loo"), "pc1_rho_all": best("pc1", "rho_all"),
            "doctor_minus_friend_rho_unseen": best("doctor_minus_friend", "rho_unseen"),
            "pls_heldout_rho_all": best("pls_heldout", "rho_all"),
        },
        "by_layer": by_layer,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    return summary


# Stage 3: steering
def random_directions(num_random: int, n_layers: int, d_model: int, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        f"random_{k}": torch.nn.functional.normalize(torch.randn(n_layers, d_model, generator=generator), dim=-1)
        for k in range(num_random)
    }


@torch.no_grad()
def run_steering(model, tokenizer, dataset, args, yes_ids, no_ids, directions_path: str, steering_path: str) -> None:
    saved = torch.load(directions_path)
    n_layers, d_model = saved["directions"][args.scale_direction].shape
    directions = {name: saved["directions"][name] for name in args.steer_directions}
    directions.update(random_directions(args.num_random, n_layers, d_model, args.seed))
    gap = saved["doctor_friend_gap"][args.scale_direction]
    layers = []
    for layer in args.steer_layers:
        if gap[layer] <= 0:
            print(f"WARNING: {args.scale_direction} does not put doctor above friend at layer {layer} "
                  f"(gap {gap[layer]:.3f}); skipping it")
        else:
            layers.append(layer)
            print(f"Layer {layer}: steering unit = doctor-friend gap {gap[layer]:.3f} along {args.scale_direction} "
                  f"({gap[layer] / saved['friend_norm'][layer]:.3f} of the friend token's residual norm)")
    alphas = [a for a in args.alphas if a != 0]

    done = load_done(steering_path)
    print(f"Steering: runs already done (resuming): {len(done)}")
    with open(steering_path, "a", encoding="utf-8") as out_file:
        for example in tqdm(dataset, desc="Steering: reviews"):
            rct_inputs = format_rct_inputs(example["Inputs"])
            for claim in get_claims(example):
                for source in args.steer_sources:
                    for frame in args.steer_frames:
                        key = f"{example['ReviewID']}_{claim['claim_key']}_{prompt_key(source, frame)}"
                        if key in done:
                            continue
                        question, span = attributed_question(claim["claim"], source, frame, args.determiner)
                        ids, pos = encode_with_source(tokenizer, question, span, rct_inputs, model.device)
                        positions = [pos["last"]] if args.steer_span == "source_token" else list(range(pos["last"], ids.shape[1]))
                        clean_d = readout(model, ids, yes_ids, no_ids)["d"]
                        steered = {}
                        for name, unit_dirs in directions.items():
                            steered[name] = {}
                            for layer in layers:
                                steered[name][str(layer)] = {}
                                for alpha in alphas:
                                    vector = alpha * gap[layer] * unit_dirs[layer]
                                    with steer_residual(model, layer, positions, vector):
                                        steered[name][str(layer)][str(alpha)] = readout(model, ids, yes_ids, no_ids)["d"]
                        record = {
                            "pair_key": key,
                            "review_id": example["ReviewID"],
                            "claim_key": claim["claim_key"],
                            "polarity": claim["polarity"],
                            "source": source,
                            "frame": frame,
                            "source_token": pos["last_token"],
                            "steer_span": args.steer_span,
                            "scale_direction": args.scale_direction,
                            "steer_scale": {str(l): float(gap[l]) for l in layers},
                            "clean_d": clean_d,
                            "steered_d": steered,
                        }
                        out_file.write(json.dumps(record) + "\n")
                        out_file.flush()
            gc.collect()
            torch.cuda.empty_cache()


def linear_slope(x: list[float], y: list[float]) -> float | None:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2 or np.var(x) == 0:
        return None
    return float(np.cov(x, y, bias=True)[0, 1] / np.var(x))


def summarize_steering(steering_path: str, behavior_path: str, summary_path: str) -> dict:
    records = load_jsonl(steering_path)
    behavior = {r["pair_key"]: r for r in load_jsonl(behavior_path)}

    curves = {}         # direction -> layer -> polarity -> alpha -> agreement change values
    pair_slopes = {}    # direction -> layer -> polarity -> per-pair slopes
    doctor_ref = {}     # polarity -> A(doctor) - A(steered source), same claim and frame
    for r in records:
        sign_d = lambda d: agreement(d, r["polarity"])
        a0 = sign_d(r["clean_d"])
        b_record = behavior.get(f"{r['review_id']}_{r['claim_key']}")
        if b_record is not None:
            a_doc = b_record["readouts"][prompt_key(HIGH_END, r["frame"])]["agreement"]
            for pol in ["all", r["polarity"]]:
                doctor_ref.setdefault(pol, []).append(a_doc - a0)
        for name, by_layer in r["steered_d"].items():
            for layer, by_alpha in by_layer.items():
                alphas = [0.0] + [float(a) for a in by_alpha]
                changes = [0.0] + [sign_d(d) - a0 for d in by_alpha.values()]
                for pol in ["all", r["polarity"]]:
                    for alpha, change in zip(alphas, changes):
                        curves.setdefault(name, {}).setdefault(layer, {}).setdefault(pol, {}).setdefault(alpha, []).append(change)
                    pair_slopes.setdefault(name, {}).setdefault(layer, {}).setdefault(pol, []).append(linear_slope(alphas, changes))

    summary = {"num_runs": len(records), "doctor_reference": {pol: aggregate(v) for pol, v in doctor_ref.items()}, "by_direction": {}}
    for name, by_layer in curves.items():
        summary["by_direction"][name] = {}
        for layer, by_pol in sorted(by_layer.items(), key=lambda kv: int(kv[0])):
            summary["by_direction"][name][layer] = {}
            for pol, by_alpha in by_pol.items():
                alphas = sorted(by_alpha)
                means = [float(np.mean(by_alpha[a])) for a in alphas]
                slopes = [s for s in pair_slopes[name][layer][pol] if s is not None]
                summary["by_direction"][name][layer][pol] = {
                    "alphas": alphas,
                    "agreement_change": {str(a): aggregate(by_alpha[a]) for a in alphas},
                    "mean_slope_per_alpha": aggregate(slopes),
                    "fraction_pairs_positive_slope": float(np.mean([s > 0 for s in slopes])) if slopes else None,
                    "rho_alpha_vs_mean_change": spearman(alphas, means),
                    "monotonic_increasing": bool(all(b > a for a, b in zip(means, means[1:]))),
                }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    return summary


# Plots
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#9aa0a8"     # positive claims, negative claims, random controls
INK, INK2, GRID = "#111214", "#4d5058", "#e6e8ec"
POLARITY_STYLE = {"positive": (BLUE, "o", '"will work" claims'), "negative": (ORANGE, "s", '"won\'t work" claims')}


def style_axes(ax) -> None:
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2)


def plot_behavior(summary: dict, output_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = summary["source_order_by_shift"][::-1]
    pooled = summary["shift_vs_bare"]["pooled"]
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(order) + 1.6))
    for offset, (pol, (color, marker, label)) in zip([0.15, -0.15], POLARITY_STYLE.items()):
        ys = np.arange(len(order)) + offset
        means = [pooled[s][pol]["mean"] for s in order]
        sems = [pooled[s][pol]["sem"] for s in order]
        ax.errorbar(means, ys, xerr=sems, fmt=marker, color=color, markersize=7, markeredgecolor="white", capsize=0,
                    elinewidth=2, label=f"{label} (n={pooled[order[0]][pol]['n']})")
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(np.arange(len(order)), order, color=INK)
    ax.set_xlabel("Agreement shift vs. bare claim (nats, mean ± SEM)", color=INK2)
    rho = summary["polarity_consistency"]["rho_positive_vs_negative_source_order"]
    ax.set_title(f"Behavioral effect of each source\nsource order, will-work vs won't-work: Spearman ρ = {rho:.2f}"
                 if rho is not None else "Behavioral effect of each source", loc="left", color=INK, fontsize=11)
    style_axes(ax)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    path = os.path.join(output_dir, "credibility_behavior.png")
    fig.savefig(path, dpi=200, facecolor="white")
    print(f"Saved {path}")


def plot_direction(summary: dict, output_dir: str, plot_layer: int, scatter_direction: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = sorted(int(l) for l in summary["by_layer"])
    lines = [  # (direction, metric, label, color, linestyle)
        ("pls_all", "rho_loo", "PLS, leave one source out", INK, "-"),
        ("pc1", "rho_all", "PC1 (no behavior used)", BLUE, "-"),
        ("doctor_minus_friend", "rho_unseen", "doctor − friend, other 6 sources", ORANGE, "-"),
        ("pls_heldout", "rho_without_doctor", "PLS without doctor/friend, sources ≠ doctor", GREY, "--"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for name, metric, label, color, ls in lines:
        values = [summary["by_layer"][str(l)][name][metric] for l in layers]
        ax.plot(layers, [np.nan if v is None else v for v in values], color=color, ls=ls, lw=2, label=label)
    ax.axhline(0, color=INK2, lw=0.7)
    ax.axvline(plot_layer, color=GRID, lw=6, zorder=0)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Layer", color=INK2)
    ax.set_ylabel("Spearman ρ: projection vs. behavioral shift", color=INK2)
    ax.set_title("A. Does one direction order the sources?", loc="left", color=INK, fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    style_axes(ax)

    ax = axes[1]
    layer_summary = summary["by_layer"][str(plot_layer)][scatter_direction]
    coord, shift = layer_summary["credibility_coordinate"], summary["behavior_shift"]
    if coord is not None:
        for s in summary["sources"]:
            ax.scatter(coord[s], shift[s], s=60, color=BLUE, edgecolor="white", linewidth=1.5, zorder=3)
            ax.annotate(s, (coord[s], shift[s]), xytext=(6, 4), textcoords="offset points", fontsize=9, color=INK)
        for x in [0, 1]:
            ax.axvline(x, color=INK2, lw=0.7, ls=(0, (2, 3)))
    ax.set_xlabel(f"Credibility coordinate along {scatter_direction} (friend = 0, doctor = 1)", color=INK2)
    ax.set_ylabel("Agreement shift vs. bare claim (nats)", color=INK2)
    rho = layer_summary["rho_all"]
    ax.set_title(f"B. Layer {plot_layer}: sources on the direction (ρ = {rho:.2f})" if rho is not None else f"B. Layer {plot_layer}",
                 loc="left", color=INK, fontsize=11, fontweight="bold")
    style_axes(ax)
    fig.tight_layout()
    path = os.path.join(output_dir, "credibility_direction.png")
    fig.savefig(path, dpi=200, facecolor="white")
    print(f"Saved {path}")


def plot_steering(summary: dict, output_dir: str, main_direction: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_direction = summary["by_direction"]
    layers = sorted(by_direction[main_direction], key=int)
    randoms = [name for name in by_direction if name.startswith("random_")]
    fig, axes = plt.subplots(1, len(layers), figsize=(4.2 * len(layers), 4.4), sharey=True, squeeze=False)
    for ax, layer in zip(axes[0], layers):
        for name in randoms:
            for pol, (color, marker, label) in POLARITY_STYLE.items():
                curve = by_direction[name][layer].get(pol)
                if curve:
                    ax.plot(curve["alphas"], [curve["agreement_change"][str(a)]["mean"] for a in curve["alphas"]],
                            color=GREY, lw=1, ls="--", label="random directions" if (name, pol) == (randoms[0], "positive") else None)
        for pol, (color, marker, label) in POLARITY_STYLE.items():
            curve = by_direction[main_direction][layer].get(pol)
            if not curve:
                continue
            alphas = curve["alphas"]
            means = np.array([curve["agreement_change"][str(a)]["mean"] for a in alphas])
            sems = np.array([curve["agreement_change"][str(a)]["sem"] for a in alphas])
            ax.fill_between(alphas, means - sems, means + sems, color=color, alpha=0.15, linewidth=0)
            ax.plot(alphas, means, color=color, lw=2, marker=marker, markersize=6, markeredgecolor="white", label=label)
            ref = summary["doctor_reference"].get(pol)
            if ref and ref["mean"] is not None:
                ax.axhline(ref["mean"], color=color, lw=1, ls=(0, (1, 2)))
        ax.axhline(0, color=INK2, lw=0.7)
        ax.axvline(1, color=GRID, lw=1)
        ax.set_xlabel("α (1 = friend's projection moved to doctor's)", color=INK2)
        ax.set_title(f"Layer {layer}", loc="left", color=INK, fontsize=11, fontweight="bold")
        style_axes(ax)
    axes[0][0].set_ylabel("Agreement change vs. unsteered (nats)", color=INK2)
    axes[0][0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle(f"Steering friend runs along {main_direction} (dotted: real doctor − friend gap)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    path = os.path.join(output_dir, "credibility_steering.png")
    fig.savefig(path, dpi=200, facecolor="white")
    print(f"Saved {path}")


def main(args):
    set_global_seed(SEED)
    os.makedirs(args.output_dir, exist_ok=True)
    paths = {name: os.path.join(args.output_dir, name) for name in [
        BEHAVIOR_FILE_NAME, ACTIVATIONS_FILE_NAME, BEHAVIOR_SUMMARY_FILE_NAME, DIRECTIONS_FILE_NAME,
        DIRECTION_SUMMARY_FILE_NAME, STEERING_FILE_NAME, STEERING_SUMMARY_FILE_NAME,
    ]}

    dataset = sample_reviews(load_dataset(args.input_path), args.sample_pool, seed=args.seed)[:args.num_reviews]
    if args.debug:
        dataset = dataset[:1]
    print(f"Number of reviews: {len(dataset)}")

    model = tokenizer = None
    if "behavior" in args.stages or "steer" in args.stages:
        model, tokenizer = load_model_and_tokenizer(args.model)
        yes_ids, no_ids = single_token_ids(tokenizer, YES_STRINGS), single_token_ids(tokenizer, NO_STRINGS)
        print(f"Yes ids {yes_ids}, No ids {no_ids}")

    if "behavior" in args.stages:
        run_behavior(model, tokenizer, dataset, args, yes_ids, no_ids, paths[BEHAVIOR_FILE_NAME])
        run_activations(model, tokenizer, dataset, args, paths[ACTIVATIONS_FILE_NAME])
    behavior_summary = summarize_behavior(paths[BEHAVIOR_FILE_NAME], paths[BEHAVIOR_SUMMARY_FILE_NAME], args.sources, args.frames)
    print(f"Claims: {behavior_summary['num_claims']}")
    print("Agreement shift vs. bare claim (pooled over frames), mean +/- SEM:")
    for s in behavior_summary["source_order_by_shift"]:
        r = behavior_summary["shift_vs_bare"]["pooled"][s]
        print(f"  {s:<11} all {r['all']['mean']:+.3f} +/- {r['all']['sem']:.3f} | will work {r['positive']['mean']:+.3f} "
              f"| won't work {r['negative']['mean']:+.3f}")
    print(f"Source order consistency across polarities: {behavior_summary['polarity_consistency']}")
    plot_behavior(behavior_summary, args.output_dir)

    if "direction" in args.stages:
        direction_summary = run_directions(paths[ACTIVATIONS_FILE_NAME], behavior_summary, paths[DIRECTIONS_FILE_NAME],
                                           paths[DIRECTION_SUMMARY_FILE_NAME], args.num_permutations, args.seed)
        print(f"Source-token activations, largest relative deviation across claims: {direction_summary['max_rel_deviation_across_claims']:.2e}")
        print("Per layer: pls_all rho_loo (p) | pc1 rho_all | doctor_minus_friend rho_unseen | pls_heldout rho_all, coordinate of stranger")
        fmt = lambda v: "  n/a" if v is None else f"{v:+.2f}"
        for layer, m in direction_summary["by_layer"].items():
            coord = m["pls_heldout"]["credibility_coordinate"]
            print(f"  layer {int(layer):>2}: {fmt(m['pls_all']['rho_loo'])} ({m['pls_all']['p_loo']:.3f}) | {fmt(m['pc1']['rho_all'])} "
                  f"| {fmt(m['doctor_minus_friend']['rho_unseen'])} | {fmt(m['pls_heldout']['rho_all'])}, "
                  f"stranger {fmt(coord['stranger']) if coord else 'n/a'}")
        print(f"Best layers: {direction_summary['best_layer']}")
        plot_direction(direction_summary, args.output_dir, args.steer_layers[0], args.steer_directions[0])

    if "steer" in args.stages:
        run_steering(model, tokenizer, dataset, args, yes_ids, no_ids, paths[DIRECTIONS_FILE_NAME], paths[STEERING_FILE_NAME])
        steering_summary = summarize_steering(paths[STEERING_FILE_NAME], paths[BEHAVIOR_FILE_NAME], paths[STEERING_SUMMARY_FILE_NAME])
        print(f"Steering runs: {steering_summary['num_runs']}")
        print(f"Real doctor - friend agreement gap: { {p: round(r['mean'], 3) for p, r in steering_summary['doctor_reference'].items()} }")
        for name, by_layer in steering_summary["by_direction"].items():
            for layer, by_pol in by_layer.items():
                parts = [f"{pol}: slope {r['mean_slope_per_alpha']['mean']:+.3f}/alpha, monotonic {r['monotonic_increasing']}, "
                         f"{r['fraction_pairs_positive_slope']:.2f} of pairs rise" for pol, r in by_pol.items() if pol != "all"]
                print(f"  {name:<20} layer {int(layer):>2} | " + " | ".join(parts))
        plot_steering(steering_summary, args.output_dir, args.steer_directions[0])


def int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",")]


def float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",")]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credibility direction at the source token: ordering test and steering dose-response")
    parser.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES, help="stages to run (summaries of earlier stages are reloaded).")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="HuggingFace model name.")
    parser.add_argument("--input_path", required=True, help="dataset with RCT inputs and framed questions (extracted question set).")
    parser.add_argument("--output_dir", required=True, help="directory to save the outputs.")
    parser.add_argument("--num_reviews", default=30, type=int, help="number of reviews.")
    parser.add_argument("--sample_pool", default=30, type=int, help="reviews are the first --num_reviews of the seeded sample of this size used by the main sweeps.")
    parser.add_argument("--sources", nargs="+", default=SOURCES, choices=SOURCES, help="sources (must include doctor and friend).")
    parser.add_argument("--frames", nargs="+", default=list(FRAMES), choices=list(FRAMES), help="attribution frames.")
    parser.add_argument("--determiner", default="natural", choices=list(SOURCE_PHRASES), help="'natural' (my doctor, a stranger) or 'indefinite' (a doctor, a stranger).")
    parser.add_argument("--num_activation_prompts", default=4, type=int, help="claims the source-token activations are averaged over (they should be identical).")
    parser.add_argument("--num_permutations", default=1000, type=int, help="shuffles of the behavioral shifts for the leave-one-source-out p-value.")
    parser.add_argument("--steer_directions", nargs="+", default=["pls_heldout"], choices=DIRECTION_NAMES, help="directions to steer with; the first is plotted.")
    parser.add_argument("--scale_direction", default=None, choices=DIRECTION_NAMES, help="direction whose doctor-friend gap sets alpha = 1 for every steered direction (default: the first --steer_directions).")
    parser.add_argument("--num_random", default=2, type=int, help="matched-norm random directions (control).")
    parser.add_argument("--steer_layers", default="2,4,6,8", type=int_list, help="comma-separated layers to steer at (the source token's information leaves it by ~layer 8).")
    parser.add_argument("--alphas", default="-2,-1,0,0.5,1,2,3,4", type=float_list, help="comma-separated multiples of the doctor-friend gap.")
    parser.add_argument("--steer_sources", nargs="+", default=["friend"], choices=SOURCES, help="sources whose runs are steered.")
    parser.add_argument("--steer_frames", nargs="+", default=["said"], choices=list(FRAMES), help="frames whose runs are steered.")
    parser.add_argument("--steer_span", default="source_token", choices=["source_token", "source_to_end"], help="add the vector at the last source token only, or at it and every later prompt token.")
    parser.add_argument("--seed", default=SEED, type=int, help="seed for sampling reviews, permutations and random directions.")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, help="only run 1 review.")
    args = parser.parse_args()
    args.scale_direction = args.scale_direction or args.steer_directions[0]
    if HIGH_END not in args.sources or LOW_END not in args.sources:
        parser.error(f"--sources must include {HIGH_END} and {LOW_END}")
    if not set(args.steer_frames) <= set(args.frames):
        parser.error("--steer_frames must be a subset of --frames")

    print()
    print("Arguments Provided for Credibility Direction:")
    for k, v in vars(args).items():
        print(f"{k:<24} {v}")
    print()

    main(args)
