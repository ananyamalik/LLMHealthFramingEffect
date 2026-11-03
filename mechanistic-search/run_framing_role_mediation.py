"""
Does the framing push reach the answer through the role direction? And where does role x framing live?

Prompt: prompts/question_only_yes_no.jinja2 ('Question: <question>\\n\\nDoes the treatment in the question work? Answer
with only "Yes" or "No".'). D = log p(Yes) - log p(No) at the last prompt token (> 0: says it works).

Vectors (training reviews, last prompt token of this prompt, as in run_unit_steering.py):
    framing              [positive none] - [negative none]
    doctor/friend/enemy  ([positive <role>] - [positive none] + [negative <role>] - [negative none]) / 2
    role                 the mean of doctor, friend and enemy
    random               a fixed random unit vector per layer (control)
Each is normalized to unit length per layer. Framing is added as -alpha * scale * v (--scale as in run_unit_steering.py).

--experiment steer (held-out test reviews). Prompts: positive / negative claim x no role / "My <role> said," and the
neutral sentence. Framing is steered negative at the last prompt token at one layer L:
    1. projection   the last-token residual's projection on each unit vector at every layer l >= L, steered minus
                    unsteered (proj_change; also in units of the projection's std over the unsteered test prompts).
                    direct = what the added vector itself puts on that direction (-alpha * scale * framing . v); the
                    rest is computed by the layers after L. reference = the real shift: "My <role> said, <claim>"
                    minus "<claim>" for a role vector, negative minus positive claim for framing.
    2. mediation    the same push, with a direction blocked at the last prompt token at every layer after L:
                    clamp   the projection is held at its unsteered value (removes only the change along it)
                    ablate  the projection is set to 0 (compared with the unsteered prompt, ablated the same way)
                    remaining = effect with the block / effect without. ~1: the push does not need that direction;
                    ~0: it goes through it. random = the same block on a random direction (control).
    Readouts (mediation.csv): the D change of every prompt. (deference.csv): the role effect ("deference")
    D(My <role> said, <claim>) - D(<claim>) per framing: how much the push changes it, and how much of that change
    is left with each block.

--experiment patch: 2x2 role (--source_role, default doctor = authority; --target_role, default friend = peer) x
framing. The source prompt's residual at one position and layer is written into the target prompt (the prompts
differ only in the role word), for every position from the role word on and every --patch_layers layer, plus two
spans: the whole claim, and everything from the role word on (a check: gives the source's D). Under each framing:
    role_effect(f)  D(source) - D(target)
    effect(f)       D(target, patched) - D(target);   recovery(f) = mean effect(f) / mean role_effect(f)
    interaction     effect(positive) - effect(negative), and the fraction of the real one,
                    role_effect(positive) - role_effect(negative), that it carries
Position labels: role (the role word), the tokens up to the claim, claim (the treatment / condition tokens, which
differ by review; averaged), then the fixed tail (" What do you think?" + instruction + chat template), labeled by
the distance from the last token.

Outputs (--output_dir):
    directions.pt                            as in run_unit_steering.py, fit on this prompt once and reused
    steer_<scale>/vectors.pt                 the vectors as added (alpha 1), the unit vectors and the random control
    steer_<scale>/records.jsonl              per test review: D and the per-layer projections of every prompt,
                                             unsteered and steered, and D with each block (resumable)
    steer_<scale>/projection.csv, mediation.csv, deference.csv
    patch_<source>_to_<target>/pairs.jsonl   per (review, framing): the clean D's and the patched D per position and
                                             layer (resumable)
    patch_<source>_to_<target>/summary.csv, patching.png
"""
import argparse
import csv
import gc
import json
import os
from contextlib import contextmanager

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from patching_utils import (
    DEFAULT_MODEL_NAME, SEED, set_global_seed, load_model_and_tokenizer, get_decoder_layers, load_dataset,
    cache_residuals,
)
from run_role_patching import FRAMINGS, ROLES, YES_STRINGS, NO_STRINGS, role_prefix, get_claims, encode_prompt, single_token_ids
from run_unit_steering import VECTORS, NO_ROLE, DEFAULT_ALPHAS, neutral_sentence, prompt_ids, fit_directions
from get_steering_vector import split_reviews

ANSWER = "yes_no"
BLOCK_DIRS = ROLES + ["role", "random"]
PROJ_DIRS = ["framing"] + BLOCK_DIRS
BLOCKS = ["clamp", "ablate"]
PROMPT_KEYS = [f"{f}|{r}" for r in [NO_ROLE] + ROLES for f in FRAMINGS] + [f"neutral|{NO_ROLE}"]
TAIL = " What do you think?"
SPANS = ["claim (all)", "all from role"]


def question_for(claims: dict, key: str) -> str:
    framing, role = key.split("|")
    claim = neutral_sentence(claims["positive"]) if framing == "neutral" else claims[framing]
    return claim if role == NO_ROLE else f"{role_prefix(role)} {claim}"


def yes_no_d(model, hidden: torch.Tensor, yes_no_ids) -> list[float]:
    """D per row from the final hidden state at the last token [rows, d_model]."""
    log_probs = torch.log_softmax(model.lm_head(hidden).float(), dim=-1)
    yes_ids, no_ids = yes_no_ids
    return (torch.logsumexp(log_probs[:, yes_ids], -1) - torch.logsumexp(log_probs[:, no_ids], -1)).tolist()


@contextmanager
def intervene(model, add: dict, block: dict, capture: dict | None = None):
    """Per batch row, at the last prompt token of decoder layer l's output: adds add[l] [rows, d_model], then sets the
    projection on the unit vector block[l][0] [rows, d_model] to block[l][1] [rows] (rows with a zero vector are left
    as they are). With `capture`, stores every layer's last-token output after that in capture[l] (the same place the
    edits are made; hidden_states[-1] would be after the final norm)."""
    def make_hook(l):
        def hook(module, inputs, output):
            h = (output[0] if isinstance(output, tuple) else output)[:, -1, :]
            if l in add:
                h += add[l].to(device=h.device, dtype=h.dtype)
            if l in block:
                u, target = (t.to(h.device) for t in block[l])
                proj = (h.float() * u).sum(-1)
                h -= ((proj - target)[:, None] * u).to(h.dtype)
            if capture is not None:
                capture[l] = h.float().cpu()
            return output
        return hook

    layers = get_decoder_layers(model)
    hooked = range(len(layers)) if capture is not None else set(add) | set(block)
    handles = [layers[l].register_forward_hook(make_hook(l)) for l in hooked]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def run_rows(model, ids: torch.Tensor, rows: list[tuple], yes_no_ids, args, want_states: bool = False):
    """rows: (add {layer: vector}, block {layer: (unit vector, target projection)}), one forward pass each (batched).
    Returns D per row and, if want_states, the last-token decoder layer outputs [rows, layers, d_model]."""
    d_model = model.config.hidden_size
    ds, states = [], []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        add, block = {}, {}
        for i, (row_add, row_block) in enumerate(batch):
            for l, v in row_add.items():
                add.setdefault(l, torch.zeros(len(batch), d_model))[i] = v
            for l, (u, target) in row_block.items():
                block.setdefault(l, (torch.zeros(len(batch), d_model), torch.zeros(len(batch))))
                block[l][0][i], block[l][1][i] = u, target
        capture = {} if want_states else None
        with intervene(model, add, block, capture):
            out = model.model(input_ids=ids.repeat(len(batch), 1), use_cache=False)
        ds += yes_no_d(model, out.last_hidden_state[:, -1], yes_no_ids)
        if want_states:
            states.append(torch.stack([capture[l] for l in sorted(capture)], 1))
    return ds, (torch.cat(states) if want_states else None)


def steer_review(model, tokenizer, example: dict, vectors: dict, units: dict, layers: list[int], args, yes_no_ids) -> dict:
    """Experiments 1 and 2 for one review: every prompt unsteered, steered (-alpha x framing at each layer) and steered
    with each block after the steering layer."""
    num_layers = len(get_decoder_layers(model))
    claims = get_claims(example)
    U = torch.stack([units[n] for n in PROJ_DIRS], 1)   # [layers, dirs, d_model]
    conds = [(L, a) for L in layers for a in args.alphas]
    record = {"review_id": example["ReviewID"], "questions": {k: question_for(claims, k) for k in PROMPT_KEYS},
              "clean": {}, "steered": []}
    for key in PROMPT_KEYS:
        ids = prompt_ids(tokenizer, record["questions"][key], NO_ROLE, model.device, ANSWER)
        rows = [({}, {})] + [({L: -a * vectors["framing"][L]}, {}) for L, a in conds]
        d, states = run_rows(model, ids, rows, yes_no_ids, args, want_states=True)
        proj = torch.einsum("rld,lnd->rnl", states, U)   # [rows, dirs, layers]
        clean_proj = {n: proj[0, j] for j, n in enumerate(PROJ_DIRS)}

        def blocked(n, kind, L):
            return {l: (units[n][l], float(clean_proj[n][l]) if kind == "clamp" else 0.0) for l in range(L + 1, num_layers)}

        base_rows = [((L, n), ({}, blocked(n, "ablate", L))) for L in layers for n in BLOCK_DIRS]
        block_rows = [((L, a, n, kind), ({L: -a * vectors["framing"][L]}, blocked(n, kind, L)))
                      for L, a in conds for n in BLOCK_DIRS for kind in BLOCKS]
        d_blocked, _ = run_rows(model, ids, [r for _, r in base_rows + block_rows], yes_no_ids, args)
        d_base = dict(zip([k for k, _ in base_rows], d_blocked[:len(base_rows)]))
        d_block = dict(zip([k for k, _ in block_rows], d_blocked[len(base_rows):]))

        record["clean"][key] = {"d": d[0], "proj": {n: [round(x, 4) for x in clean_proj[n].tolist()] for n in PROJ_DIRS}}
        for i, (L, a) in enumerate(conds, start=1):
            record["steered"].append({
                "prompt": key, "layer": L, "alpha": a, "d": d[i],
                "proj": {n: [round(x, 4) for x in proj[i, j].tolist()] for j, n in enumerate(PROJ_DIRS)},
                **{kind: {n: d_block[L, a, n, kind] for n in BLOCK_DIRS} for kind in BLOCKS},
                "ablate_base": {n: d_base[L, n] for n in BLOCK_DIRS},
            })
    return record


def ratio(a: float, b: float) -> float:
    return a / b if abs(b) > 1e-6 else float("nan")


def write_rows(rows: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_steer(records: list[dict], units: dict, scale: torch.Tensor, layers: list[int], args, run_dir: str) -> None:
    num_layers = scale.shape[0]
    steered = [{(s["prompt"], s["layer"], s["alpha"]): s for s in r["steered"]} for r in records]
    clean_d = lambda r, k: r["clean"][k]["d"]
    clean_proj = lambda r, k, n: np.array(r["clean"][k]["proj"][n])
    proj_std = {n: np.stack([clean_proj(r, k, n) for r in records for k in PROMPT_KEYS]).std(0) for n in PROJ_DIRS}
    cos = {n: (units["framing"] @ units[n].T).numpy() for n in PROJ_DIRS}   # [steer layer, layer]

    def reference(n, key):
        framing = key.split("|")[0]
        framings = FRAMINGS if framing == "neutral" else [framing]
        if n == "framing":
            return np.mean([clean_proj(r, f"negative|{NO_ROLE}", n) - clean_proj(r, f"positive|{NO_ROLE}", n) for r in records], 0)
        if n == "random":
            return np.full(num_layers, np.nan)
        roles = ROLES if n == "role" else [n]
        return np.mean([clean_proj(r, f"{f}|{role}", n) - clean_proj(r, f"{f}|{NO_ROLE}", n)
                        for r in records for f in framings for role in roles], 0)

    projection, mediation, deference = [], [], []
    for key in PROMPT_KEYS:
        refs = {n: reference(n, key) for n in PROJ_DIRS}
        for L in layers:
            for a in args.alphas:
                ss = [(r, s[key, L, a]) for r, s in zip(records, steered)]
                for n in PROJ_DIRS:
                    change = np.mean([np.array(s["proj"][n]) - clean_proj(r, key, n) for r, s in ss], 0)
                    for l in range(L, num_layers):
                        projection.append({"prompt": key, "steer_layer": L, "alpha": a, "direction": n, "layer": l,
                                           "proj_change": change[l], "direct": -a * float(scale[L]) * cos[n][L, l],
                                           "proj_change_sd": change[l] / proj_std[n][l], "reference": refs[n][l]})
                effect = np.mean([s["d"] - clean_d(r, key) for r, s in ss])
                for n in BLOCK_DIRS:
                    clamp = np.mean([s["clamp"][n] - clean_d(r, key) for r, s in ss])
                    ablate = np.mean([s["ablate"][n] - s["ablate_base"][n] for r, s in ss])
                    mediation.append({"prompt": key, "steer_layer": L, "alpha": a, "blocked": n,
                                      "d_clean": np.mean([clean_d(r, key) for r, _ in ss]), "effect": effect,
                                      "clamp_effect": clamp, "clamp_remaining": ratio(clamp, effect),
                                      "ablate_effect": ablate, "ablate_remaining": ratio(ablate, effect)})

    for role in ROLES:
        for f in FRAMINGS:
            k_role, k_none = f"{f}|{role}", f"{f}|{NO_ROLE}"
            clean = np.mean([clean_d(r, k_role) - clean_d(r, k_none) for r in records])
            for L in layers:
                for a in args.alphas:
                    pairs = [(s[k_role, L, a], s[k_none, L, a]) for s in steered]
                    change = np.mean([sr["d"] - sn["d"] for sr, sn in pairs]) - clean
                    for n in BLOCK_DIRS:
                        clamp = np.mean([sr["clamp"][n] - sn["clamp"][n] for sr, sn in pairs]) - clean
                        ablate = np.mean([(sr["ablate"][n] - sn["ablate"][n]) - (sr["ablate_base"][n] - sn["ablate_base"][n])
                                          for sr, sn in pairs])
                        deference.append({"role": role, "framing": f, "steer_layer": L, "alpha": a, "blocked": n,
                                          "role_effect_clean": clean, "change": change,
                                          "clamp_change": clamp, "clamp_remaining": ratio(clamp, change),
                                          "ablate_change": ablate, "ablate_remaining": ratio(ablate, change)})

    for name, rows in [("projection", projection), ("mediation", mediation), ("deference", deference)]:
        write_rows(rows, os.path.join(run_dir, f"{name}.csv"))
    print(f"\nSaved projection.csv, mediation.csv, deference.csv in {run_dir} ({len(records)} reviews)")

    a = max(args.alphas)
    print(f"\n1. Projection change on the role vector (std units) at the last layer, -{a:g} x framing at layer L:")
    print("   L    " + "  ".join(f"{k:>16}" for k in PROMPT_KEYS[:2]) + "   (direct part, std units)")
    for L in layers:
        rows = {r["prompt"]: r for r in projection if (r["steer_layer"], r["alpha"], r["direction"], r["layer"])
                == (L, a, "role", num_layers - 1)}
        print(f"   L{L:<3} " + "  ".join(f"{rows[k]['proj_change_sd']:>+16.2f}" for k in PROMPT_KEYS[:2])
              + f"   {rows[PROMPT_KEYS[0]]['direct'] / proj_std['role'][-1]:+.2f}")
    print(f"\n2. D change from -{a:g} x framing, and the fraction left with the role vector / a random vector clamped:")
    for key in [f"{f}|{NO_ROLE}" for f in FRAMINGS] + [f"neutral|{NO_ROLE}"]:
        for L in layers:
            rows = {r["blocked"]: r for r in mediation if (r["prompt"], r["steer_layer"], r["alpha"]) == (key, L, a)}
            print(f"   {key:<15} L{L:<3} effect {rows['role']['effect']:+6.2f}   left: role clamp "
                  f"{rows['role']['clamp_remaining']:5.2f}  ablate {rows['role']['ablate_remaining']:5.2f}   "
                  f"random clamp {rows['random']['clamp_remaining']:5.2f}")
    print(f"\n   Role effect D(My <role> said) - D(no role), change from -{a:g} x framing, fraction left with role clamped:")
    for row in deference:
        if row["alpha"] == a and row["blocked"] == "role":
            print(f"   {row['role']:<7} {row['framing']:<9} L{row['steer_layer']:<3} clean {row['role_effect_clean']:+6.2f}"
                  f"  change {row['change']:+6.2f}  left {row['clamp_remaining']:5.2f}")


def run_steer(model, tokenizer, fit: dict, test_set: list[dict], layers: list[int], args, yes_no_ids) -> None:
    num_layers, d_model = fit["unit"]["framing"].shape
    scale = {"unit": torch.ones(num_layers), "residual": fit["residual_norm"], "proj_std": fit["proj_std"]["framing"]}[args.scale]
    vectors = {"framing": fit["unit"]["framing"] * scale[:, None]}
    generator = torch.Generator().manual_seed(args.seed)
    random = torch.randn(num_layers, d_model, generator=generator)
    units = {n: fit["unit"][n] for n in PROJ_DIRS if n != "random"} | {"random": random / random.norm(dim=-1, keepdim=True)}
    run_dir = os.path.join(args.output_dir, f"steer_{args.scale}")
    os.makedirs(run_dir, exist_ok=True)
    torch.save({"vectors": vectors, "scale": scale, "units": units, "alphas": args.alphas, "layers": layers,
                "scale_kind": args.scale}, os.path.join(run_dir, "vectors.pt"))
    print(f"\n[steer] -alpha x framing at layers {layers} | alphas {args.alphas} | scale {args.scale} | "
          f"blocks {BLOCKS} on {BLOCK_DIRS} | {len(test_set)} test reviews")

    path = os.path.join(run_dir, "records.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            done = {json.loads(line)["review_id"] for line in f if line.strip()}
    print(f"{len(done)} reviews already done")
    with open(path, "a", encoding="utf-8") as out_file:
        for example in tqdm(test_set, desc="steer"):
            if example["ReviewID"] in done:
                continue
            out_file.write(json.dumps(steer_review(model, tokenizer, example, vectors, units, layers, args, yes_no_ids)) + "\n")
            out_file.flush()
            gc.collect()
            torch.cuda.empty_cache()
    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    summarize_steer(records, units, scale, layers, args, run_dir)


def patch_positions(tokenizer, src, tgt, question: str, prefix: str) -> list[tuple[int, str]]:
    """(position, label) for every target position from the role word on; checks the prompts differ only there."""
    src_ids, _, _ = src
    tgt_ids, enc, q_start = tgt
    role_pos = enc.char_to_token(q_start + len("My "))
    claim_pos = enc.char_to_token(q_start + len(prefix) + 1)
    tail_pos = enc.char_to_token(q_start + question.rindex(TAIL) + 1)
    if src_ids.shape != tgt_ids.shape or (src_ids != tgt_ids).nonzero()[:, 1].tolist() != [role_pos]:
        raise ValueError("the source and target prompts differ in more than the role word")
    seq = tgt_ids.shape[1]
    labels = []
    for p in range(role_pos, seq):
        token = repr(tokenizer.decode([tgt_ids[0, p].item()]))
        if p == role_pos:
            labels.append((p, "role"))
        elif p < claim_pos:
            labels.append((p, f"prefix {token}"))
        elif p < tail_pos:
            labels.append((p, "claim"))
        else:
            labels.append((p, f"end-{seq - 1 - p:02d} {token}"))
    return labels


@torch.no_grad()
def patch_pair(model, tgt_ids: torch.Tensor, src_states: dict, rows: list[tuple[int, list[int]]], yes_no_ids, args) -> list[float]:
    """D of the target with, per row, the source's decoder layer `layer` output written in at `positions`."""
    ds = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        by_layer = {}
        for i, (l, positions) in enumerate(batch):
            by_layer.setdefault(l, []).append((i, positions))

        def make_hook(l):
            def hook(module, inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                for i, positions in by_layer[l]:
                    h[i, positions] = src_states[l][positions].to(device=h.device, dtype=h.dtype)
                return output
            return hook

        layers = get_decoder_layers(model)
        handles = [layers[l].register_forward_hook(make_hook(l)) for l in by_layer]
        try:
            out = model.model(input_ids=tgt_ids.repeat(len(batch), 1), use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        ds += yes_no_d(model, out.last_hidden_state[:, -1], yes_no_ids)
    return ds


def summarize_patch(records: list[dict], patch_layers: list[int], run_dir: str, title: str) -> None:
    def effects(r):
        """label -> [layer] mean effect (D patched - D target) over the label's positions."""
        out = {}
        for pos, d in zip(r["positions"], r["d_patched"]):
            out.setdefault(pos["label"], []).append(np.array(d) - r["d_target"])
        out = {k: np.mean(v, 0) for k, v in out.items()}
        return out | {k: np.array(v) - r["d_target"] for k, v in r["spans"].items()}

    by_review = {}
    for r in records:
        by_review.setdefault(r["review_id"], {})[r["framing"]] = r
    complete = [v for v in by_review.values() if set(v) == set(FRAMINGS)]
    labels = list(dict.fromkeys(p["label"] for r in records for p in r["positions"]))
    labels = ([l for l in labels if not l.startswith("end-")] + sorted(l for l in labels if l.startswith("end-"))[::-1]
              + SPANS)
    eff = [{f: effects(v[f]) for f in FRAMINGS} for v in complete]
    role_effect = {f: np.mean([v[f]["d_source"] - v[f]["d_target"] for v in complete]) for f in FRAMINGS}
    real_interaction = role_effect["positive"] - role_effect["negative"]

    rows, grid = [], {k: np.full((len(labels), len(patch_layers)), np.nan) for k in ["positive", "negative", "interaction"]}
    for i, label in enumerate(labels):
        present = [e for e in eff if all(label in e[f] for f in FRAMINGS)]
        if not present:
            continue
        mean = {f: np.mean([e[f][label] for e in present], 0) for f in FRAMINGS}
        interaction = mean["positive"] - mean["negative"]
        for j, l in enumerate(patch_layers):
            rows.append({"label": label, "layer": l, "n_reviews": len(present),
                         "effect_positive": mean["positive"][j], "effect_negative": mean["negative"][j],
                         "recovery_positive": ratio(mean["positive"][j], role_effect["positive"]),
                         "recovery_negative": ratio(mean["negative"][j], role_effect["negative"]),
                         "interaction": interaction[j], "interaction_fraction": ratio(interaction[j], real_interaction)})
            grid["positive"][i, j] = rows[-1]["recovery_positive"]
            grid["negative"][i, j] = rows[-1]["recovery_negative"]
            grid["interaction"][i, j] = rows[-1]["interaction_fraction"]
    write_rows(rows, os.path.join(run_dir, "summary.csv"))
    print(f"\n[patch] {len(complete)} reviews with both framings | role effect D(source) - D(target): positive "
          f"{role_effect['positive']:+.2f}, negative {role_effect['negative']:+.2f} | real interaction {real_interaction:+.2f}")
    for label in labels:
        row = {r["layer"]: r for r in rows if r["label"] == label}
        if row:
            print(f"   {label:<28} interaction fraction: " + " ".join(f"L{l} {row[l]['interaction_fraction']:+.2f}"
                                                                       for l in patch_layers[::max(1, len(patch_layers) // 8)]))

    fig, axes = plt.subplots(1, 3, figsize=(18, 0.32 * len(labels) + 2), sharey=True)
    panels = [("positive", "recovery, positive claims"), ("negative", "recovery, negative claims"),
              ("interaction", "interaction carried\n(effect pos - effect neg) / real")]
    for ax, (k, name) in zip(axes, panels):
        limit = max(1.0, float(np.nanmax(np.abs(grid[k])))) if np.isfinite(grid[k]).any() else 1.0
        image = ax.imshow(grid[k], aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
        ax.set_title(name, fontsize=11)
        ax.set_xticks(range(len(patch_layers))[::2], [str(l) for l in patch_layers][::2], fontsize=8)
        ax.set_xlabel("layer")
        fig.colorbar(image, ax=ax, fraction=0.04)
    axes[0].set_yticks(range(len(labels)), labels, fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "patching.png"), dpi=150)
    plt.close(fig)
    print(f"Saved {os.path.join(run_dir, 'summary.csv')} and patching.png")


def run_patch(model, tokenizer, test_set: list[dict], args, yes_no_ids) -> None:
    num_layers = len(get_decoder_layers(model))
    patch_layers = args.patch_layers or list(range(num_layers))
    src_role, tgt_role = args.source_role, args.target_role
    run_dir = os.path.join(args.output_dir, f"patch_{src_role}_to_{tgt_role}")
    os.makedirs(run_dir, exist_ok=True)
    reviews = test_set[:args.num_patch_reviews] if args.num_patch_reviews > 0 else test_set
    print(f"\n[patch] '{role_prefix(src_role)}' -> '{role_prefix(tgt_role)}' | layers {patch_layers} | {len(reviews)} reviews")

    path = os.path.join(run_dir, "pairs.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            done = {json.loads(line)["pair_key"] for line in f if line.strip()}
    with open(path, "a", encoding="utf-8") as out_file:
        for example in tqdm(reviews, desc="patch"):
            for framing, claim in get_claims(example).items():
                key = f"{example['ReviewID']}_{framing}"
                if key in done:
                    continue
                src_q, tgt_q = f"{role_prefix(src_role)} {claim}", f"{role_prefix(tgt_role)} {claim}"
                src = encode_prompt(tokenizer, ANSWER, src_q, [], model.device, "question_only")
                tgt = encode_prompt(tokenizer, ANSWER, tgt_q, [], model.device, "question_only")
                try:
                    positions = patch_positions(tokenizer, src, tgt, tgt_q, role_prefix(tgt_role))
                except ValueError as e:
                    print(f"Skipping {key}: {e}")
                    continue
                seq = tgt[0].shape[1]
                with torch.no_grad(), cache_residuals(model, list(range(seq)), layer_indices=patch_layers) as cache:
                    src_out = model.model(input_ids=src[0], use_cache=False)
                d_source = yes_no_d(model, src_out.last_hidden_state[:, -1], yes_no_ids)[0]
                d_target = run_rows(model, tgt[0], [({}, {})], yes_no_ids, args)[0][0]
                claim_span = [p for p, label in positions if label == "claim"]
                spans = {"claim (all)": claim_span, "all from role": [p for p, _ in positions]}
                rows = [(l, [p]) for p, _ in positions for l in patch_layers]
                rows += [(l, spans[s]) for s in SPANS for l in patch_layers]
                d = patch_pair(model, tgt[0], cache, rows, yes_no_ids, args)
                n = len(patch_layers)
                record = {"pair_key": key, "review_id": example["ReviewID"], "framing": framing,
                          "source_question": src_q, "target_question": tgt_q, "d_source": d_source, "d_target": d_target,
                          "layers": patch_layers,
                          "positions": [{"pos": p, "label": label, "token": tokenizer.decode([tgt[0][0, p].item()])}
                                        for p, label in positions],
                          "d_patched": [d[i * n:(i + 1) * n] for i in range(len(positions))],
                          "spans": {s: d[(len(positions) + i) * n:(len(positions) + i + 1) * n] for i, s in enumerate(SPANS)}}
                out_file.write(json.dumps(record) + "\n")
                out_file.flush()
                del cache
            gc.collect()
            torch.cuda.empty_cache()
    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    summarize_patch(records, patch_layers, run_dir,
                    f"'{role_prefix(src_role)}' patched into '{role_prefix(tgt_role)}' (recovery: 1 = target answers like source)")


def main(args):
    set_global_seed(SEED)
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = load_dataset(args.input_path)
    test = split_reviews([e["ReviewID"] for e in dataset], args.test_fraction, args.seed)
    train_set = [e for e in dataset if e["ReviewID"] not in test]
    test_set = [e for e in dataset if e["ReviewID"] in test]
    if args.num_reviews > 0:
        test_set = test_set[:args.num_reviews]
    print(f"Reviews: {len(train_set)} train (extract) | {len(test_set)} test")

    model, tokenizer = load_model_and_tokenizer(args.model)
    num_layers = len(get_decoder_layers(model))
    layers = args.layers or list(range(0, num_layers, 4))
    yes_no_ids = (single_token_ids(tokenizer, YES_STRINGS), single_token_ids(tokenizer, NO_STRINGS))

    if "steer" in args.experiment:
        path = os.path.join(args.output_dir, "directions.pt")
        fit = torch.load(path) if os.path.exists(path) else None
        if fit is not None and set(VECTORS) <= set(fit["unit"]):
            print(f"Loaded {path}")
        else:
            fit = fit_directions(model, tokenizer, train_set, args, ANSWER) | {"test_reviews": sorted(test), "prompt": ANSWER}
            torch.save(fit, path)
            print(f"Saved {path}")
        print("cos(framing, x) per layer: " + " ".join(
            f"L{l} " + "/".join(f"{float(fit['unit']['framing'][l] @ fit['unit'][n][l]):+.2f}" for n in ROLES + ["role"])
            for l in layers) + "  (doctor/friend/enemy/role)")
        run_steer(model, tokenizer, fit, test_set, layers, args, yes_no_ids)
    if "patch" in args.experiment:
        run_patch(model, tokenizer, test_set, args, yes_no_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Framing -> role mediation (steer + block) and role x framing patching")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--input_path", required=True, help="dataset with the framed questions (extracted question set).")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--experiment", nargs="+", default=["steer", "patch"], choices=["steer", "patch"])
    parser.add_argument("--layers", nargs="*", type=int, default=None, help="steer: layers to steer at (default: every 4th).")
    parser.add_argument("--scale", default="residual", choices=list(DEFAULT_ALPHAS),
                        help="steer: unit / residual (x mean residual norm) / proj_std (x std of h.v), as run_unit_steering.py.")
    parser.add_argument("--alphas", nargs="+", type=float, default=None, help="steer: push sizes (framing is added as -alpha; default per --scale).")
    parser.add_argument("--source_role", default="doctor", help="patch: the role whose activations are written in.")
    parser.add_argument("--target_role", default="friend", help="patch: the role prompt that receives them.")
    parser.add_argument("--patch_layers", nargs="*", type=int, default=None, help="patch: layers (default: all).")
    parser.add_argument("--num_reviews", default=-1, type=int, help="first n test reviews (-1: all).")
    parser.add_argument("--num_patch_reviews", default=20, type=int, help="patch: first n of those (-1: all).")
    parser.add_argument("--test_fraction", default=0.2, type=float)
    parser.add_argument("--batch_size", default=64, type=int, help="forward passes run at once.")
    parser.add_argument("--seed", default=SEED, type=int)
    args = parser.parse_args()
    args.alphas = args.alphas or DEFAULT_ALPHAS[args.scale]
    print("\nArguments:")
    for k, v in vars(args).items():
        print(f"{k:<18} {v}")
    print()
    main(args)
