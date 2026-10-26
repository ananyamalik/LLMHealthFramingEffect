"""
Subject probe: can the residual stream of the last prompt token tell which subject a question is about, does an
expert speaker in the prompt ("My mathematician said, <question>") change that, and what does it say about the medical
claims?

Data: MMLU-Pro (TIGER-Lab/MMLU-Pro, test split; "other" is dropped by default since it has no natural expert).
--per_category questions are drawn per category (seeded) and split into train/test by question (--test_fraction,
stratified), so no question is in both. Each question is rendered in three versions:
    none        "<question>"
    matched     "My <expert of its category> said, <question>"     (EXPERTS, e.g. math -> mathematician, health -> doctor)
    mismatched  "My <expert of another category> said, <question>" (seeded random other category; test only)
in the question-only prompt of run_role_patching.py (prompts/question_only.jinja2, "Question: <question>", chat
template, generation prompt), without the answer options by default. The residual stream (decoder layer output) of the
last prompt token is cached at every layer; this is the position of the `last` patching setup.

Probes (per layer, multinomial logistic regression on the standardized residual, full-batch L-BFGS, L2 penalty chosen
on a validation split of the train questions, then refit on all of train):
    expert     trained on matched
    no_expert  trained on none
    mixed      trained on one version per train question, matched or none at random (same size as the others)
Every probe is tested on the test questions in all three versions. Accuracy is against the question's category; on
mismatched prompts, speaker_rate is how often the probe predicts the speaker's category instead. A probe that reads
the subject keeps its accuracy on mismatched prompts; one that reads the speaker follows speaker_rate.

Medical claims: the claims of run_role_patching.py (the `doctor` question without "My doctor says "), in both framings,
with no role ("<claim>") and with each of --roles ("My <role> said, <claim>"), in the same prompt, through every
probe. Note that "doctor" is the health expert, so p(health) for the doctor claims includes the speaker cue for the
expert and mixed probes; the bare/friend/enemy claims do not have it.

Outputs (in --output_dir):
    probe.pt             per probe and layer: weights [layers, classes, d_model], bias, mean, std; categories; l2
    mmlu_summary.json    per probe and layer: l2, val/train accuracy, test accuracy on none/matched/mismatched and the
                         mismatched speaker_rate; per probe, the best layer (by validation) with its confusion matrix
                         on its own test version(s)
    claims.jsonl         one line per (review, framing, role): question, p(health) and top subject per probe and layer
    claims_summary.json  per probe, role (and framing) and layer: mean p(health), fraction with top subject = health,
                         and the top-subject counts
    subject_probe.png    per probe: MMLU-Pro test accuracy by layer and version, and mean p(health) of the claims by role
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from patching_utils import (
    DEFAULT_MODEL_NAME, SEED, render_prompt, set_global_seed, load_model_and_tokenizer, get_decoder_layers,
    load_dataset, sample_reviews, cache_residuals,
)
from run_role_patching import FRAMINGS, QUESTION_ONLY_TEMPLATES, get_claims

MMLU_NAME = "TIGER-Lab/MMLU-Pro"
TARGET_SUBJECT = "health"
NO_ROLE = "none"
OPTION_LETTERS = "ABCDEFGHIJ"
EXPERTS = {
    "biology": "biologist",
    "business": "business consultant",
    "chemistry": "chemist",
    "computer science": "computer scientist",
    "economics": "economist",
    "engineering": "engineer",
    "health": "doctor",
    "history": "historian",
    "law": "lawyer",
    "math": "mathematician",
    "philosophy": "philosopher",
    "physics": "physicist",
    "psychology": "psychologist",
}
VERSIONS = ["none", "matched", "mismatched"]
PROBES = {"expert": ["matched"], "no_expert": ["none"], "mixed": ["matched", "none"]}   # probe -> training version(s)


def speaker_prefix(speaker_format: str, speaker: str) -> str:
    """e.g. "My mathematician said," with the default --speaker_format (the claims' "My <role> said,")."""
    return speaker_format.format(speaker=speaker)


def encode_question(tokenizer, question: str, device) -> torch.Tensor:
    """Chat-template ids [1, seq] of the question-only prompt ("Question: <question>"), as in run_role_patching.py."""
    template_name, template_dir = QUESTION_ONLY_TEMPLATES["full"]
    user_text = render_prompt(template_name, template_dir=template_dir, question=question)
    ids = tokenizer.apply_chat_template([{"role": "user", "content": user_text}], tokenize=True, add_generation_prompt=True)
    if not isinstance(ids, list):  # newer transformers can return a BatchEncoding
        ids = ids["input_ids"]
    return torch.tensor([ids], device=device)


@torch.no_grad()
def last_token_residuals(model, tokenizer, questions: list[str], desc: str) -> torch.Tensor:
    """Residual stream of the last prompt token at every layer: [len(questions), layers, d_model], float16 on CPU."""
    out = []
    num_layers = len(get_decoder_layers(model))
    for question in tqdm(questions, desc=desc):
        ids = encode_question(tokenizer, question, model.device)
        with cache_residuals(model, [ids.shape[1] - 1]) as cache:
            model.model(input_ids=ids, use_cache=False)
        out.append(torch.stack([cache[l][0].float().cpu() for l in range(num_layers)]).half())
    return torch.stack(out)


def load_mmlu(per_category: int, with_options: bool, drop_categories: list[str], seed: int) -> tuple[list[str], list[str]]:
    """Seeded sample of up to per_category test questions per category. Returns (questions, categories)."""
    from datasets import load_dataset as hf_load_dataset

    rows = hf_load_dataset(MMLU_NAME, split="test")
    by_category = defaultdict(list)
    for row in rows:
        if row["category"] in drop_categories:
            continue
        question = row["question"].strip()
        if with_options:
            question += "\n" + "\n".join(f"{OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(row["options"]))
        by_category[row["category"]].append(question)
    missing = set(by_category) - set(EXPERTS)
    if missing:
        raise ValueError(f"No expert for categories {sorted(missing)}: add them to EXPERTS or --drop_categories")

    rng = random.Random(seed)
    questions, categories = [], []
    for category in sorted(by_category):
        items = by_category[category]
        for question in rng.sample(items, min(per_category, len(items))):
            questions.append(question)
            categories.append(category)
    print("MMLU-Pro questions per category: " + ", ".join(f"{c} {n}" for c, n in sorted(Counter(categories).items())))
    return questions, categories


def build_versions(questions: list[str], labels: np.ndarray, categories: list[str], speaker_format: str,
                   seed: int) -> tuple[dict[str, list[str]], np.ndarray]:
    """Prompts per version, and the category index of the mismatched speaker for every question."""
    rng = np.random.default_rng(seed)
    offsets = rng.integers(1, len(categories), size=len(labels))   # never 0, so never the question's own category
    speaker_labels = (labels + offsets) % len(categories)
    prompts = {
        "none": questions,
        "matched": [f"{speaker_prefix(speaker_format, EXPERTS[categories[c]])} {q}" for q, c in zip(questions, labels)],
        "mismatched": [f"{speaker_prefix(speaker_format, EXPERTS[categories[c]])} {q}" for q, c in zip(questions, speaker_labels)],
    }
    return prompts, speaker_labels


def stratified_split(labels: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices (keep, held_out), with `fraction` of every class held out."""
    rng = np.random.default_rng(seed)
    keep, held = [], []
    for c in np.unique(labels):
        idx = rng.permutation(np.where(labels == c)[0])
        n_held = max(1, int(round(fraction * len(idx))))
        held.extend(idx[:n_held])
        keep.extend(idx[n_held:])
    return np.sort(keep), np.sort(held)


def fit_logistic(x: torch.Tensor, y: torch.Tensor, num_classes: int, l2: float, max_iter: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Multinomial logistic regression on standardized x [n, d]: mean CE + l2 * ||W||^2, full-batch L-BFGS."""
    w = torch.zeros(num_classes, x.shape[1], device=x.device, requires_grad=True)
    b = torch.zeros(num_classes, device=x.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([w, b], lr=1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(x @ w.T + b, y) + l2 * (w ** 2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return w.detach(), b.detach()


def accuracy(x, y, w, b) -> float:
    return ((x @ w.T + b).argmax(-1) == y).float().mean().item()


def train_probe(acts: torch.Tensor, labels: np.ndarray, num_classes: int, args, device) -> tuple[dict, list[dict]]:
    """
    One probe per layer on acts [n, layers, d_model], with l2 chosen on a stratified validation split of the rows.
    Rows are one version per question, so the validation split is also by question.
    """
    fit_idx, val_idx = stratified_split(labels, args.test_fraction, args.seed)
    y = torch.tensor(labels.tolist(), device=device)

    probe = {"weight": [], "bias": [], "mean": [], "std": []}
    per_layer = []
    for layer in range(acts.shape[1]):
        x_all = acts[:, layer].float().to(device)

        def standardize(fit_rows):
            mean, std = x_all[fit_rows].mean(0), x_all[fit_rows].std(0) + 1e-6
            return (x_all - mean) / std, mean, std

        x, _, _ = standardize(fit_idx)
        val_acc = {}
        for l2 in args.l2_grid:
            w, b = fit_logistic(x[fit_idx], y[fit_idx], num_classes, l2, args.max_iter)
            val_acc[l2] = accuracy(x[val_idx], y[val_idx], w, b)
        best_l2 = max(args.l2_grid, key=lambda l2: (val_acc[l2], l2))   # ties go to the stronger penalty

        x, mean, std = standardize(slice(None))
        w, b = fit_logistic(x, y, num_classes, best_l2, args.max_iter)
        per_layer.append({
            "layer": layer,
            "l2": best_l2,
            "val_accuracy": val_acc[best_l2],
            "val_accuracy_by_l2": {str(k): v for k, v in val_acc.items()},
            "train_accuracy": accuracy(x, y, w, b),
        })
        for k, v in zip(["weight", "bias", "mean", "std"], [w, b, mean, std]):
            probe[k].append(v.cpu())
    return {k: torch.stack(v) for k, v in probe.items()}, per_layer


@torch.no_grad()
def probe_probs(acts: torch.Tensor, probe: dict) -> torch.Tensor:
    """p(subject) [n, layers, classes] from residuals [n, layers, d_model]."""
    x = (acts.float() - probe["mean"]) / probe["std"]
    logits = torch.einsum("nld,lcd->nlc", x, probe["weight"]) + probe["bias"]
    return torch.softmax(logits, dim=-1)


def claim_prompts(dataset: list[dict], roles: list[str], speaker_format: str) -> list[dict]:
    """One entry per (review, framing, role); role "none" is the bare claim."""
    prompts = []
    for example in dataset:
        for framing, claim in get_claims(example).items():
            for role in [NO_ROLE] + roles:
                question = claim if role == NO_ROLE else f"{speaker_prefix(speaker_format, role)} {claim}"
                prompts.append({"review_id": example["ReviewID"], "framing": framing, "role": role, "question": question})
    return prompts


def summarize_claims(records: list[dict], probe_name: str, num_layers: int) -> dict:
    def stats(group):
        out = []
        for layer in range(num_layers):
            tops = Counter(r["top_subject"][probe_name][layer] for r in group)
            out.append({
                "layer": layer,
                "mean_p_health": float(np.mean([r["p_health"][probe_name][layer] for r in group])),
                "frac_top_health": tops[TARGET_SUBJECT] / len(group),
                "top_subject_counts": dict(tops.most_common()),
            })
        return out

    summary = {}
    for role in sorted({r["role"] for r in records}, key=lambda r: (r != NO_ROLE, r)):
        group = [r for r in records if r["role"] == role]
        summary[role] = {"all": stats(group)}
        for framing in FRAMINGS:
            summary[role][framing] = stats([r for r in group if r["framing"] == framing])
    return summary


def plot(mmlu_summary: dict, claims_summary: dict, path: str) -> None:
    probe_names = list(PROBES)
    fig, axes = plt.subplots(2, len(probe_names), figsize=(5 * len(probe_names), 8), sharey=True)
    chance = 1 / len(mmlu_summary["categories"])
    for col, name in enumerate(probe_names):
        per_layer = mmlu_summary["probes"][name]["per_layer"]
        layers = [p["layer"] for p in per_layer]
        ax = axes[0, col]
        for version in VERSIONS:
            ax.plot(layers, [p["test_accuracy"][version] for p in per_layer], marker="o", ms=3, label=f"test: {version}")
        ax.plot(layers, [p["mismatched_speaker_rate"] for p in per_layer], ls="--", label="mismatched: predicts speaker")
        ax.axhline(chance, color="gray", lw=0.8, ls=":", label="chance")
        ax.set(title=f"{name} probe (trained on {' + '.join(PROBES[name])})", xlabel="layer", ylim=(0, 1.02))
        ax = axes[1, col]
        for role, by_framing in claims_summary[name].items():
            ax.plot(layers, [s["mean_p_health"] for s in by_framing["all"]], marker="o", ms=3, label=role)
        ax.set(title=f"{name} probe on the medical claims", xlabel="layer")
    axes[0, 0].set_ylabel("MMLU-Pro accuracy (question's category)")
    axes[1, 0].set_ylabel(f"mean p({TARGET_SUBJECT})")
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(title="role", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


def main(args):
    set_global_seed(SEED)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(args.model)
    num_layers = len(get_decoder_layers(model))

    # MMLU-Pro activations for every version
    questions, subject_names = load_mmlu(args.per_category, args.with_options, args.drop_categories, args.seed)
    categories = sorted(set(subject_names))
    if TARGET_SUBJECT not in categories:
        raise ValueError(f"'{TARGET_SUBJECT}' is not an MMLU-Pro category: {categories}")
    labels = np.array([categories.index(c) for c in subject_names])
    prompts, speaker_labels = build_versions(questions, labels, categories, args.speaker_format, args.seed)
    print("Example prompts: " + " | ".join(f"{v}: {prompts[v][0][:80]!r}" for v in VERSIONS))
    acts = {v: last_token_residuals(model, tokenizer, prompts[v], f"MMLU-Pro ({v})") for v in VERSIONS}
    train_idx, test_idx = stratified_split(labels, args.test_fraction, args.seed)
    print(f"Train {len(train_idx)} | test {len(test_idx)} questions | {len(categories)} categories | {num_layers} layers")

    # mixed: one version per train question, matched or none at random
    mixed_choice = np.random.default_rng(args.seed + 1).integers(0, 2, size=len(train_idx))
    train_sets = {
        "expert": acts["matched"][train_idx],
        "no_expert": acts["none"][train_idx],
        "mixed": torch.where(torch.tensor(mixed_choice.tolist(), dtype=torch.bool)[:, None, None],
                             acts["matched"][train_idx], acts["none"][train_idx]),
    }

    y_test, y_speaker = torch.tensor(labels[test_idx].tolist()), torch.tensor(speaker_labels[test_idx].tolist())
    probes, mmlu_summary = {}, {"model": args.model, "categories": categories, "experts": {c: EXPERTS[c] for c in categories},
                                "speaker_format": args.speaker_format, "with_options": args.with_options,
                                "n_train": len(train_idx), "n_test": len(test_idx), "probes": {}}
    for name, train_versions in PROBES.items():
        print(f"\n=== Probe: {name} (trained on {' + '.join(train_versions)}) ===")
        probe, per_layer = train_probe(train_sets[name], labels[train_idx], len(categories), args, device)
        probes[name] = probe
        preds = {v: probe_probs(acts[v][test_idx], probe).argmax(-1) for v in VERSIONS}   # [n_test, layers]
        for layer, p in enumerate(per_layer):
            p["test_accuracy"] = {v: (preds[v][:, layer] == y_test).float().mean().item() for v in VERSIONS}
            p["mismatched_speaker_rate"] = (preds["mismatched"][:, layer] == y_speaker).float().mean().item()

        best = max(per_layer, key=lambda p: p["val_accuracy"])
        confusion = np.zeros((len(categories), len(categories)), dtype=int)
        for v in train_versions:
            for true, pred in zip(y_test.tolist(), preds[v][:, best["layer"]].tolist()):
                confusion[true, pred] += 1
        mmlu_summary["probes"][name] = {
            "train_versions": train_versions, "per_layer": per_layer, "best_layer": best["layer"],
            "best_layer_test_accuracy": best["test_accuracy"],
            "best_layer_mismatched_speaker_rate": best["mismatched_speaker_rate"],
            "best_layer_confusion": {"versions": train_versions, "rows_true_cols_pred": confusion.tolist()},
            "best_layer_per_class_recall": {c: float(confusion[i, i] / max(confusion[i].sum(), 1)) for i, c in enumerate(categories)},
        }
        accs = ", ".join(f"{v} {best['test_accuracy'][v]:.3f}" for v in VERSIONS)
        print(f"Best layer {best['layer']} (by validation): test accuracy {accs} | mismatched predicts speaker "
              f"{best['mismatched_speaker_rate']:.3f} | chance {1 / len(categories):.3f}")

    torch.save({"probes": probes, "categories": categories, "experts": EXPERTS, "speaker_format": args.speaker_format,
                "model": args.model, "with_options": args.with_options},
               os.path.join(args.output_dir, "probe.pt"))
    with open(os.path.join(args.output_dir, "mmlu_summary.json"), "w", encoding="utf-8") as f:
        json.dump(mmlu_summary, f, indent=2)
    del acts, train_sets

    # Medical claims through every probe
    dataset = sample_reviews(load_dataset(args.input_path), args.num_reviews, seed=args.seed)
    claims = claim_prompts(dataset, args.roles, args.speaker_format)
    claim_acts = last_token_residuals(model, tokenizer, [c["question"] for c in claims], "claims")
    health = categories.index(TARGET_SUBJECT)
    probs = {name: probe_probs(claim_acts, probe) for name, probe in probes.items()}
    records = []
    with open(os.path.join(args.output_dir, "claims.jsonl"), "w", encoding="utf-8") as f:
        for i, c in enumerate(claims):
            record = c | {
                "p_health": {name: [round(v, 5) for v in p[i, :, health].tolist()] for name, p in probs.items()},
                "top_subject": {name: [categories[k] for k in p[i].argmax(-1).tolist()] for name, p in probs.items()},
            }
            records.append(record)
            f.write(json.dumps(record) + "\n")
    claims_summary = {name: summarize_claims(records, name, num_layers) for name in probes}
    with open(os.path.join(args.output_dir, "claims_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"n": len(records), "categories": categories, "probes": claims_summary}, f, indent=2)

    print(f"\nMedical claims ({len(dataset)} reviews), each probe at its best layer:")
    for name in probes:
        layer = mmlu_summary["probes"][name]["best_layer"]
        print(f"  {name} probe, layer {layer}")
        for role, by_framing in claims_summary[name].items():
            s = by_framing["all"][layer]
            top3 = ", ".join(f"{c} {n}" for c, n in list(s["top_subject_counts"].items())[:3])
            print(f"    {role:<8} mean p(health) {s['mean_p_health']:.3f} | top = health {s['frac_top_health']:.2f} | {top3}")
    plot(mmlu_summary, claims_summary, os.path.join(args.output_dir, "subject_probe.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train last-token subject probes on MMLU-Pro (with/without an expert speaker) and apply them to the medical claims")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="HuggingFace model name.")
    parser.add_argument("--input_path", required=True, help="dataset with the framed questions (extracted question set).")
    parser.add_argument("--output_dir", required=True, help="where the probes and summaries go.")
    parser.add_argument("--per_category", default=350, type=int, help="MMLU-Pro questions per category (the smallest has ~380).")
    parser.add_argument("--drop_categories", nargs="*", default=["other"], help="MMLU-Pro categories left out (need an entry in EXPERTS otherwise).")
    parser.add_argument("--speaker_format", default="My {speaker} said,", help="speaker prefix for experts and claim roles, e.g. 'The {speaker} said,'.")
    parser.add_argument("--with_options", action="store_true", help="append the answer options to the MMLU-Pro questions.")
    parser.add_argument("--test_fraction", default=0.2, type=float, help="held-out fraction of questions per category (also the validation fraction for l2).")
    parser.add_argument("--l2_grid", nargs="+", type=float, default=[1e-4, 1e-3, 1e-2, 1e-1], help="L2 penalties tried per layer.")
    parser.add_argument("--max_iter", default=200, type=int, help="L-BFGS iterations per fit.")
    parser.add_argument("--roles", nargs="*", default=["doctor", "friend", "enemy"], help="roles for 'My <role> said, <claim>' (the bare claim is always included).")
    parser.add_argument("--num_reviews", default=-1, type=int, help="seeded sample of reviews (-1: all).")
    parser.add_argument("--seed", default=SEED, type=int, help="seed for sampling questions, speakers, reviews and splits.")
    args = parser.parse_args()

    print("\nArguments:")
    for k, v in vars(args).items():
        print(f"{k:<16} {v}")
    print()
    main(args)
