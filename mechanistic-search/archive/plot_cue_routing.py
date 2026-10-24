import json, math, statistics as st, collections, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = sys.argv[1]
OUT = sys.argv[2]
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#111214", "#4d5058", "#e6e8ec"

def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def long_form(path, setups, min_effect=3.0):
    R = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in load(path):
        c = r["metrics"]["clean"]
        d = lambda m: m["logp_pos_resp"] - m["logp_neg_resp"]
        dt = d(c["target_prompt"]); e = d(c["source_prompt"]) - dt
        if r["identical_responses"] or abs(e) < min_effect:
            continue
        for s in setups:
            for L, m in r["metrics"]["patched"].get(s, {}).items():
                R[s][int(L)].append((d(m) - dt) / e)
    return R

def yes_no(path, setups, min_effect=0.5):
    R = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in load(path):
        c = r["metrics"]["clean"]
        dt = c["target_prompt"]["d"]; e = c["source_prompt"]["d"] - dt
        if abs(e) < min_effect:
            continue
        for s in setups:
            for L, v in r["metrics"]["patched"].get(s, {}).items():
                R[s][int(L)].append((v - dt) / e)
    return R

def curve(ax, by_layer, color, ls, label, marker):
    layers = sorted(by_layer)
    m = [st.mean(by_layer[L]) for L in layers]
    se = [st.stdev(by_layer[L]) / math.sqrt(len(by_layer[L])) for L in layers]
    n = len(by_layer[layers[0]])
    ax.fill_between(layers, [a - b for a, b in zip(m, se)], [a + b for a, b in zip(m, se)], color=color, alpha=0.12, linewidth=0)
    ax.plot(layers, m, color=color, linestyle=ls, linewidth=2, marker=marker, markersize=5 if len(layers) < 12 else 0,
            markeredgecolor="white", markeredgewidth=1, label=f"{label} (n={n})")

lf_fr = long_form(f"{ROOT}/span/span_patching_pairs.jsonl", ["diff", "after"])
lf_df = long_form(f"{ROOT}/doctor_friend/span/span_patching_pairs.jsonl", ["diff", "after"])
yn_fr = yes_no(f"{ROOT}/yes_no/yes_no_patching_pairs.jsonl", ["diff", "last"])
yn_df = yes_no(f"{ROOT}/doctor_friend/yes_no/yes_no_patching_pairs.jsonl", ["diff", "last"])
yn_db = yes_no(f"{ROOT}/doctor/yes_no/yes_no_patching_pairs.jsonl", ["last"])

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK})
fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), sharey=True)

ax = axes[0]
curve(ax, lf_fr["diff"], BLUE, "-", "Framing: the cue word (effective → ineffective)", "o")
curve(ax, lf_fr["after"], BLUE, "--", "Framing: every token after the word", "o")
curve(ax, lf_df["diff"], ORANGE, "-", "Authority: the cue word (doctor → friend)", "s")
curve(ax, lf_df["after"], ORANGE, "--", "Authority: every token after the word", "s")
ax.set_title("A. Long-form answer (≈270 tokens)\nWhere the cue sits in the prompt", loc="left", fontsize=12, fontweight="bold")
ax.annotate("doctor/friend identity has moved\noff the word by layer 10", xy=(10, 0.63), xytext=(17, 0.74),
            fontsize=9.5, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.annotate("framing mostly stays\nat its word", xy=(10, 0.33), xytext=(17, 0.50),
            fontsize=9.5, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))

ax = axes[1]
curve(ax, yn_fr["diff"], BLUE, "-", "Framing: the cue word", "o")
curve(ax, yn_fr["last"], BLUE, ":", "Framing: last prompt token", "o")
curve(ax, yn_df["diff"], ORANGE, "-", "Doctor → friend: the cue word", "s")
curve(ax, yn_df["last"], ORANGE, ":", "Doctor → friend: last prompt token", "s")
curve(ax, yn_db["last"], AQUA, ":", "Doctor → no prefix: last prompt token", "^")
ax.axvspan(12.5, 15.5, color=GRID, alpha=0.6, linewidth=0)
ax.text(14, 1.045, "hand-off, layers 13–15", ha="center", va="center", fontsize=9.5, color=INK2)
ax.set_title("B. Single-token Yes/No verdict\nWhen the cue reaches the answer position", loc="left", fontsize=12, fontweight="bold")

for ax in axes:
    ax.axhline(0, color=INK2, lw=0.7); ax.axhline(1, color=INK2, lw=0.7, ls=(0, (2, 3)))
    ax.set_xlim(-0.8, 31.8); ax.set_ylim(-0.08, 1.08)
    ax.set_xlabel("Layer patched")
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", fontsize=9, frameon=False, bbox_to_anchor=(0.0, -0.16), ncol=1, handlelength=4.5)
axes[0].set_ylabel("Recovery of the clean effect\n(0 = no change, 1 = behaves like the source prompt)")
fig.text(0.01, 0.012, "Llama 3 8B Instruct. Mean ± SEM (shaded).\n"
         "A: teacher-forced log-likelihood of the two clean answers; layers 0–30 every 5; pairs with effect ≥ 3.\n"
         "B: log P(Yes) − log P(No) at the last prompt position; every layer; pairs with |effect| ≥ 0.5; 18 of 30 reviews (run in progress).",
         fontsize=8.5, color=INK2, va="bottom")
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(OUT, dpi=200, facecolor="white")
print("saved", OUT)
