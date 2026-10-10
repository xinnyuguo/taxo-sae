"""Concept fragmentation (Test E), relation world layer 6, for the main text.

(a) EffFeat (log scale, lower is better) and (b) Top1C (higher is better) per taxonomy level.
Bars: mean over the level's concepts.
Data: results/E/efffeat.json (Test E).
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(os.path.join(OUT, "..", "results", "E", "efffeat.json")))["L6"]
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
SYS = (("ours", "Ours", "#4b5a9b"), ("B1", "Baseline1", "#f6918f"), ("B2", "Baseline2", "#7cc4a4"))
LV = ("leaf", "class", "root")
KEY = {"ours": "hier_concept_{}", "B1": "full_all_{}", "B2": "hybrid_all_{}"}


def entry(s, lv):
    return E["hybrid_concept_leaf" if (s == "B2" and lv == "leaf") else KEY[s].format(lv)]


fig, axes = plt.subplots(1, 2, figsize=(3.25, 1.75), gridspec_kw={"wspace": 0.42})
x = np.arange(3); wd = 0.27
for ax, metric, log in ((axes[0], "EffFeat", True), (axes[1], "Top1C", False)):
    for i, (s, name, col) in enumerate(SYS):
        for j, lv in enumerate(LV):
            e = entry(s, lv); xi = x[j] + (i - 1) * wd
            shared = s == "B2" and lv == "leaf"            # Baseline2's leaf level is our leaf module
            ax.bar(xi, e[metric], wd * 0.9, color=col, alpha=0.35 if shared else 0.9, edgecolor=col, lw=0.5,
                   label=name if j == 0 else None, zorder=2)
            v = e[metric]
            lab = (f"{v:.1f}" if v < 10 else f"{v:.0f}") if metric == "EffFeat" else f"{v:.2f}".lstrip("0")
            top = v
            if log or v > 0.1:                                 # tiny baseline Top1C bars (.02-.04) are given in the caption
                ax.text(xi, top * 1.25 if log else top + 0.025, lab, ha="center", va="bottom", fontsize=5, color="#333333")
    ax.set_xticks(x); ax.set_xticklabels(LV); ax.tick_params(length=2, width=0.5)
    ax.grid(axis="y", color="#e6e6e6", lw=0.5, zorder=0)
    if log:
        ax.set_yscale("log"); ax.set_ylim(0.7, 4000); ax.set_ylabel("EffFeat (log scale, ↓)")
    else:
        ax.set_ylim(0, 1.12); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_ylabel("Top1C (↑)")
    ax.set_title("(a) units per concept" if log else "(b) mass in the top unit", fontsize=7, pad=3)
from matplotlib.patches import Patch
h = [Patch(color=c, alpha=0.9, lw=0) for _, _, c in SYS]; l = [n for _, n, _ in SYS]
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.1), handlelength=1.0, columnspacing=1.2)
for ext in ("pdf", "png"): fig.savefig(f"{OUT}/fig_frag_relation.{ext}", dpi=300)
print("wrote fig_frag_relation(.pdf/.png)")
