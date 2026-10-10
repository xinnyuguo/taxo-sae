"""Activation-profile heatmaps (AlignSAE-style): concepts x most-activated features, row-normalized.

Columns are chosen WITHOUT concept labels, as in AlignSAE:
Left  (Baseline1): the N features with the highest mean activation over all concepts, sorted by mean activation.
Right (ours):      the N free slots with the highest mean activation, sorted by mean activation, followed by the
                   |C| concept slots (outlined; one fixed slot per concept, in row order).
Each row is divided by its maximum over the displayed columns.
Data: readout_raw/profiles_*.npz from activation_profiles.py (same A[c, j] as Test E / EffFeat).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, PowerNorm, Normalize
from matplotlib.patches import FancyBboxPatch

OUT = os.path.dirname(os.path.abspath(__file__))   # figures/: reads OUT/readout_raw, writes OUT/*.pdf|png
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 7, "pdf.fonttype": 42,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
from matplotlib.colors import LinearSegmentedColormap
CMAP = LinearSegmentedColormap.from_list("teal_purple", ["#eef6f6", "#3e9fa1", "#4b4a93"])   # user-chosen teal -> slate purple, linear
N = 100
FREE_FRAC = None          # None = every column has the same width; a number squeezes the free slots into that share


def baseline_cols(A):
    return np.argsort(-A.mean(0))[:N]                                 # label-free, sorted by mean activation (AlignSAE)


def ours_cols(A, nf, n, order):
    free = np.argsort(-A[:, :nf].mean(0))[:N]                       # top-N free slots, sorted by mean activation (AlignSAE)
    return np.concatenate([free, nf + np.asarray(order)])            # concept slots in row order


def norm(M): return M / (M.max(1, keepdims=True) + 1e-12)


def pretty(names, short=None): return short or [s.replace("_", " ") for s in names]


def figure(world, fname, leaf_names, class_names, leaf_order=None, height=3.6, gamma=1.0, crange=(0.0, 1.0)):
    cmap = ListedColormap(CMAP(np.linspace(*crange, 256)))       # optionally use only the middle of magma
    cnorm = PowerNorm(gamma, vmin=0, vmax=1) if gamma != 1.0 else Normalize(vmin=0, vmax=1)
    d = np.load(f"{OUT}/readout_raw/profiles_{world}.npz"); nf = int(d["n_free"])
    raw_leaves = list(d["leaves"]); perm = [raw_leaves.index(x) for x in leaf_order] if leaf_order else list(range(len(raw_leaves)))
    fig, axes = plt.subplots(2, 2, figsize=(6.3, height), gridspec_kw={"height_ratios": [len(perm), len(class_names) * 1.6], "hspace": 0.32, "wspace": 0.04})
    for r, (lv, names) in enumerate((("leaf", leaf_names), ("class", class_names))):
        n = len(names); P = perm if lv == "leaf" else list(range(n))
        for c, (sys, title) in enumerate((("full", "Baseline1 (unsupervised SAE)"), ("hier", "Ours (hierarchical SAE)"))):
            A = d[f"{sys}_{lv}"][P]
            cols = baseline_cols(A) if sys == "full" else ours_cols(A, nf, n, P)
            M = norm(A[:, cols]); ax = axes[r, c]
            # x layout in [0, 1]: Baseline1 columns share the width uniformly; for ours the free slots are
            # squeezed into FREE_FRAC of the width and the n concept slots get the rest (wider cells)
            if sys == "hier":
                nfc = len(cols) - n; ff = FREE_FRAC if FREE_FRAC is not None else nfc / len(cols)
                xe = np.concatenate([np.linspace(0, ff, nfc + 1), np.linspace(ff, 1, n + 1)[1:]])
            else:
                xe = np.linspace(0, 1, len(cols) + 1)
            im = ax.pcolormesh(xe, np.arange(n + 1) - 0.5, M, cmap=cmap, norm=cnorm, shading="flat", rasterized=True)
            ax.set_xlim(0, 1); ax.set_ylim(n - 0.5, -0.5)
            ax.set_yticks(range(n)); ax.set_yticklabels(names if c == 0 else [], fontsize=6)
            ax.set_xticks([]); ax.tick_params(length=0)
            for sp in ax.spines.values(): sp.set_visible(False)
            if sys == "hier":
                ff = FREE_FRAC if FREE_FRAC is not None else (len(cols) - n) / len(cols); cw = (1 - ff) / n
                ax.add_patch(FancyBboxPatch((ff + 0.1 * cw, -0.46), 1 - ff - 0.2 * cw, n - 0.08, zorder=5, clip_on=False,
                                            boxstyle="round,pad=0.004,rounding_size=0.012", fill=False, ec="#2a78d6", lw=1.3,
                                            mutation_aspect=n / 0.6))
                ax.text(1.0, n - 0.2, "concept slots", ha="right", va="top", fontsize=6, color="#2a78d6")
                ax.text(ff / 2, n - 0.2, f"top-{N} free slots", ha="center", va="top", fontsize=6, color="#3c3c3c")
            else:
                ax.text(0.5, n - 0.2, f"top-{N} most active features (sorted by mean activation)", ha="center", va="top", fontsize=6, color="#3c3c3c")
            if r == 0: ax.set_title(title, fontsize=7.5)
        axes[r, 0].set_ylabel(f"{lv} concepts", fontsize=7)
    fig.subplots_adjust(bottom=0.55 / height)                       # fixed margins in inches, any figure height
    l0 = axes[1, 0].get_position().x0; l1 = axes[1, 1].get_position().x1
    cax = fig.add_axes([l0, 0.20 / height, l1 - l0, 0.08 / height])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb.ax.set_xticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=6, fontweight="bold")
    cb.ax.tick_params(direction="in", length=3, width=0.7, colors="black")
    cb.outline.set_edgecolor("black"); cb.outline.set_linewidth(0.8)
    cb.ax.annotate("Normalized Activation", xy=(1.0, 0), xycoords="axes fraction", xytext=(0, -14), textcoords="offset points",
                   ha="right", va="top", fontsize=6, fontweight="bold")
    for ext in ("pdf", "png"): fig.savefig(f"{OUT}/{fname}.{ext}", dpi=300)
    plt.close(fig)

