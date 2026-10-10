"""Concept x concept-slot binding heatmaps (style of the AlignSAE binding figure).

Rows: the concept a held-out-template question asks about (leaf / class / root, derived through the taxonomy).
Columns: the concept slots of that level. Cell: mean raw (unclipped) slot activation, divided by the row's
maximum (the concept's own slot when binding is correct) and clipped to [-1, 1]. Panel titles give the argmax binding accuracy.
Data: readout_raw/binding_*.npz from slot_binding.py.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

OUT = os.path.dirname(os.path.abspath(__file__))   # figures/: reads OUT/readout_raw, writes OUT/*.pdf|png
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 7, "pdf.fonttype": 42,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
# positive half: white -> mint -> teal -> indigo (palette of the reference bar chart);
# negative half shown only as pale grey so negative off-diagonal cells do not compete with the diagonal.
# The colour bar still spans -1..1.
_pos = LinearSegmentedColormap.from_list("pos", ["#ffffff", "#7cc4a4", "#2b8fa3", "#4b5a9b"])
_neg = LinearSegmentedColormap.from_list("neg", ["#cfd3d8", "#ffffff"])
CMAP = ListedColormap(np.vstack([_neg(np.linspace(0, 1, 128)), _pos(np.linspace(0, 1, 128))]))


SHORT = {"professional background": "professional bg.", "immediate family": "immediate fam.", "extended family": "extended fam.",
         "invert": "invertebrate", "aquatic": "aquatic animal"}


def pretty(xs): return [str(x).replace("_", " ") for x in xs]


def figure(world, fname, order=None, height=3.4, right=0.36, sys=None, label="slot"):
    d = dict(np.load(f"{OUT}/readout_raw/binding_{world}.npz"))
    if sys:                                            # baseline: its own class/root (and leaf for Baseline1)
        b = np.load(f"{OUT}/readout_raw/binding_{world}_{sys}.npz")
        for k in b.files: d[k] = b[k]
    fig = plt.figure(figsize=(6.3, height))
    gl = fig.add_gridspec(1, 1, left=0.17, right=0.55, top=0.88, bottom=0.12)
    gr = fig.add_gridspec(2, 1, left=0.80 - right / 2, right=0.80, top=0.88, bottom=0.12, height_ratios=[2, 1.2], hspace=0.75)
    axes = [fig.add_subplot(gl[0]), fig.add_subplot(gr[0]), fig.add_subplot(gr[1])]
    for ax, lv in zip(axes, ("leaf", "class", "root")):
        M = d[f"{lv}_M"]; names = list(d[f"{lv}_names"])
        if lv == "leaf" and order:
            p = [names.index(x) for x in order]; M = M[np.ix_(p, p)]; names = order
        M = np.clip(M / (M.max(1, keepdims=True) + 1e-12), -1, 1)
        n = len(names)
        im = ax.imshow(M, cmap=CMAP, vmin=-1, vmax=1, aspect="equal", interpolation="nearest")
        ax.set_xticks(np.arange(n + 1) - 0.5, minor=True); ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", lw=0.5); ax.tick_params(which="minor", length=0)
        ax.set_xticks(range(n)); ax.set_xticklabels(range(n), fontsize=5.5)
        ax.set_yticks(range(n)); ax.set_yticklabels([SHORT.get(x, x) for x in pretty(names)], fontsize=5.5 if n > 8 else 6)
        ax.tick_params(length=0)
        for sp in ax.spines.values(): sp.set_visible(True); sp.set_color("#b0b0b0"); sp.set_linewidth(0.4)   # frame: white cells at the edge
        shared = sys == "hybrid" and lv == "leaf"
        what = "slot" if (not sys or shared) else "feature"
        ax.set_xlabel(f"{lv} {what} index", fontsize=6.5)
        head = f"{lv.capitalize()} {what}s" + (" (= ours)" if shared else "")
        ax.set_title(f"{head}\nbinding accuracy {100 * float(d[f'{lv}_acc']):.1f}%", fontsize=7)
    axes[0].set_ylabel("queried concept", fontsize=6.5)
    cax = fig.add_axes([0.86, 0.25, 0.013, 0.5])
    cb = fig.colorbar(im, cax=cax, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.ax.tick_params(labelsize=6); cb.set_label("Normalized Activation", fontsize=6.5); cb.outline.set_visible(False)
    for ext in ("pdf", "png"): fig.savefig(f"{OUT}/{fname}.{ext}", dpi=300)
    plt.close(fig)


def load(world, sys):
    d = dict(np.load(f"{OUT}/readout_raw/binding_{world}.npz"))
    if sys:
        b = np.load(f"{OUT}/readout_raw/binding_{world}_{sys}.npz")
        for k in b.files: d[k] = b[k]
    return d


def heat(ax, M, names, ylab, fs):
    M = np.clip(M / (M.max(1, keepdims=True) + 1e-12), -1, 1); n = len(names)
    im = ax.imshow(M, cmap=CMAP, vmin=-1, vmax=1, aspect="equal", interpolation="nearest")
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True); ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", lw=0.5); ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(n)); ax.set_xticklabels(range(n), fontsize=fs)
    ax.set_yticks(range(n)); ax.set_yticklabels([SHORT.get(x, x) for x in pretty(names)] if ylab else [], fontsize=fs)
    ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_color("#b0b0b0"); sp.set_linewidth(0.4)   # frame: white cells at the edge
    return im


def combined(world, fname, order=None, leaf=1.75, s=0.6):
    """Left: leaf heatmaps of Ours and Baseline1 (Baseline2 reuses our leaf slots, so it is not repeated).
    Right: a 2x3 grid of the class (top) and root (bottom) panels of Ours / Baseline1 / Baseline2."""
    systems = ((None, "Ours"), ("full", "Baseline1"), ("hybrid", "Baseline2"))
    D = [load(world, sy) for sy, _ in systems]
    x0, gap, yb = 0.95, 0.12, 0.32                              # left margin, gap between leaf panels, bottom margin
    xs = x0 + 2 * leaf + gap + 0.68                             # small grid starts after room for its y labels
    pitch = s + 0.14
    W, H = xs + 3 * pitch + 0.6, yb + leaf + 0.32
    fig = plt.figure(figsize=(W, H))
    ax_in = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])
    tag = iter("abcdefgh")
    for c, ((sy, name), d) in enumerate(zip(systems[:2], D[:2])):
        M = d["leaf_M"]; names = list(d["leaf_names"])
        if order:
            p = [names.index(x) for x in order]; M = M[np.ix_(p, p)]; names = order
        ax = ax_in(x0 + c * (leaf + gap), yb, leaf, leaf)
        im = heat(ax, M, names, c == 0, 5.2)
        what = "feature" if sy == "full" else "slot"
        ax.set_xlabel(f"leaf {what} index", fontsize=6.5)
        ax.set_title(f"({next(tag)}) {name}" + (" (= Baseline2)" if sy is None else "") + f": leaf {what}s"
                     + f"\nbinding accuracy {100 * float(d['leaf_acc']):.1f}%", fontsize=7)
        if c == 0: ax.set_ylabel("queried concept", fontsize=6.5)
    for r, lv in enumerate(("class", "root")):
        y = yb + (1 - r) * (s + 0.55)
        for c, ((sy, name), d) in enumerate(zip(systems, D)):
            ax = ax_in(xs + c * pitch, y, s, s)
            heat(ax, d[f"{lv}_M"], list(d[f"{lv}_names"]), c == 0, 5)
            ax.set_xlabel("slot" if sy is None else "feature", fontsize=5.5, labelpad=1.5)
            ax.set_title(f"({next(tag)}) {name}\n{lv} {100 * float(d[f'{lv}_acc']):.1f}%", fontsize=5.8, pad=3)
    cax = ax_in(xs + 2 * pitch + s + 0.1, yb + 0.25 * leaf, 0.07, 0.5 * leaf)
    cb = fig.colorbar(im, cax=cax, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.ax.tick_params(labelsize=6); cb.set_label("Normalized Activation", fontsize=6.5); cb.outline.set_visible(False)
    for ext in ("pdf", "png"): fig.savefig(f"{OUT}/{fname}.{ext}", dpi=300)
    plt.close(fig)


