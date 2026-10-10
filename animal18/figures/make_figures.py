"""Paper figures for the 18-animal world (layer 9), from the trained models and the Test F results.

1. figure data -> figures/readout_raw/ (skipped when present)
     binding_animal_L9{,_full,_hybrid}.npz   slot_binding.py / slot_binding_baseline.py  (held-out-template leaf questions)
     profiles_animal_L9.npz                  activation_profiles.py                      (training leaf questions)
     selectivity_matrix_L9_k1.json           scripts/eval_selectivity_matrix.py          (Test F operating point)
2. drawing -> figures/pdf/*.pdf, figures/png/*.png
     fig_binding_animal{,_baseline1,_baseline2,_combined}, fig_activation_profile_animal, fig_selectivity_matrix_animal,
     fig_frag_animal.  Leaves are grouped by class.
Run from anywhere: python figures/make_figures.py   (GPU = first entry of $GPUS, default 0; needs `run.sh test` first)
"""
import json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; RAW = HERE / "readout_raw"
os.chdir(ROOT); RAW.mkdir(exist_ok=True)
PY = sys.executable
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("GPUS", "0").split(",")[0], PYTORCH_ALLOC_CONF="expandable_segments:True")
ORDER = ["cow", "antelope", "bat", "cockatoo", "penguin", "eagle", "frog", "toad", "salamander",
         "salmon", "shark", "clownfish", "firefly", "woodworm", "honeybee", "octopus", "snail", "clam"]
CLASSES = ["mammal", "bird", "amphibian", "fish", "insect", "mollusc"]


def make(out, cmd, env=None):
    if (RAW / out).exists(): return
    print("[data]", out, flush=True)
    subprocess.run(cmd, shell=True, check=True, env=dict(ENV, **(env or {})), stdout=subprocess.DEVNULL)


# ---- 1. figure data ----
make("binding_animal_L9.npz", f"{PY} figures/slot_binding.py concepts 9 {RAW}/binding_animal_L9.npz")
for kind in ("full", "hybrid"):
    make(f"binding_animal_L9_{kind}.npz", f"{PY} figures/slot_binding_baseline.py concepts 9 {kind} {RAW}/binding_animal_L9_{kind}.npz")
make("profiles_animal_L9.npz", f"{PY} figures/activation_profiles.py concepts 9 {RAW}/profiles_animal_L9.npz")
sel = RAW / "sel"; sel.mkdir(exist_ok=True)
shutil.copy("results/F/selectivity_models.json", sel / "selectivity_L9.json")          # the Test F strengths (s0)
make("selectivity_matrix_L9_k1.json", f"{PY} scripts/eval_selectivity_matrix.py --tag L9 --k 1 --lm models/base_lm "
     f"--sel {sel} --out {RAW}/selectivity_matrix_L9_k1.json", {"HSAE_MODELS": "models", "HSAE_ACTS": "data/07_test_F"})

# ---- 2. drawing (functions copied verbatim from the paper's figure scripts) ----
sys.path.insert(0, str(HERE))
import draw_binding as B, draw_profile as P
for sys_, tag in ((None, ""), ("full", "_baseline1"), ("hybrid", "_baseline2")):
    B.figure("animal_L9", f"fig_binding_animal{tag}", order=ORDER, height=3.6, right=0.22, sys=sys_)
B.combined("animal_L9", "fig_binding_animal_combined", order=ORDER)
P.figure("animal_L9", "fig_activation_profile_animal", ORDER, CLASSES, leaf_order=ORDER, height=4.0)

plt = P.plt
from matplotlib.colors import LinearSegmentedColormap
CMAP = LinearSegmentedColormap.from_list("peachpurple", ["#ffffff", "#f6918f", "#c66ca6", "#7a5a9c"])
m = json.load(open(RAW / "selectivity_matrix_L9_k1.json"))
perm = [m["leaves"].index(x) for x in ORDER]; cperm = [m["classes"].index(x) for x in CLASSES]
fig = plt.figure(figsize=(7.4, 3.6))
gs = fig.add_gridspec(1, 2, left=0.08, right=0.70, wspace=0.06)
gsr = fig.add_gridspec(2, 1, left=0.77, right=0.91, hspace=0.9, top=0.88, bottom=0.15)


def panel(ax, M, names, title, ylab, seps, fs, text=True):
    n = len(names); im = ax.imshow(M, cmap=CMAP, vmin=0, vmax=1)
    for b in seps: ax.axhline(b, color="white", lw=0.9); ax.axvline(b, color="white", lw=0.9)
    if text:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{M[i, j]:.2f}".lstrip("0") if M[i, j] < 1 else "1", ha="center", va="center",
                        fontsize=fs - 1.2, color="white" if M[i, j] > 0.55 else "#3c3c3c")
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=90, fontsize=fs)
    ax.set_yticks(range(n)); ax.set_yticklabels(names if ylab else [], fontsize=fs)
    ax.tick_params(length=0); ax.grid(False)
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_color("#b0b0b0"); sp.set_linewidth(0.4)
    ax.set_title(f"{title}\nown {np.diag(M).mean():.2f} / others {M[~np.eye(n, dtype=bool)].mean():.2f}", fontsize=7)
    return im


for c, (key, title) in enumerate((("hier_leaf", "(a) Ours: leaf ablation"), ("full_k1_leaf", "(b) Baseline1: leaf ablation"))):
    im = panel(fig.add_subplot(gs[0, c]), np.array(m[key]["matrix"])[np.ix_(perm, perm)], ORDER, title, c == 0,
               (2.5, 5.5, 8.5, 11.5, 14.5), 5.2, text=False)
short = ["mamm.", "bird", "amph.", "fish", "insect", "moll."]
for c, (key, title) in enumerate((("hier_class", "(c) Ours: class"), ("full_k1_class", "(d) Baseline1: class"))):
    panel(fig.add_subplot(gsr[c, 0]), np.array(m[key]["matrix"])[np.ix_(cperm, cperm)], short, title, True, (), 5.0)
cax = fig.add_axes([0.95, 0.30, 0.012, 0.45]); cb = fig.colorbar(im, cax=cax)
cb.ax.tick_params(labelsize=6); cb.set_label("exact match", fontsize=6.5); cb.outline.set_visible(False)
for ext in ("pdf", "png"): fig.savefig(HERE / f"fig_selectivity_matrix_animal.{ext}", dpi=300)
plt.close(fig)
import draw_frag  # noqa: F401  (draws on import)

for ext in ("pdf", "png"):
    (HERE / ext).mkdir(exist_ok=True)
    for f in HERE.glob(f"fig_*.{ext}"): os.replace(f, HERE / ext / f.name)
print("figures in", HERE / "pdf", "and", HERE / "png")
