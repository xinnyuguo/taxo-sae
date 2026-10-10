"""Paper figures for the relation world (layer 6), from the trained models and the Test F results.

1. figure data -> figures/readout_raw/ (skipped when present)
     binding_relation_L6{,_full,_hybrid}.npz   slot_binding.py / slot_binding_baseline.py  (held-out-template leaf questions)
     profiles_relation_L6.npz                  activation_profiles.py                      (training leaf questions)
     selectivity_matrix_L6_k1.json             scripts/eval_selectivity_matrix.py          (Test F operating point)
2. drawing -> figures/pdf/*.pdf, figures/png/*.png
     fig_binding_relation{,_baseline1,_baseline2,_combined}   binding heatmaps (titles: Test B readout rule)
     fig_activation_profile                                   most active units per concept, row-normalized
     fig_selectivity_matrix                                   ablate concept i, exact match on concept j
     fig_frag_relation                                        EffFeat / Top1C (results/E/efffeat.json)
Run from anywhere: python figures/make_figures.py   (GPU = first entry of $GPUS, default 0; needs `run.sh test` first)
"""
import json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; RAW = HERE / "readout_raw"
os.chdir(ROOT); RAW.mkdir(exist_ok=True)
PY = sys.executable
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("GPUS", "0").split(",")[0], PYTORCH_ALLOC_CONF="expandable_segments:True")


def make(out, cmd, env=None):
    if (RAW / out).exists(): return
    print("[data]", out, flush=True)
    subprocess.run(cmd, shell=True, check=True, env=dict(ENV, **(env or {})), stdout=subprocess.DEVNULL)


# ---- 1. figure data ----
make("binding_relation_L6.npz", f"{PY} figures/slot_binding.py relations 6 {RAW}/binding_relation_L6.npz")
for kind in ("full", "hybrid"):
    make(f"binding_relation_L6_{kind}.npz", f"{PY} figures/slot_binding_baseline.py relations 6 {kind} {RAW}/binding_relation_L6_{kind}.npz")
make("profiles_relation_L6.npz", f"{PY} figures/activation_profiles.py relations 6 {RAW}/profiles_relation_L6.npz")
sel = RAW / "sel"; sel.mkdir(exist_ok=True)
shutil.copy("results/F/selectivity_models.json", sel / "selectivity_L6.json")          # the Test F strengths (s0)
make("selectivity_matrix_L6_k1.json", f"{PY} scripts/eval_selectivity_matrix.py --tag L6 --k 1 --lm models/base_lm/final "
     f"--sel {sel} --out {RAW}/selectivity_matrix_L6_k1.json", {"HSAE_MODELS": "models", "HSAE_ACTS": "data/07_test_F"})

# ---- 2. drawing (functions copied verbatim from the paper's figure scripts) ----
sys.path.insert(0, str(HERE))
import draw_binding as B, draw_profile as P
for sys_, tag in ((None, ""), ("full", "_baseline1"), ("hybrid", "_baseline2")):
    B.figure("relation_L6", f"fig_binding_relation{tag}", height=3.6, right=0.22, sys=sys_)
B.combined("relation_L6", "fig_binding_relation_combined")
rel = np.load(RAW / "profiles_relation_L6.npz")
P.figure("relation_L6", "fig_activation_profile", P.pretty(list(rel["leaves"])),
         ["place of birth", "place of death", "employment", "education", "immediate family", "extended family"],
         height=4.0, gamma=1.0, crange=(0.0, 1.0))

plt = P.plt
from matplotlib.colors import LinearSegmentedColormap
CMAP = LinearSegmentedColormap.from_list("peachpurple", ["#ffffff", "#f6918f", "#c66ca6", "#7a5a9c"])
m = json.load(open(RAW / "selectivity_matrix_L6_k1.json"))
leaves = [s.replace("_", " ") for s in m["leaves"]]; classes = [s.replace("_", " ") for s in m["classes"]]
fig = plt.figure(figsize=(6.3, 3.35))
gs = fig.add_gridspec(1, 2, left=0.10, right=0.70, wspace=0.06)
gsr = fig.add_gridspec(2, 1, left=0.78, right=0.90, hspace=0.95, top=0.88, bottom=0.17)


def frame(ax):
    ax.tick_params(length=0); ax.grid(False)
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_color("#b0b0b0"); sp.set_linewidth(0.4)


for spec, key, title, ylab in ((gs[0, 0], "hier_leaf", "(a) Ours: leaf ablation", True), (gs[0, 1], "full_k1_leaf", "(b) Baseline1: leaf ablation", False)):
    ax = fig.add_subplot(spec); M = np.array(m[key]["matrix"]); n = len(leaves)
    im = ax.imshow(M, cmap=CMAP, vmin=0, vmax=1)
    for b in (2.5, 5.5, 8.5, 11.5, 14.5): ax.axhline(b, color="white", lw=0.9); ax.axvline(b, color="white", lw=0.9)
    ax.set_xticks(range(n)); ax.set_xticklabels(leaves, rotation=90, fontsize=5.2)
    ax.set_yticks(range(n)); ax.set_yticklabels(leaves if ylab else [], fontsize=5.2); frame(ax)
    ax.set_title(f"{title}\nown {np.diag(M).mean():.2f} / others {M[~np.eye(n, dtype=bool)].mean():.2f}", fontsize=7)
short = ["birth", "death", "employ.", "educ.", "imm. fam.", "ext. fam."]
for r, (key, title) in enumerate((("hier_class", "(c) Ours: class"), ("full_k1_class", "(d) Baseline1: class"))):
    ax = fig.add_subplot(gsr[r]); M = np.array(m[key]["matrix"]); n = len(classes)
    ax.imshow(M, cmap=CMAP, vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(short, rotation=90, fontsize=5)
    ax.set_yticks(range(n)); ax.set_yticklabels(short, fontsize=5); frame(ax)
    ax.set_title(f"{title}\nown {np.diag(M).mean():.2f} / others {M[~np.eye(n, dtype=bool)].mean():.2f}", fontsize=6.5)
cax = fig.add_axes([0.94, 0.30, 0.012, 0.45]); cb = fig.colorbar(im, cax=cax)
cb.ax.tick_params(labelsize=6); cb.set_label("exact match", fontsize=6.5); cb.outline.set_visible(False)
for ext in ("pdf", "png"): fig.savefig(HERE / f"fig_selectivity_matrix.{ext}", dpi=300)
plt.close(fig)
import draw_frag  # noqa: F401  (draws on import)

for ext in ("pdf", "png"):
    (HERE / ext).mkdir(exist_ok=True)
    for f in HERE.glob(f"fig_*.{ext}"): os.replace(f, HERE / ext / f.name)
print("figures in", HERE / "pdf", "and", HERE / "png")
