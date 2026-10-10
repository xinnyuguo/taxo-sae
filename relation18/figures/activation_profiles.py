"""Concept x feature mean activations (the quantity behind EffFeat), saved for the activation-profile heatmaps.

A[c, j] = mean over the training leaf questions of concept c of max(z_j, 0), for every feature j of each level.
Concepts: leaves at level 1, classes at level 2 (derived from the leaf label through the taxonomy).

Usage (from the repository root; run by figures/make_figures.py):
  python activation_profiles.py <relations|concepts> <layer> <out.npz>
"""
import sys, os, pickle
world, layer, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import _bootstrap  # noqa: F401  (cuBLAS fallback, device order)
import numpy as np, torch
from hsae.common import Tree
from hsae.systems import System

tree = Tree("data/00_taxonomy/tree.json")
PL = "data/00_taxonomy/property_labels.json"
T = f"L{layer}"; DEV = "cuda"
rows = pickle.load(open(f"data/01_activations/{T}_train_layer{layer}.pkl", "rb"))
H = torch.tensor(np.stack([r["h"] for r in rows]), dtype=torch.float32)
leaf = torch.tensor([r["concept_idx"] for r in rows])
cls = torch.tensor([tree.class_idx_of_leaf(tree.leaves[i]) for i in leaf.tolist()])
labels = (leaf, cls); ncon = (len(tree.leaves), len(tree.classes))
systems = {
    "hier": System("hier", tree, DEV, hier_ckpt=f"models/hier_sae_{T}/hier_value_final.pt", property_labels=PL),
    "full": System("full", tree, DEV, unsup_ckpt=f"models/baseline1_{T}/unsup_full_final.pt", k=1),
    "hybrid": System("hybrid", tree, DEV, hier_ckpt=f"models/hier_sae_{T}/hier_value_final.pt",
                     unsup_ckpt=f"models/baseline2_{T}/unsup_anc_final.pt", property_labels=PL, k=1),
}
res = {"leaves": np.array(tree.leaves), "classes": np.array(tree.classes)}
for name, S in systems.items():
    acc = None
    with torch.no_grad():
        for s in range(0, len(H), 512):
            zs = [z.clamp(min=0).float() for (z, _) in S.forward_stack(H[s:s + 512].to(DEV))][:2]
            if acc is None: acc = [torch.zeros(n, z.shape[1], device=DEV) for n, z in zip(ncon, zs)]
            for lv in range(2): acc[lv].index_add_(0, labels[lv][s:s + 512].to(DEV), zs[lv])
    for lv, lvname in enumerate(("leaf", "class")):
        cnt = torch.bincount(labels[lv], minlength=ncon[lv]).float().to(DEV)
        res[f"{name}_{lvname}"] = (acc[lv] / cnt[:, None]).cpu().numpy().astype(np.float32)
        feats = S.leaf_feats if lvname == "leaf" else S.class_feats       # the slot / k=1 feature of each concept
        res[f"{name}_{lvname}_sel"] = np.array([feats(c)[0] for c in range(ncon[lv])])
    if name == "hier": res["n_free"] = np.array(S.n_free)
    del S; torch.cuda.empty_cache()
np.savez_compressed(out, **res); print("saved", out, {k: v.shape for k, v in res.items()})
