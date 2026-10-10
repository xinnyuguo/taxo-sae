"""Concept x concept-slot binding matrices (raw, unclipped slot activations), for the binding heatmaps.

M[c, s] = mean over held-out-template questions of concept c of z_s, for the concept slots s of each level
(leaf, class, root; class/root labels derived through the taxonomy). Binding accuracy (Test B rule: diagonal accuracy with abstention) = argmax over the level's
concept slots == the question's concept at that level.

Usage (from the repository root; run by figures/make_figures.py):
  python slot_binding.py <relations|concepts> <layer> <out.npz>
"""
import sys, os, pickle
world, layer, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import _bootstrap  # noqa: F401
import numpy as np, torch
from hsae.common import Tree
from hsae.systems import System


def diag_acc(z, y, n, tau=0.1):
    """Test B readout rule: argmax over the level's n slots / features, abstaining (scored wrong) when the top value does
    not beat the runner-up by more than tau * |top|; AlignSAE diagonal accuracy = mean over concepts of per-concept accuracy."""
    s = np.sort(z, 1); ok = (s[:, -1] - s[:, -2]) > tau * np.abs(s[:, -1])
    hit = (z.argmax(1) == y) & ok
    return np.mean([hit[y == c].mean() for c in range(n) if (y == c).any()])

tree = Tree("data/00_taxonomy/tree.json")
PL = "data/00_taxonomy/property_labels.json"
T = f"L{layer}"
rows = pickle.load(open(f"data/01_activations/{T}_ooda_layer{layer}.pkl", "rb"))
H = torch.tensor(np.stack([r["h"] for r in rows]), dtype=torch.float32)
leaf = np.array([r["concept_idx"] for r in rows])
cls = np.array([tree.class_idx_of_leaf(tree.leaves[i]) for i in leaf])
root = np.array([tree.root_idx_of_class(tree.classes[i]) for i in cls])
S = System("hier", tree, "cuda", hier_ckpt=f"models/hier_sae_{T}/hier_value_final.pt", property_labels=PL)
nf = S.n_free; names = (tree.leaves, tree.classes, tree.roots); labs = (leaf, cls, root)
Z = [[], [], []]
with torch.no_grad():
    for s in range(0, len(H), 512):
        for lv, (z, _) in enumerate(S.forward_stack(H[s:s + 512].cuda())[:3]):
            Z[lv].append(z[:, nf:nf + len(names[lv])].float().cpu().numpy())
res = {"n_questions": len(rows)}
for lv, key in enumerate(("leaf", "class", "root")):
    z = np.concatenate(Z[lv]); n = len(names[lv])
    res[f"{key}_names"] = np.array(names[lv])
    res[f"{key}_M"] = np.stack([z[labs[lv] == c].mean(0) for c in range(n)])
    res[f"{key}_acc_argmax"] = np.array((z.argmax(1) == labs[lv]).mean())
    res[f"{key}_acc"] = np.array(diag_acc(z, labs[lv], n))
    print(key, "binding acc %.3f" % res[f"{key}_acc"])
np.savez_compressed(out, **res); print("saved", out)
