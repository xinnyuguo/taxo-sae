"""Concept x concept-feature binding matrices for the baselines (k=1 differential feature per concept).

Same questions and layout as slot_binding.py: M[c, j] = mean activation, on the held-out-template questions of
concept c, of the feature that the baseline assigns to concept j. Binding accuracy (Test B rule: diagonal accuracy with abstention) = argmax over these features
(= the Test B readout rule). Baseline2's leaf level is our leaf module and is not recomputed.

Usage (from the repository root; run by figures/make_figures.py):
  python slot_binding_baseline.py <relations|concepts> <layer> <full|hybrid> <out.npz>
"""
import sys, os, pickle
world, layer, kind, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import _bootstrap  # noqa: F401
import numpy as np, torch
from hsae.common import Tree
from hsae.systems import System, feature_sets


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
H_CK = f"models/hier_sae_{T}/hier_value_final.pt"
U_CK = f"models/baseline1_{T}/unsup_full_final.pt" if kind == "full" else f"models/baseline2_{T}/unsup_anc_final.pt"
S = System(kind, tree, "cuda", hier_ckpt=H_CK, unsup_ckpt=U_CK, property_labels=PL, k=1)
feats = {"class": [S.CF[c][0] for c in range(len(tree.classes))], "root": [S.RF[r][0] for r in range(len(tree.roots))]}
if kind == "full":
    LF = feature_sets(S.uck, "leaf_features", 1); feats["leaf"] = [LF[i][0] for i in range(len(tree.leaves))]
names = {"leaf": tree.leaves, "class": tree.classes, "root": tree.roots}
labs = {"leaf": leaf, "class": cls, "root": root}
levels = [lv for lv in ("leaf", "class", "root") if lv in feats]
Z = {lv: [] for lv in levels}
with torch.no_grad():
    for s in range(0, len(H), 512):
        st = dict(zip(("leaf", "class", "root"), S.forward_stack(H[s:s + 512].cuda())))
        for lv in levels: Z[lv].append(st[lv][0][:, feats[lv]].float().cpu().numpy())
res = {"n_questions": len(rows)}
for lv in levels:
    z = np.concatenate(Z[lv]); n = len(names[lv])
    res[f"{lv}_names"] = np.array(names[lv]); res[f"{lv}_feat"] = np.array(feats[lv])
    res[f"{lv}_M"] = np.stack([z[labs[lv] == c].mean(0) for c in range(n)])
    res[f"{lv}_acc_argmax"] = np.array((z.argmax(1) == labs[lv]).mean())
    res[f"{lv}_acc"] = np.array(diag_acc(z, labs[lv], n))
    print(kind, lv, "binding acc %.3f" % res[f"{lv}_acc"], "distinct features", len(set(feats[lv])), "/", n)
np.savez_compressed(out, **res); print("saved", out)
