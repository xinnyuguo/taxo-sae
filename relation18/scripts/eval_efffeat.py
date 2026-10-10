"""Concept fragmentation (EffFeat) and concentration (Top1C), as in AlignSAE §5.2 / Figure 5.

For concept c and feature k:  A[c,k] = mean_{i: c(i)=c} z_k(i),  B[c,k] = A[c,k] / (sum_k' A[c,k'] + eps)
    EffFeat(c) = exp(-sum_k B[c,k] log B[c,k])      (lower = less fragmented)
    Top1C(c)   = max_k B[c,k]                        (higher = more concentrated)
averaged over concepts.  Concepts: 18 leaves at level 1, 6 classes at level 2, 3 roots at level 3.

The Hierarchical SAE encoder is linear (no ReLU), so codes can be negative; we use the positive part
max(z, 0) per example as activation mass (a no-op for the ReLU baselines).  Hier is scored two ways:
  concept  - concept slots only (AlignSAE's convention for its own model)
  all      - every slot (free + concept + property), the same scope as the baselines' full dictionary.
Activations: the 14,400 leaf-question training activations of the layer.

Usage: python scripts/eval_efffeat.py [--tags L5 L6 L7 L9 L10] [--out results/testE/efffeat.json]
"""
import _bootstrap
import os as _os
MODELS = _os.environ.get("HSAE_MODELS", "models"); ACTS = _os.environ.get("HSAE_ACTS", "data/activations"); RES = _os.environ.get("HSAE_RESULTS", "results")
import argparse, json, pickle, re, os
import torch
from hsae.common import Tree
from hsae.systems import System

ap = argparse.ArgumentParser()
ap.add_argument("--tags", nargs="+", default=["L5", "L6", "L7", "L9", "L10"])
ap.add_argument("--tree", default="data/00_taxonomy/tree.json")
ap.add_argument("--property_labels", default="data/00_taxonomy/property_labels.json")
ap.add_argument("--out", default="results/testE/efffeat.json")
args = ap.parse_args()
DEV = "cuda"; EPS = 1e-12
LAYER = {"p20v2_goal": 9}
tree = Tree(args.tree)


def metrics(sums, counts):
    A = sums / counts[:, None]
    B = A / (A.sum(1, keepdim=True) + EPS)
    ent = -(B * torch.log(B + EPS)).sum(1)
    eff, top = torch.exp(ent), B.max(1).values
    return {"EffFeat": eff.mean().item(), "Top1C": top.mean().item(),
            "EffFeat_per_concept": [round(x, 3) for x in eff.tolist()],
            "Top1C_per_concept": [round(x, 4) for x in top.tolist()]}


out = {}
for tag in args.tags:
    layer = int(re.search(r"L(\d+)$", tag).group(1))
    rows = pickle.load(open(f"{ACTS}/{tag}_train_layer{layer}.pkl", "rb"))
    H = torch.tensor([r["h"] for r in rows], dtype=torch.float32)
    leaf = torch.tensor([r["concept_idx"] for r in rows])
    cls = torch.tensor([tree.class_idx_of_leaf(tree.leaves[i]) for i in leaf.tolist()])
    root = torch.tensor([tree.root_idx_of_leaf(tree.leaves[i]) for i in leaf.tolist()])
    labels = [leaf, cls, root]; ncon = [len(tree.leaves), len(tree.classes), len(tree.roots)]
    systems = {
        "hier": System("hier", tree, DEV, hier_ckpt=f"{MODELS}/hier_sae_{tag}/hier_value_final.pt",
                       property_labels=args.property_labels),
        "full": System("full", tree, DEV, unsup_ckpt=f"{MODELS}/baseline1_{tag}/unsup_full_final.pt"),
        "hybrid": System("hybrid", tree, DEV, hier_ckpt=f"{MODELS}/hier_sae_{tag}/hier_value_final.pt",
                         unsup_ckpt=f"{MODELS}/baseline2_{tag}/unsup_anc_final.pt",
                         property_labels=args.property_labels),
    }
    res = {}
    for name, S in systems.items():
        acc = None
        for s in range(0, len(H), 512):
            h = H[s:s + 512].to(DEV)
            zs = [z.clamp(min=0).float() for (z, _) in S.forward_stack(h)]
            if acc is None:
                acc = [torch.zeros(n, z.shape[1], device=DEV) for n, z in zip(ncon, zs)]
            for lv in range(3):
                acc[lv].index_add_(0, labels[lv][s:s + 512].to(DEV), zs[lv])
        counts = [torch.bincount(l, minlength=n).float().to(DEV) for l, n in zip(labels, ncon)]
        for lv, lvname in enumerate(("leaf", "class", "root")):
            supervised = name == "hier" or (name == "hybrid" and lv == 0)
            if supervised:
                nf = S.n_free
                res[f"{name}_all_{lvname}"] = metrics(acc[lv], counts[lv])
                res[f"{name}_concept_{lvname}"] = metrics(acc[lv][:, nf:nf + ncon[lv]], counts[lv])
            else:
                res[f"{name}_all_{lvname}"] = metrics(acc[lv], counts[lv])
        del S; torch.cuda.empty_cache()
    out[tag] = {"layer": layer, "n": len(rows), **res}
    print(f"== {tag} (layer {layer}, n={len(rows)})")
    for k, v in res.items():
        print(f"  {k:24s} EffFeat {v['EffFeat']:10.2f}   Top1C {v['Top1C']:.4f}")
os.makedirs(os.path.dirname(args.out), exist_ok=True); json.dump(out, open(args.out, "w"), indent=1); print("saved", args.out)
