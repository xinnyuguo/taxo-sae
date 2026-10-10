"""Step 4 (Baseline 1): three stacked unsupervised SAEs + differential-activation feature selection.

FULLY UNSUPERVISED baseline: three stacked traditional SAEs, no supervision at all.

    h --[unsup SAE1]--> h1 --[unsup SAE2]--> h2 --[unsup SAE3]

Each level is reconstruction + L1 only (no alignment CE, no hierarchy loss, no property
CE, no value head). Leaf / class / root identities are then mined POST-HOC by
differential activation at the level that would carry them in the supervised model:

    leaf  from z1 (level 1), class from z2 (level 2), root from z3 (level 3)
    diff(f) = mean_act[f | samples of g] - max_{g' != g} mean_act[f | samples of g']
    top-k features become that node's "slot"

Swapping also uses UNSUPERVISED directions: the steering vector is the mean of the
decoder columns of the target's top-k leaf features minus the source's — no supervised
information anywhere in the pipeline.

Usage:
  python scripts/05_train_unsup_full_sae.py \
      --activation_file data/activations/x2m_train_activations_layer9.pkl \
      --output_dir models_opt2/unsup_full/layer9
"""
import _bootstrap
import os
import json, argparse, pickle
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from hsae.models.losses import nmse
from hsae.models.unsup_sae import TraditionalSAE




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activation_file", required=True)
    ap.add_argument("--class_activation_file", default="",
                    help="CEILING: class/root-question activations (prepare_class_qa_for_sae.py rows with "
                         "node_level/node_idx). They are pooled into SAE training, and class/root features are "
                         "selected from these explicit questions instead of tree-derived leaf groups.")
    ap.add_argument("--tree", default="data/00_taxonomy/tree.json")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_features", type=int, default=10000)
    ap.add_argument("--lam_sparse", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_min", type=float, default=1e-6)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import random as _r, numpy as _np; _r.seed(args.seed); _np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    tree = json.load(open(args.tree))
    leaves, classes, roots = tree["layer1"], tree["layer2"], tree["layer3"]
    cls_of, root_of = tree["class_of_leaf"], tree["root_of_class"]

    raw = pickle.load(open(args.activation_file, "rb"))
    H = torch.stack([torch.as_tensor(r["h"]) for r in raw]).float()
    ci = torch.tensor([r["concept_idx"] for r in raw])
    n_leafrows = len(raw)
    craw = []
    if args.class_activation_file:
        craw = pickle.load(open(args.class_activation_file, "rb"))
        cH = torch.stack([torch.as_tensor(r["h"]) for r in craw]).float()
        H = torch.cat([H, cH], 0)
        print(f"CEILING: pooled {len(craw)} class/root-question activations into training "
              f"({sum(r['node_level']=='class' for r in craw)} class + {sum(r['node_level']=='root' for r in craw)} root)")
    d_model = H.shape[1]

    saes = nn.ModuleList([TraditionalSAE(d_model, args.n_features) for _ in range(3)]).to(dev)
    opt = torch.optim.AdamW(saes.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr_min)
    dl = DataLoader(TensorDataset(H), batch_size=args.batch_size, shuffle=True)
    print(f"3 stacked UNSUPERVISED SAEs, {args.n_features} feats each, "
          f"recon+L1({args.lam_sparse}), {args.epochs} ep cosine")

    hist = []
    for e in range(args.epochs):
        agg = {}
        for (b,) in dl:
            b = b.to(dev)
            z1, h1 = saes[0](b); z2, h2 = saes[1](h1); z3, h3 = saes[2](h2)
            L = (nmse(h1, b) + nmse(h2, h1) + nmse(h3, h2)
                 + args.lam_sparse * (z1.abs().mean() + z2.abs().mean() + z3.abs().mean()))
            opt.zero_grad(); L.backward()
            torch.nn.utils.clip_grad_norm_(saes.parameters(), args.clip); opt.step()
            for k, v in dict(total=L, rec1=nmse(h1, b), rec2=nmse(h2, h1),
                             rec3=nmse(h3, h2), sp1=z1.abs().mean()).items():
                agg[k] = agg.get(k, 0.0) + v.item()
        sched.step()
        for k in agg: agg[k] /= len(dl)
        agg["epoch"] = e + 1; hist.append(agg)
        if (e + 1) % 25 == 0 or e == 0:
            print(f"  e{e+1:3d}: tot={agg['total']:.4f} rec[{agg['rec1']:.4f}/"
                  f"{agg['rec2']:.4f}/{agg['rec3']:.4f}] sp1={agg['sp1']:.2f}", flush=True)

    # ---- differential activation at each level ----
    saes.eval()
    with torch.no_grad():
        Z1 = torch.cat([saes[0](H[i:i+512].to(dev))[0].cpu() for i in range(0, len(H), 512)])
        H1 = torch.cat([saes[0](H[i:i+512].to(dev))[1].cpu() for i in range(0, len(H), 512)])
        Z2 = torch.cat([saes[1](H1[i:i+512].to(dev))[0].cpu() for i in range(0, len(H1), 512)])
        H2 = torch.cat([saes[1](H1[i:i+512].to(dev))[1].cpu() for i in range(0, len(H1), 512)])
        Z3 = torch.cat([saes[2](H2[i:i+512].to(dev))[0].cpu() for i in range(0, len(H2), 512)])

    def pick(Z, labels, n_groups, name):
        means = torch.stack([Z[labels == g].mean(0) for g in range(n_groups)])
        sel = {}
        for g in range(n_groups):
            other = torch.cat([means[:g], means[g+1:]]).max(0).values
            top = torch.topk(means[g] - other, k=args.top_k)
            sel[g] = {"feature_indices": top.indices.tolist(),
                      "differential_scores": top.values.tolist()}
            print(f"  {name}[{g}]: {top.indices.tolist()} "
                  f"diff {[round(x,2) for x in top.values.tolist()]}")
        return sel

    cls_lab = torch.tensor([classes.index(cls_of[leaves[i]]) for i in ci.tolist()])
    root_lab = torch.tensor([roots.index(root_of[cls_of[leaves[i]]]) for i in ci.tolist()])
    print("\nleaf features (level 1):");  leaf_sel = pick(Z1[:n_leafrows], ci, len(leaves), "leaf")
    if craw:   # explicit class / root questions -> group by the asked node
        isC = torch.tensor([r["node_level"] == "class" for r in craw]); isR = ~isC
        nid = torch.tensor([int(r["node_idx"]) for r in craw])
        Zc2, Zr3 = Z2[n_leafrows:][isC], Z3[n_leafrows:][isR]
        print("class features (level 2, from CLASS questions):"); class_sel = pick(Zc2, nid[isC], len(classes), "class")
        print("root features (level 3, from ROOT questions):");   root_sel = pick(Zr3, nid[isR], len(roots), "root")
    else:
        print("class features (level 2):");   class_sel = pick(Z2[:n_leafrows], cls_lab, len(classes), "class")
        print("root features (level 3):");    root_sel = pick(Z3[:n_leafrows], root_lab, len(roots), "root")

    torch.save({"sae1": saes[0].state_dict(), "sae2": saes[1].state_dict(),
                "sae3": saes[2].state_dict(), "d_model": d_model,
                "n_features": args.n_features, "top_k": args.top_k,
                "leaf_features": leaf_sel, "class_features": class_sel,
                "root_features": root_sel, "args": vars(args)},
               out / "unsup_full_final.pt")
    json.dump(hist, open(out / "history.json", "w"), indent=2)
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
