"""Step 4 (Baseline 2): frozen Hierarchical-SAE level 1 + unsupervised levels 2-3.

HYBRID baseline: supervised leaf level + UNSUPERVISED class/root levels.

Keeps SAE1 exactly as in the hierarchical model (frozen, copied from the trained
checkpoint) so the leaf-swap protocol is identical, and replaces levels 2 and 3 with
TRADITIONAL SAEs — reconstruction + L1 only, no alignment, no hierarchy loss, no
value head:

    h --[frozen supervised SAE1]--> h1 --[unsup SAE2]--> h2 --[unsup SAE3]

Concept slots for class/root are then chosen post-hoc by DIFFERENTIAL ACTIVATION
(01_find_unsupervised_features.py's protocol): for each class c,
    diff(f) = mean_act[f | samples of c] - max_{c' != c} mean_act[f | samples of c']
and the top-k features become that class's "slot". Same for root.

This isolates ONE variable: whether the ancestor levels are supervised.

Usage:
  python scripts/05_train_unsup_ancestor_sae.py \
      --hier_model models_opt2/main_spall_cos/layer9/hier_value_final.pt \
      --activation_file data/activations/x2m_train_activations_layer9.pkl \
      --output_dir models_opt2/unsup_anc/layer9
"""
import _bootstrap
import os
import json, argparse, copy
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from hsae.models.losses import nmse
from hsae.models.unsup_sae import TraditionalSAE
from hsae.models import hier_value_model as hvm




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hier_model", required=True, help="trained hierarchical ckpt (SAE1 source)")
    ap.add_argument("--activation_file", required=True)
    ap.add_argument("--class_activation_file", default="",
                    help="CEILING: class/root-question activations (prepare_class_qa_for_sae.py rows with "
                         "node_level/node_idx). They are pooled into SAE training, and class/root features are "
                         "selected from these explicit questions instead of tree-derived leaf groups.")
    ap.add_argument("--tree", default="data/00_taxonomy/tree.json")
    ap.add_argument("--property_labels", default="data/00_taxonomy/property_labels.json")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_features", type=int, default=10000,
                    help="hidden features per unsupervised level (AlignSAE baseline used 10k)")
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
    _, _, vocab_sizes, _, _, _, _ = hvm.load_property_tables(args.property_labels, dev)

    # ---- frozen supervised SAE1 ----
    ck = torch.load(args.hier_model, map_location="cpu")
    hier = hvm.HierValueSAE(ck["d_model"], ck["n_free"], ck["n1"], ck["n2"], ck["n3"],
                            vocab_sizes, value_head=ck.get("value_head", "bilinear2")).to(dev)
    hier.load_state_dict(ck["model_state_dict"]); hier.eval()
    for p in hier.parameters(): p.requires_grad = False

    import pickle
    raw = pickle.load(open(args.activation_file, "rb"))
    H = torch.stack([torch.as_tensor(r["h"]) for r in raw]).float()
    ci = torch.tensor([r["concept_idx"] for r in raw])
    n_leafrows = len(raw)
    craw = []
    if args.class_activation_file:
        craw = pickle.load(open(args.class_activation_file, "rb"))
        H = torch.cat([H, torch.stack([torch.as_tensor(r["h"]) for r in craw]).float()], 0)
        print(f"CEILING: pooled {len(craw)} class/root-question activations")
    with torch.no_grad():
        H1 = torch.cat([hier.sae1(H[i:i+512].to(dev))[1].cpu()
                        for i in range(0, len(H), 512)])
    print(f"SAE1 frozen; h1 computed: {tuple(H1.shape)}")

    d_model = H1.shape[1]
    sae2 = TraditionalSAE(d_model, args.n_features).to(dev)
    sae3 = TraditionalSAE(d_model, args.n_features).to(dev)
    opt = torch.optim.AdamW(list(sae2.parameters()) + list(sae3.parameters()), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=args.lr_min)
    dl = DataLoader(TensorDataset(H1), batch_size=args.batch_size, shuffle=True)
    print(f"training 2 unsupervised levels ({args.n_features} feats each), "
          f"recon + L1({args.lam_sparse}) only, {args.epochs} ep cosine")

    hist = []
    for e in range(args.epochs):
        agg = {}
        for (b,) in dl:
            b = b.to(dev)
            z2, h2 = sae2(b)
            z3, h3 = sae3(h2)
            L = (nmse(h2, b) + nmse(h3, h2)
                 + args.lam_sparse * (z2.abs().mean() + z3.abs().mean()))
            opt.zero_grad(); L.backward()
            torch.nn.utils.clip_grad_norm_(list(sae2.parameters()) + list(sae3.parameters()),
                                           args.clip)
            opt.step()
            for k, v in dict(total=L, rec2=nmse(h2, b), rec3=nmse(h3, h2),
                             sp2=z2.abs().mean(), sp3=z3.abs().mean()).items():
                agg[k] = agg.get(k, 0.0) + v.item()
        sched.step()
        for k in agg: agg[k] /= len(dl)
        agg["epoch"] = e + 1; hist.append(agg)
        if (e + 1) % 25 == 0 or e == 0:
            print(f"  e{e+1:3d}: tot={agg['total']:.4f} rec2={agg['rec2']:.4f} "
                  f"rec3={agg['rec3']:.4f} sp[{agg['sp2']:.2f}/{agg['sp3']:.2f}]", flush=True)

    # ---- differential activation: pick class / root slots ----
    sae2.eval(); sae3.eval()
    with torch.no_grad():
        Z2 = torch.cat([sae2(H1[i:i+512].to(dev))[0].cpu() for i in range(0, len(H1), 512)])
        H2 = torch.cat([sae2(H1[i:i+512].to(dev))[1].cpu() for i in range(0, len(H1), 512)])
        Z3 = torch.cat([sae3(H2[i:i+512].to(dev))[0].cpu() for i in range(0, len(H2), 512)])

    def pick(Z, labels, n_groups, name):
        means = torch.stack([Z[labels == g].mean(0) for g in range(n_groups)])  # [G, F]
        sel = {}
        for g in range(n_groups):
            other = torch.cat([means[:g], means[g+1:]]).max(0).values
            diff = means[g] - other
            top = torch.topk(diff, k=args.top_k)
            sel[g] = {"feature_indices": top.indices.tolist(),
                      "differential_scores": top.values.tolist()}
            print(f"  {name}[{g}]: feats {top.indices.tolist()} "
                  f"diff {[round(x,2) for x in top.values.tolist()]}")
        return sel

    cls_lab = torch.tensor([classes.index(cls_of[leaves[i]]) for i in ci.tolist()])
    root_lab = torch.tensor([roots.index(root_of[cls_of[leaves[i]]]) for i in ci.tolist()])
    print("\ndifferential activation — class slots (level 2):")
    if craw:
        isC = torch.tensor([r["node_level"] == "class" for r in craw]); isR = ~isC
        nid = torch.tensor([int(r["node_idx"]) for r in craw])
        class_sel = pick(Z2[n_leafrows:][isC], nid[isC], len(classes), "class")
        root_sel = pick(Z3[n_leafrows:][isR], nid[isR], len(roots), "root")
    else:
        class_sel = pick(Z2[:n_leafrows], cls_lab, len(classes), "class")
    print("differential activation — root slots (level 3):")
    if not craw: root_sel = pick(Z3[:n_leafrows], root_lab, len(roots), "root")

    torch.save({"sae2": sae2.state_dict(), "sae3": sae3.state_dict(),
                "d_model": d_model, "n_features": args.n_features,
                "hier_model": args.hier_model, "top_k": args.top_k,
                "class_features": class_sel, "root_features": root_sel,
                "args": vars(args)}, out / "unsup_anc_final.pt")
    json.dump(hist, open(out / "history.json", "w"), indent=2)
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
