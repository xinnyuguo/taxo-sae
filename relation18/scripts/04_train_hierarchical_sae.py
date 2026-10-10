"""Step 4: Train the Hierarchical SAE (three stacked levels, derived class/root targets, value head).

Hierarchical SAE + per-slot multi-label value head — END-TO-END JOINT training.

Builds on the stabilized joint recipe (warm concept/hier via CE to DETACHED
parent/root labels, cross-level recon grads, grad clip, hier anneal, best-ckpt) and
ADDS a property block + per-slot value head at every level (see hier_value_model.py).

Value supervision (the new part):
  Every sample carries a leaf concept_idx -> maps to a class idx and a root idx.
  For each property p, the node's target is the MULTI-HOT union of its descendant
  leaves' values (leaf: single value). Value head logits (multi-label) are trained
  with BCE against that multi-hot. This makes "top-K contains all values" the goal.

Correctness metric (per level, per property): take top-K_layer logits; success iff
they contain ALL true values in the multi-hot. Reported as mean over properties.
"""
import _bootstrap
import os
import json, argparse, copy
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from hsae.common import ActDataset, build_masks
from hsae.models.losses import nmse, hier_target, align_loss, value_loss_and_acc
from hsae.models.hier_value_model import HierValueSAE, load_property_tables





def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activation_file", default="data/activations/goal_train_layer9.pkl")
    ap.add_argument("--class_activation_file", default="",
                    help="SAE-side ceiling (Experi #3 group 3): extra activations from "
                         "CLASS/ROOT-level questions (prepare_class_qa_for_sae.py + 04). "
                         "For such a row the leaf level is supervised MULTI-HOT over the "
                         "node's descendant leaves (BCE), the class/root levels with the "
                         "node itself (CE), property as usual.")
    ap.add_argument("--lam_class_qa", type=float, default=1.0,
                    help="weight of the class-QA rows' losses")
    ap.add_argument("--tree", default="data/00_taxonomy/tree.json")
    ap.add_argument("--property_labels", default="data/00_taxonomy/property_labels.json")
    ap.add_argument("--output_dir", default="models/hier_sae_goal")
    ap.add_argument("--value_head", default="bilinear",
                    choices=["bilinear", "bilinear2", "per_slot"])
    ap.add_argument("--n_free", type=int, default=100000)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--lam_sparse", type=float, default=1e-3)
    ap.add_argument("--lam_sparse_c", type=float, default=-1.0,
                    help="L1 on CONCEPT slots at every level (advisor 2026-08-18: sparsity "
                         "must cover the whole vector — inactive concept slots were "
                         "unconstrained and left noise in the swap signal). "
                         "-1 = use lam_sparse; 0 = off (legacy behaviour).")
    ap.add_argument("--lam_sparse_p", type=float, default=-1.0,
                    help="L1 on PROPERTY slots at every level. -1 = use lam_sparse; "
                         "0 = off (legacy behaviour).")
    ap.add_argument("--class_target", default="derived", choices=["derived", "gt"],
                    help="class/root CE target: 'derived' from detached child softmax "
                         "(legacy) or 'gt' straight from the tree mapping of the sample's "
                         "leaf label (advisor bug-check).")
    ap.add_argument("--level_order", default="normal", choices=["normal", "flipped"],
                    help="flipped = stack top-down (sae1=root, sae2=class, sae3=leaf; "
                         "Ming's experiment). Forces gt targets at every level.")
    ap.add_argument("--align_loss", default="ce", choices=["ce", "mse_onehot"],
                    help="CE is scale-invariant, so concept-slot activations drift "
                         "freely (measured ~13.5 here) and a slot edit of size alpha has "
                         "no canonical meaning. AlignSAE instead uses MSE to a one-hot "
                         "target, which ANCHORS active slots at 1.0 / inactive at 0.0 — "
                         "that is why its swap protocol works with alpha in [0,1].")
    ap.add_argument("--lam_align", type=float, default=1.0)
    ap.add_argument("--lam_palign", type=float, default=0.0,
                    help="property-slot alignment CE (dual-softmax, mirrors 05_train_sae.py "
                         "L_align_property): asked property's slot must win among the P slots, "
                         "applied at every level. 0 = off (legacy behaviour). Needed to make "
                         "property decoder columns causally steerable.")
    ap.add_argument("--lam_ortho_cp", type=float, default=0.0,
                    help="concept-block vs property-block activation cross-covariance penalty "
                         "(mirrors 05_train_sae.py L_ortho_cp, squared-cov sum). 0 = off. "
                         "Old flat trainer default was 0.5.")
    ap.add_argument("--lam_indep", type=float, default=0.0,
                    help="AlignSAE's L_indep: decorrelate FREE slots among themselves "
                         "(off-diagonal of their covariance). Stops the free block from "
                         "ganging up to encode a concept. AlignSAE uses 1e-2.")
    ap.add_argument("--lam_ortho_cf", type=float, default=0.0,
                    help="AlignSAE's L_ortho: concept slots must be UNCORRELATED with the "
                         "free slots, i.e. concept information may not leak into the free "
                         "block. This (not n_free or the loss type) is what gives AlignSAE's "
                         "relation slots their causal weight. AlignSAE uses 1e-2. "
                         "NOTE this is different from --lam_ortho_cp (concept vs property).")
    ap.add_argument("--ortho_cf_form", default="cov", choices=["cov", "corr"],
                    help="L_ortho_cf as the raw cross-covariance sum (AlignSAE appendix) or the cross-correlation "
                         "||corr(z_concept, z_free)||_F^2 (AlignSAE main text); corr is scale-invariant and cannot blow up")
    ap.add_argument("--lam_hier", type=float, default=5.0)
    ap.add_argument("--lam_value", type=float, default=1.0)
    ap.add_argument("--init_from", default="",
                    help="warm-start from an existing hier_value_final.pt (weights only; "
                         "the optimizer restarts, so expect a short re-warm-up). Use to "
                         "CONTINUE training a run whose reconstruction was still falling.")
    ap.add_argument("--lr_schedule", default="const", choices=["const", "cosine"],
                    help="cosine decay to --lr_min over the run; the constant-lr runs "
                         "oscillate by ~20x in reconstruction late in training, which makes "
                         "the saved checkpoint depend on where the noise happened to land.")
    ap.add_argument("--lr_min", type=float, default=1e-5)
    ap.add_argument("--anneal_hier", type=int, default=40)
    ap.add_argument("--anneal_value", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import random as _r, numpy as _np; _r.seed(args.seed); _np.random.seed(args.seed); torch.manual_seed(args.seed)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    tree = json.load(open(args.tree))
    n1, n2, n3 = len(tree["layer1"]), len(tree["layer2"]), len(tree["layer3"])
    m2, m3 = build_masks(tree, dev)

    # property tables
    props, vocab, vocab_sizes, vmask, vidx, K, layer_tensor = load_property_tables(
        args.property_labels, dev)
    T1 = layer_tensor("layer1", tree["layer1"])   # [12, P, Vmax]
    T2 = layer_tensor("layer2", tree["layer2"])   # [4,  P, Vmax]
    T3 = layer_tensor("layer3", tree["layer3"])   # [2,  P, Vmax]
    K1, K2, K3 = K["layer1"], K["layer2"], K["layer3"]
    print(f"K per layer: leaf={K1} class={K2} root={K3} | P={len(props)} props")

    # per-sample node maps: leaf_idx -> class_idx -> root_idx
    class_names = tree["layer2"]; root_names = tree["layer3"]
    cls_of = {i: class_names.index(tree["class_of_leaf"][a])
              for i, a in enumerate(tree["layer1"])}
    root_of = {ci: root_names.index(tree["root_of_class"][c])
               for ci, c in enumerate(class_names)}
    leaf2cls = torch.tensor([cls_of[i] for i in range(n1)], device=dev)
    cls2root = torch.tensor([root_of[c] for c in range(n2)], device=dev)

    ds = ActDataset(args.activation_file)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    d_model = ds[0]["h"].shape[0]

    # ---- SAE-side ceiling: class/root-level question activations (separate loader,
    # one batch interleaved per leaf batch so both objectives see every epoch) ----
    cls_dl = None
    if args.class_activation_file:
        import pickle as _pk
        craw = _pk.load(open(args.class_activation_file, "rb"))
        cH = torch.stack([torch.as_tensor(r["h"]) for r in craw]).float()
        cP = torch.tensor([int(r["property_idx"]) for r in craw])
        cLvl = torch.tensor([0 if r["node_level"] == "class" else 1 for r in craw])
        cIdx = torch.tensor([int(r["node_idx"]) for r in craw])
        cMulti = torch.zeros(len(craw), n1)
        for i, r in enumerate(craw):
            for m in r["leaf_members"]:
                cMulti[i, m] = 1.0
        cRoot = torch.tensor([int(r["root_idx"]) for r in craw])
        cds = torch.utils.data.TensorDataset(cH, cP, cLvl, cIdx, cMulti, cRoot)
        cls_dl = DataLoader(cds, batch_size=args.batch_size, shuffle=True)
        print(f"class-QA rows: {len(craw)} "
              f"({int((cLvl==0).sum())} class + {int((cLvl==1).sum())} root), "
              f"lam_class_qa={args.lam_class_qa}")

    flipped = args.level_order == "flipped"
    if flipped and args.class_target == "derived":
        print("level_order=flipped forces class_target=gt (no top-down derivation)")
        args.class_target = "gt"
    # constructed level sizes / value tables / Ks, in stacking order
    if flipped:
        s1, s2, s3 = n3, n2, n1
        Ts, Ks = (T3, T2, T1), (K3, K2, K1)
    else:
        s1, s2, s3 = n1, n2, n3
        Ts, Ks = (T1, T2, T3), (K1, K2, K3)
    model = HierValueSAE(d_model, args.n_free, s1, s2, s3, vocab_sizes,
                         value_head=args.value_head).to(dev)

    if args.init_from:
        _ck = torch.load(args.init_from, map_location="cpu")
        model.load_state_dict(_ck["model_state_dict"])
        print(f"warm-started from {args.init_from}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(
                 opt, T_max=args.epochs, eta_min=args.lr_min)
             if args.lr_schedule == "cosine" else None)
    lam_sc = args.lam_sparse if args.lam_sparse_c < 0 else args.lam_sparse_c
    lam_spp = args.lam_sparse if args.lam_sparse_p < 0 else args.lam_sparse_p
    hist, best = [], ((-1.0, float("-inf")), None, None)
    print(f"joint end-to-end + value | lr={args.lr} clip={args.clip} "
          f"lam_hier={args.lam_hier} lam_value={args.lam_value} "
          f"sparse[free={args.lam_sparse} concept={lam_sc} prop={lam_spp}] "
          f"class_target={args.class_target} order={args.level_order} "
          f"align={args.align_loss} n_free={args.n_free}")

    for e in range(args.epochs):
        model.train()
        warm = min(1.0, (e + 1) / max(1, args.anneal_hier))
        lam_h = args.lam_hier * warm
        lam_v = args.lam_value * min(1.0, (e + 1) / max(1, args.anneal_value))
        agg, diverged = {}, False
        cls_iter = iter(cls_dl) if cls_dl is not None else None
        for b in dl:
            h = b["h"].to(dev); ci = b["concept_idx"].to(dev)
            pi = b["property_idx"].to(dev)
            (z1, h1), (z2, h2), (z3, h3) = model(h)
            cls_i = leaf2cls[ci]; root_i = cls2root[cls_i]

            # reconstruction (cross-level grads on)
            L_rec = nmse(h1, h) + nmse(h2, h1) + nmse(h3, h2)
            nf = model.sae1.n_free
            L_sp = (z1[:, :nf].abs().mean() + z2[:, :nf].abs().mean()
                    + z3[:, :nf].abs().mean())
            # advisor fix (2026-08-18): L1 on the FIXED slots too — inactive concept /
            # property slots were previously unconstrained ("they could be anything"),
            # leaving noise in the very directions the swap steers along
            L_sp_c = sum(zz[:, nf:nf + sl.n_concept].abs().mean() for zz, sl in
                         ((z1, model.sae1), (z2, model.sae2), (z3, model.sae3)))
            L_sp_p = sum(zz[:, nf + sl.n_concept:].abs().mean() for zz, sl in
                         ((z1, model.sae1), (z2, model.sae2), (z3, model.sae3)))

            # concept / hierarchy CE targets per stacking order / target mode
            if flipped:
                t1, t2, t3 = root_i, cls_i, ci
            elif args.class_target == "gt":
                t1, t2, t3 = ci, cls_i, root_i
            else:
                t1, t2, t3 = ci, None, None      # derived from child softmax below
            lg1 = model.sae1.concept_logits(z1)
            L_a1 = align_loss(lg1, t1, args.align_loss)
            acc1 = (lg1.argmax(-1) == t1).float().mean()
            if t2 is None:
                with torch.no_grad():
                    p1 = F.softmax(lg1.detach(), -1); t2 = hier_target(p1, m2).argmax(-1)
            lg2 = model.sae2.concept_logits(z2)
            L_a2 = align_loss(lg2, t2, args.align_loss)
            acc2 = (lg2.argmax(-1) == t2).float().mean()
            if t3 is None:
                with torch.no_grad():
                    p2 = F.softmax(lg2.detach(), -1); t3 = hier_target(p2, m3).argmax(-1)
            lg3 = model.sae3.concept_logits(z3)
            L_a3 = align_loss(lg3, t3, args.align_loss)
            acc3 = (lg3.argmax(-1) == t3).float().mean()
            # advisor bug-check telemetry: class/root accuracy against TREE-GT ancestors,
            # independent of what the training target was
            if flipped:
                acc2_gt = acc2
                acc3_gt = (lg3.argmax(-1) == ci).float().mean()
            else:
                acc2_gt = (lg2.argmax(-1) == cls_i).float().mean()
                acc3_gt = (lg3.argmax(-1) == root_i).float().mean()

            # property alignment (dual-softmax, every level): the asked property's slot
            # must win among the P property slots — this is what makes the property
            # decoder columns causal steering directions (value-head BCE alone leaves
            # them readable but not writable).
            L_p = torch.tensor(0.0, device=dev); pacc = torch.tensor(0.0, device=dev)
            if args.lam_palign > 0:
                pls = [model.sae1.property_slots(z1), model.sae2.property_slots(z2),
                       model.sae3.property_slots(z3)]
                L_p = sum(align_loss(pl_, pi, args.align_loss) for pl_ in pls) / 3
                pacc = (pls[0].argmax(-1) == pi).float().mean()

            # concept-block vs property-block orthogonality. CORRELATION form (the old
            # trainer's ortho_regularize=True path): scale-invariant, so it doesn't
            # explode while early-epoch activations are still huge (the raw squared-cov
            # SUM hit ~1e15 at e1 and wrecked reconstruction).
            L_ocp = torch.tensor(0.0, device=dev)
            if args.lam_ortho_cp > 0:
                for zz, sae_l in ((z1, model.sae1), (z2, model.sae2), (z3, model.sae3)):
                    zc = sae_l.concept_logits(zz); zp = sae_l.property_slots(zz)
                    zcc = zc - zc.mean(0, keepdim=True); zpc = zp - zp.mean(0, keepdim=True)
                    cov = (zcc.T @ zpc) / zz.shape[0]
                    corr = cov / (zc.std(0).unsqueeze(1) * zp.std(0).unsqueeze(0) + 1e-8)
                    L_ocp = L_ocp + corr.pow(2).mean()

            # AlignSAE-style free-slot regularisers (both default OFF).
            # L_indep: off-diagonal of the free-slot covariance. Computing the full
            #   [n_free, n_free] covariance is impossible at 10k-100k slots, so use the
            #   identity  sum_offdiag(C^2) = ||C||_F^2 - ||diag(C)||^2  with
            #   ||C||_F^2 = ||Zc^T Zc||_F^2 / B^2 = ||Zc Zc^T||_F^2 / B^2  (Gram trick,
            #   [B,B] instead of [n_free,n_free]).
            L_indep = torch.tensor(0.0, device=dev)
            if args.lam_indep > 0:
                for zz in (z1, z2, z3):
                    zf = zz[:, :nf]
                    zfc = zf - zf.mean(0, keepdim=True)
                    B = zfc.shape[0]
                    gram = zfc @ zfc.T                      # [B, B]
                    fro_sq = (gram ** 2).sum() / (B * B)    # == ||cov||_F^2
                    diag_sq = ((zfc ** 2).sum(0) / B).pow(2).sum()
                    L_indep = L_indep + (fro_sq - diag_sq).clamp(min=0)
            # L_ortho_cf: concept block must be uncorrelated with the free block.
            L_ocf = torch.tensor(0.0, device=dev)
            if args.lam_ortho_cf > 0:
                for zz, sl in ((z1, model.sae1), (z2, model.sae2), (z3, model.sae3)):
                    zc = sl.concept_logits(zz); zf = zz[:, :nf]
                    zcc = zc - zc.mean(0, keepdim=True)
                    zfc = zf - zf.mean(0, keepdim=True)
                    cross = (zcc.T @ zfc) / zz.shape[0]     # [n_concept, n_free]
                    if args.ortho_cf_form == "corr":         # ||corr(z_concept, z_free)||_F^2, scale-invariant
                        cross = cross / (zc.std(0).unsqueeze(1) * zf.std(0).unsqueeze(0) + 1e-8)
                    L_ocf = L_ocf + (cross ** 2).sum()

            # value: multi-label BCE to the sample's node union-target at each level
            if flipped:
                i1, i2, i3 = root_i, cls_i, ci
            else:
                i1, i2, i3 = ci, cls_i, root_i
            tv1 = Ts[0][i1]; tv2 = Ts[1][i2]; tv3 = Ts[2][i3]   # [B, P, Vmax]
            Lv1, av1 = value_loss_and_acc(model.sae1.value_logits(z1), tv1, vmask, Ks[0])
            Lv2, av2 = value_loss_and_acc(model.sae2.value_logits(z2), tv2, vmask, Ks[1])
            Lv3, av3 = value_loss_and_acc(model.sae3.value_logits(z3), tv3, vmask, Ks[2])

            # ---- SAE-side ceiling: one class-QA batch alongside this leaf batch ----
            L_cq = torch.tensor(0.0, device=dev)
            cq_leaf = cq_node = torch.tensor(0.0, device=dev)
            if cls_iter is not None:
                try:
                    cb = next(cls_iter)
                except StopIteration:
                    cls_iter = iter(cls_dl); cb = next(cls_iter)
                ch, cp, clvl, cidx, cmulti, croot = [t.to(dev) for t in cb]
                (cz1, ch1), (cz2, ch2), (cz3, ch3) = model(ch)
                # reconstruction + sparsity on these inputs too
                L_cq = L_cq + nmse(ch1, ch) + nmse(ch2, ch1) + nmse(ch3, ch2)
                L_cq = L_cq + args.lam_sparse * (cz1[:, :nf].abs().mean()
                                                 + cz2[:, :nf].abs().mean()
                                                 + cz3[:, :nf].abs().mean())
                clg1 = model.sae1.concept_logits(cz1)
                clg2 = model.sae2.concept_logits(cz2)
                clg3 = model.sae3.concept_logits(cz3)
                # leaf level: MULTI-HOT over the node's descendant leaves
                L_cq = L_cq + args.lam_align * F.binary_cross_entropy_with_logits(
                    clg1, cmulti)
                # class level: the class itself for class rows; root rows carry no
                # single class target, so they only supervise the root level
                is_cls = clvl == 0
                if is_cls.any():
                    L_cq = L_cq + lam_h * F.cross_entropy(clg2[is_cls], cidx[is_cls])
                L_cq = L_cq + lam_h * F.cross_entropy(clg3, croot)
                # property slot: the asked property (same dual-softmax CE)
                if args.lam_palign > 0:
                    L_cq = L_cq + warm * args.lam_palign * F.cross_entropy(
                        model.sae1.property_slots(cz1), cp)
                # value heads: the node's own union target at its own level
                cls_rows = torch.where(is_cls)[0]; root_rows = torch.where(~is_cls)[0]
                if len(cls_rows):
                    tv = T2[cidx[cls_rows]]
                    lv, _ = value_loss_and_acc(
                        [v[cls_rows] for v in model.sae2.value_logits(cz2)],
                        tv, vmask, K2)
                    L_cq = L_cq + lam_v * lv
                if len(root_rows):
                    tv = T3[cidx[root_rows]]
                    lv, _ = value_loss_and_acc(
                        [v[root_rows] for v in model.sae3.value_logits(cz3)],
                        tv, vmask, K3)
                    L_cq = L_cq + lam_v * lv
                with torch.no_grad():
                    # telemetry: leaf multi-hot top-|members| hit rate, node argmax acc
                    k = int(cmulti.sum(1).max().item())
                    topk = clg1.topk(k, -1).indices
                    pred = torch.zeros_like(cmulti).scatter_(1, topk, 1.0)
                    cq_leaf = ((cmulti * (1 - pred)).sum(-1) == 0).float().mean()
                    cq_node = torch.cat([
                        (clg2[is_cls].argmax(-1) == cidx[is_cls]).float()
                        if is_cls.any() else torch.zeros(0, device=dev),
                        (clg3[~is_cls].argmax(-1) == cidx[~is_cls]).float()
                        if (~is_cls).any() else torch.zeros(0, device=dev)]).mean()

            total = (L_rec + args.lam_sparse * L_sp
                     + lam_sc * L_sp_c + lam_spp * L_sp_p
                     + args.lam_align * L_a1 + lam_h * (L_a2 + L_a3)
                     + warm * (args.lam_palign * L_p + args.lam_ortho_cp * L_ocp
                               + args.lam_indep * L_indep + args.lam_ortho_cf * L_ocf)
                     + lam_v * (Lv1 + Lv2 + Lv3)
                     + args.lam_class_qa * L_cq)
            if not torch.isfinite(total):
                print(f"  [e{e+1}] NON-FINITE -> stop"); diverged = True; break
            opt.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            for k, v in dict(total=total, rec=L_rec, a1=L_a1, a2=L_a2, a3=L_a3,
                             v1=Lv1, v2=Lv2, v3=Lv3, acc1=acc1, acc2=acc2, acc3=acc3,
                             acc2gt=acc2_gt, acc3gt=acc3_gt,
                             spf=L_sp, spc=L_sp_c, spp=L_sp_p,
                             cq=L_cq, cqleaf=cq_leaf, cqnode=cq_node,
                             vacc1=av1, vacc2=av2, vacc3=av3,
                             lp=L_p, ocp=L_ocp, indep=L_indep, ocf=L_ocf,
                             pacc=pacc).items():
                agg[k] = agg.get(k, 0.0) + v.item()
        if diverged: break
        if sched is not None: sched.step()
        for k in agg: agg[k] /= len(dl)
        agg.update(epoch=e + 1, lam_hier=lam_h, lam_value=lam_v,
                   lr=opt.param_groups[0]["lr"]); hist.append(agg)
        # best on mean(concept accs + value accs); with the class-QA objective the accs
        # saturate by ~e36, so tie-break by RECON — otherwise an unconverged checkpoint
        # (garbage decoder columns, dead swaps) wins on a rounding tie
        sc = (agg["acc1"] + agg["acc2"] + agg["acc3"]
              + agg["vacc1"] + agg["vacc2"] + agg["vacc3"]) / 6
        if args.lam_palign > 0:
            # property gating must also be converged for the checkpoint to be useful
            sc = (sc * 6 + agg["pacc"]) / 7
        if cls_dl is not None:
            sc = (sc * 7 + agg["cqleaf"] + agg["cqnode"]) / 9
        score = (round(sc, 4), -agg["rec"])
        if score > best[0]:
            best = (score, copy.deepcopy(model.state_dict()), dict(agg))
        if (e + 1) % 10 == 0 or e == 0:
            print(f"  e{e+1:3d}: tot={agg['total']:.2f} rec={agg['rec']:.3f} "
                  f"C[{agg['acc1']:.2f}/{agg['acc2']:.2f}/{agg['acc3']:.2f}] "
                  f"GT[{agg['acc2gt']:.2f}/{agg['acc3gt']:.2f}] "
                  f"Vacc[{agg['vacc1']:.2f}/{agg['vacc2']:.2f}/{agg['vacc3']:.2f}] "
                  f"Pacc={agg['pacc']:.2f} ocp={agg['ocp']:.3f} "
                  + (f"indep={agg['indep']:.3f} " if args.lam_indep > 0 else "")
                  + (f"ocf={agg['ocf']:.3f} " if args.lam_ortho_cf > 0 else "")
                  + f"sp[{agg['spf']:.2f}/{agg['spc']:.2f}/{agg['spp']:.2f}] "
                  + (f"CQ[{agg['cqleaf']:.2f}/{agg['cqnode']:.2f}] "
                     if cls_dl is not None else "")
                  + f"lh={lam_h:.1f} lv={lam_v:.2f}")

    if best[1] is not None:
        model.load_state_dict(best[1])
        b = best[2]
        print(f"restored BEST score={best[0][0]:.3f} rec={-best[0][1]:.3f} @e{b['epoch']} "
              f"C[{b['acc1']:.2f}/{b['acc2']:.2f}/{b['acc3']:.2f}] "
              f"Vacc[{b['vacc1']:.2f}/{b['vacc2']:.2f}/{b['vacc3']:.2f}]")
    torch.save({"model_state_dict": model.state_dict(), "d_model": d_model,
                "n_free": args.n_free, "n1": s1, "n2": s2, "n3": s3,
                "vocab_sizes": vocab_sizes, "K": K, "props": props,
                "value_head": args.value_head, "level_order": args.level_order,
                "args": vars(args)}, out / "hier_value_final.pt")
    json.dump(hist, open(out / "history.json", "w"), indent=2)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
