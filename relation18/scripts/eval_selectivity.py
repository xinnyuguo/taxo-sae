"""Selectivity of concept ablation: does removing concept c's slot / features hurt only c's questions?

For each concept c (18 leaves at SAE1, 6 classes at SAE2) and each system:
  at every token position (default; --positions last = last prompt token only), subtract the ablated features' decoder contribution
      leaf : h' = h - sum_{k in F(c)} z1_k W1[:, k]                 (z1 = SAE1(h))
      class: h' = h - sum_{k in F(c)} z2_k W2[:, k]                 (z2 = SAE2(SAE1(h)) recon)
  so only the ablated part is removed (no reconstruction error is introduced).
  Score the gold answer by teacher forcing: mean per-token log-prob and exact (all answer tokens argmax).
  self   = questions whose leaf is c (leaf) / belongs to class c (class)
  others = all remaining questions
  The removed part is scaled by s (s=1 is plain ablation, which barely moves this saturated LM);
  we sweep s and report, per system, the smallest s that breaks c's own questions
  (s50: self exact <= 0.5, s0: self exact <= 0.05) and the accuracy that the other relations keep there.
F(c): Hier = its one concept slot; baselines = top-k differential features (k = 1..5).
Baseline2's SAE1 is Hier's SAE1, so it is evaluated at the class level only.

Usage: python scripts/eval_selectivity.py --tags L6 --out results/testF/selectivity_L6.json
"""
import _bootstrap
import os as _os
MODELS = _os.environ.get("HSAE_MODELS", "models"); ACTS = _os.environ.get("HSAE_ACTS", "data/activations"); RES = _os.environ.get("HSAE_RESULTS", "results")
import argparse, json, pickle, random, collections, re, os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from hsae.common import Tree
from hsae.systems import System
from hsae.steering import add_hook

ap = argparse.ArgumentParser()
ap.add_argument("--tags", nargs="+", default=["L5", "L6", "L7", "L9", "L10"])
ap.add_argument("--lm", default="models/base_lm/final")
ap.add_argument("--tree", default="data/00_taxonomy/tree.json")
ap.add_argument("--property_labels", default="data/00_taxonomy/property_labels.json")
ap.add_argument("--per_leaf", type=int, default=20)
ap.add_argument("--ks", default="1")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--scales", default="1,2,5,10,20,40,80,160")
ap.add_argument("--positions", choices=["all", "last"], default="last",
                help="ablate at every token position (default) or only the last prompt token")
ap.add_argument("--out", default="results/testF/selectivity.json")
args = ap.parse_args()
SCALES = None
DEV = "cuda"; LAYER = {"p20v2_goal": 9}; KS = [int(x) for x in args.ks.split(",")]
SCALES = [float(x) if "." in x else int(x) for x in args.scales.split(",")]
tree = Tree(args.tree); n_leaf, n_cls = len(tree.leaves), len(tree.classes)
tok = AutoTokenizer.from_pretrained(args.lm)
lm = AutoModelForCausalLM.from_pretrained(args.lm).to(DEV).eval()


def build_batch(rows):
    seqs, plen, alen = [], [], []
    for r in rows:
        p = tok(f"Q: {r['question']}\nA:")["input_ids"]; a = list(r["answer_tokens"])
        seqs.append(p + a); plen.append(len(p)); alen.append(len(a))
    L = max(map(len, seqs))
    ids = torch.full((len(seqs), L), tok.eos_token_id); att = torch.zeros(len(seqs), L, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1
    return ids.to(DEV), att.to(DEV), torch.tensor(plen), torch.tensor(alen)


@torch.no_grad()
def score(batch, layer, delta_fn=None):
    """Per-question (mean answer-token logp, exact) with an optional edit at the last prompt token."""
    ids, att, plen, alen = batch
    pos = (plen - 1).to(DEV); bidx = torch.arange(len(ids), device=DEV)

    def hook(mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        if args.positions == "last":
            h = hs[bidx, pos, :]
            hs[bidx, pos, :] = h + delta_fn(h.float()).to(hs.dtype)
        else:
            B, T, d = hs.shape
            hs += delta_fn(hs.reshape(B * T, d).float()).reshape(B, T, d).to(hs.dtype)
        return out
    out_lp, out_ex = [], []
    CH = int(os.environ.get("SEL_CHUNK", "120"))          # micro-batch (same padded length) to bound memory
    for c0 in range(0, len(ids), CH):
        sl = slice(c0, c0 + CH); bidx = torch.arange(len(ids[sl]), device=DEV); pos = (plen[sl] - 1).to(DEV)
        hd = add_hook(lm, layer, hook) if delta_fn is not None else None
        try:
            logits = lm(input_ids=ids[sl], attention_mask=att[sl]).logits
        finally:
            if hd is not None: hd.remove()
        for j in range(len(ids[sl])):
            i = c0 + j; p, a = plen[i].item(), alen[i].item()
            tgt = ids[i, p:p + a]; pred = torch.log_softmax(logits[j, p - 1:p + a - 1].float(), -1)   # answer positions only
            out_lp.append(pred.gather(1, tgt[:, None]).mean().item())
            out_ex.append(float((pred.argmax(1) == tgt).all().item()))
    return torch.tensor(out_lp), torch.tensor(out_ex)


def make_delta(S, level, feats, scale=1.0):
    """-scale * sum_k max(z_k, 0) W[:, k]: removes the concept's positive presence (Hier codes are
    linear and negative on other concepts; zeroing those would inject the concept instead)."""
    feats = torch.tensor(feats, device=DEV)
    if level == "leaf":
        W = S.L1.decoder.weight[:, feats].float()                       # [d, k]
        return lambda h: -scale * (S.L1(h)[0][:, feats].float().clamp(min=0) @ W.T)
    W = S.L2.decoder.weight[:, feats].float()
    return lambda h: -scale * (S.L2(S.L1(h)[1])[0][:, feats].float().clamp(min=0) @ W.T)


out = {}
for tag in args.tags:
    layer = int(re.search(r"L(\d+)$", tag).group(1))
    rows = pickle.load(open(f"{ACTS}/{tag}_ooda_layer{layer}.pkl", "rb"))
    by = collections.defaultdict(list)
    for r in rows: by[r["concept_idx"]].append(r)
    rng = random.Random(args.seed)
    qs = [r for c in range(n_leaf) for r in rng.sample(by[c], min(args.per_leaf, len(by[c])))]
    leaf_of = torch.tensor([r["concept_idx"] for r in qs])
    cls_of = torch.tensor([tree.class_idx_of_leaf(tree.leaves[i]) for i in leaf_of.tolist()])
    batch = build_batch(qs)
    clean_lp, clean_ex = score(batch, layer)
    H = f"{MODELS}/hier_sae_{tag}/hier_value_final.pt"
    configs = [("hier", None, System("hier", tree, DEV, hier_ckpt=H, property_labels=args.property_labels))]
    for k in KS:
        configs.append(("full", k, System("full", tree, DEV, unsup_ckpt=f"{MODELS}/baseline1_{tag}/unsup_full_final.pt", k=k)))
        configs.append(("hybrid", k, System("hybrid", tree, DEV, hier_ckpt=H, unsup_ckpt=f"{MODELS}/baseline2_{tag}/unsup_anc_final.pt",
                                            property_labels=args.property_labels, k=k)))
    res = {"layer": layer, "n_questions": len(qs), "clean_exact": clean_ex.mean().item(), "clean_logp": clean_lp.mean().item()}
    for name, k, S in configs:
        for level, n_con, lab, feats_of in (("leaf", n_leaf, leaf_of, S.leaf_feats), ("class", n_cls, cls_of, S.class_feats)):
            if name == "hybrid" and level == "leaf":
                continue
            curve = {}
            for sc in SCALES:
                sd, od, se, oe = [], [], [], []
                for c in range(n_con):
                    lp, ex = score(batch, layer, make_delta(S, level, feats_of(c), sc))
                    m = lab == c
                    sd.append((clean_lp[m] - lp[m]).mean().item()); od.append((clean_lp[~m] - lp[~m]).mean().item())
                    se.append(ex[m].mean().item()); oe.append(ex[~m].mean().item())
                mean = lambda v: sum(v) / len(v)
                curve[sc] = {"self_exact": mean(se), "other_exact": mean(oe),
                             "self_logp_drop": mean(sd), "other_logp_drop": mean(od)}
            def first(th):
                for sc in SCALES:
                    if curve[sc]["self_exact"] <= th: return sc
                return None
            s50, s0 = first(0.5), first(0.05)
            r = {"curve": curve, "s50": s50, "other_exact_at_s50": curve[s50]["other_exact"] if s50 else None,
                 "s0": s0, "other_exact_at_s0": curve[s0]["other_exact"] if s0 else None,
                 "other_exact_at_2s0": curve[2 * s0]["other_exact"] if s0 and 2 * s0 in curve else None}
            key = f"{name}{'' if k is None else f'_k{k}'}_{level}"
            res[key] = r
            fmt = lambda v: "  –  " if v is None else f"{v:.3f}"
            print(f"[{tag}] {key:16s} s50 {s50}  other@s50 {fmt(r['other_exact_at_s50'])}   s0 {s0}  other@s0 {fmt(r['other_exact_at_s0'])}  "
                  f"other@2s0 {fmt(r['other_exact_at_2s0'])}   curve self/other: " +
                  " ".join(f"{sc}:{curve[sc]['self_exact']:.2f}/{curve[sc]['other_exact']:.2f}" for sc in SCALES), flush=True)
        del S; torch.cuda.empty_cache()
    out[tag] = res
    os.makedirs(os.path.dirname(args.out), exist_ok=True); json.dump(out, open(args.out, "w"), indent=1)
print("saved", args.out)
