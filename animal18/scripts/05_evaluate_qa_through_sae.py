"""Step 5 (Test A): QA through the SAE reconstruction — membership / leaf / class / root.

Test A — QA through the SAE.

The layer-9 activation at the last prompt token is replaced by the level-1 reconstruction of the
chosen system (hier / hybrid share SAE1; full = unsupervised SAE1) and the frozen LM answers
membership, leaf-, class- and root-level questions.
  leaf / membership : first-answer accuracy
  class / root      : exact-set@|U| — the top-|U| beam candidates must cover the node's whole
                      value union U; chance = the same candidates scored against another node's U.
--system none  = no SAE (the base-LM reference row).
"""
import _bootstrap
import json, argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from hsae.common import Tree, CLASS_SURFACE, ROOT_SURFACE, art, match, read_jsonl
from hsae.systems import System, add_system_args
from hsae.steering import beam_candidates

ap = argparse.ArgumentParser()
add_system_args(ap, allow_none=True)
ap.add_argument("--lm", default="models/base_lm_goal")
ap.add_argument("--layer", type=int, default=9)
ap.add_argument("--levels", default="leaf,class,root")
ap.add_argument("--qa", default="data/generated/qa_train.jsonl")
ap.add_argument("--membership_qa", default="data/generated/qa_membership.jsonl",
                help="single-answer QA file scored by first-answer accuracy ('' to skip)")
ap.add_argument("--templates", default="0,1,2")
ap.add_argument("--out", required=True)
args = ap.parse_args()
DEV = "cuda"
tree = Tree(args.tree)
pl = json.load(open(args.property_labels)); props = pl["properties"]
node_vals = {"leaf": pl["layers"]["layer1"], "class": pl["layers"]["layer2"], "root": pl["layers"]["layer3"]}
surfaces = {"leaf": {l: l for l in tree.leaves}, "class": CLASS_SURFACE, "root": ROOT_SURFACE}
node_lists = {"leaf": tree.leaves, "class": tree.classes, "root": tree.roots}

tok = AutoTokenizer.from_pretrained(args.lm)
lm = AutoModelForCausalLM.from_pretrained(args.lm).to(DEV).eval()
sysm = None if args.system == "none" else System(args.system, tree, DEV, hier_ckpt=args.hier_model,
                                                   unsup_ckpt=args.unsup_model, property_labels=args.property_labels, k=args.k)

def hook(mod, inp, out):
    hs = out[0] if isinstance(out, tuple) else out
    if hs.shape[1] > 1:                         # prompt pass only
        hs = hs.clone()
        with torch.no_grad():
            _, rec = sysm.L1(hs[:, -1, :])
        hs[:, -1, :] = rec
        return (hs,) + out[1:] if isinstance(out, tuple) else hs
    return out
hk = None if sysm is None else hook

def gen(q, k):
    inputs = tok(f"Q: {q}\nA:", return_tensors="pt").to(DEV)
    return beam_candidates(lm, tok, inputs, args.layer, hk, k)

results, rows = {}, []
# ---- membership (single answer) ----
if args.membership_qa:
    qrows = read_jsonl(args.membership_qa); ok = 0
    for r in qrows:
        a = gen(r["question"], 1); ok += match(r["answer"], a[0] if a else "")
    results["membership"] = {"n": len(qrows), "acc": ok / len(qrows)}
    print(f"membership: acc {ok/len(qrows):.3f}  n={len(qrows)}")
# ---- leaf / class / root property questions ----
tids = [int(t) for t in args.templates.split(",")]
base_q = {}
for r in read_jsonl(args.qa):
    if r["synonym_idx"] == 0 and r["template_idx"] in tids:
        base_q[(r["concept"], r["property_idx"], r["template_idx"])] = r["question"]
ref = tree.leaves[0]
for level in args.levels.split(","):
    hit = n = n1 = 0
    for node in node_lists[level]:
        surf = surfaces[level][node]
        for pi in range(len(props)):
            union = node_vals[level][node][props[pi]]
            for t in tids:
                q = base_q[(node, pi, t)] if level == "leaf" else \
                    base_q[(ref, pi, t)].replace(f"{art(ref)} {ref}", f"{art(surf)} {surf}").replace(ref, surf)
                cands = gen(q, len(union))
                exact = all(any(match(v, c) for c in cands) for v in union)
                top1 = any(match(v, cands[0]) for v in union) if cands else False
                hit += exact; n1 += top1; n += 1
                rows.append({"level": level, "node": node, "prop": props[pi], "q": q,
                             "candidates": cands, "union": union, "exact": bool(exact), "top1": bool(top1)})
    byprop = {}
    for r in rows:
        if r["level"] == level: byprop.setdefault(r["prop"], []).append(r)
    cex = cn = 0
    for r in rows:
        if r["level"] != level: continue
        for o in byprop[r["prop"]]:
            if o["node"] == r["node"]: continue
            cands = r["candidates"][:len(o["union"])]
            cex += all(any(match(v, c) for c in cands) for v in o["union"]); cn += 1
    results[level] = {"n": n, "exact_set": hit / n, "chance": cex / max(cn, 1), "top1_acc": n1 / n}
    print(f"{level:6s}: exact-set@|U| {hit/n:.3f} (chance {cex/max(cn,1):.3f})  top-1 {n1/n:.3f}  n={n}")

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump({"system": args.system, "lm": args.lm, "layer": args.layer, "results": results, "rows": rows},
          open(args.out, "w"), indent=1)
print("saved", args.out)
