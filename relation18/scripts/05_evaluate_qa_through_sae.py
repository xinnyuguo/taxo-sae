"""
Step 5 (Test A): QA through the SAE reconstruction — membership / leaf / class / root questions.

The layer-9 activation at the last prompt token is replaced by the level-1 reconstruction of the chosen system
(--system none = no SAE).  leaf + membership: first-answer accuracy; type / domain questions: exact-set@|U| —
the top-|U| beam candidates must cover the union U of the entity's answers under that node; chance = the same
candidates scored against another entity's union under the same node.
"""
import _bootstrap
import os as _os; DATA = _os.environ.get("HSAE_DATA", "data/generated")
import json, argparse, random, collections
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from hsae.common import Tree, match, read_jsonl
from hsae.systems import System, add_system_args
from hsae.steering import beam_candidates
from hsae.rel import RelData as FBData

ap = argparse.ArgumentParser(); add_system_args(ap, allow_none=True)
ap.add_argument("--lm", default="models/base_lm_goal/final"); ap.add_argument("--layer", type=int, default=9)
ap.add_argument("--n_leaf", type=int, default=600); ap.add_argument("--n_type", type=int, default=400); ap.add_argument("--n_domain", type=int, default=200)
ap.add_argument("--type_templates", default="0,1,2"); ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--out", required=True); args = ap.parse_args()
DEV = "cuda"; random.seed(args.seed)
tree = Tree(args.tree); fb = FBData(tree)
tok = AutoTokenizer.from_pretrained(args.lm); lm = AutoModelForCausalLM.from_pretrained(args.lm).to(DEV).eval()
sysm = None if args.system == "none" else System(args.system, tree, DEV, hier_ckpt=args.hier_model, unsup_ckpt=args.unsup_model,
                                                   property_labels=args.property_labels, k=args.k)
def hook(mod, inp, out):
    hs = out[0] if isinstance(out, tuple) else out
    if hs.shape[1] > 1:
        hs = hs.clone()
        with torch.no_grad(): _, rec = sysm.L1(hs[:, -1, :])
        hs[:, -1, :] = rec
        return (hs,) + out[1:] if isinstance(out, tuple) else hs
    return out
hk = None if sysm is None else hook
def gen(q, k): return beam_candidates(lm, tok, tok(f"Q: {q}\nA:", return_tensors="pt").to(DEV), args.layer, hk, k)

results = {}
def single(rows, name):
    ok = 0
    for r in rows:
        a = gen(r["question"], 1); ok += match(r["answer"], a[0] if a else "")
    results[name] = {"n": len(rows), "acc": ok / len(rows)}; print(f"{name:10s}: acc {ok/len(rows):.3f}  n={len(rows)}")
mem = read_jsonl(f"{DATA}/qa_membership.jsonl"); single(random.sample(mem, min(args.n_leaf, len(mem))), "membership")
leaf = [r for r in read_jsonl(f"{DATA}/qa_train.jsonl") if r["template_idx"] == 0]
single(random.sample(leaf, min(args.n_leaf, len(leaf))), "leaf")
tids = [int(t) for t in args.type_templates.split(",")]
tl = [r for r in read_jsonl(f"{DATA}/qa_node_level.jsonl") if r["template_idx"] in tids]
for kind, n in [("class", args.n_type), ("root", args.n_domain)]:
    items = random.sample([r for r in tl if r["kind"] == kind], min(n, len([r for r in tl if r["kind"] == kind])))
    by_node = collections.defaultdict(list) if False else {}
    for r in items: by_node.setdefault(r["node"], []).append(r)
    hit = n1 = cex = cn = 0
    for r in items:
        U = r["union"]; cands = gen(r["question"], len(U))
        hit += all(any(match(v, c) for c in cands) for v in U); n1 += any(match(v, cands[0]) for v in U) if cands else 0
        others = [o for o in by_node[r["node"]] if o["e1"] != r["e1"] and sorted(o["union"]) != sorted(U)]
        if others:
            o = random.choice(others); cex += all(any(match(v, c) for c in cands[:len(o["union"])]) for v in o["union"]); cn += 1
    results[kind] = {"n": len(items), "exact_set": hit / len(items), "top1": n1 / len(items), "chance": cex / max(cn, 1)}
    print(f"{kind:10s}: exact-set@|U| {hit/len(items):.3f} (chance {cex/max(cn,1):.3f})  top-1 {n1/len(items):.3f}  n={len(items)}")
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump({"system": args.system, "lm": args.lm, "results": results}, open(args.out, "w"), indent=1); print("saved", args.out)
