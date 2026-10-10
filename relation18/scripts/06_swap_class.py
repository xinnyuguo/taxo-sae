"""
Step 6 (Test D): class-level swap — synthetic biography world.

Leaf question (entity e, source relation s of type T_s); target type T_t != T_s under which e has facts.
    h -> L1 -> h1 -> L2 -> z2 ; slot edit at the last token: z2[src] = 0, z2[tgt] = alpha, decode h2' ; LM continues from h2'
    (root read from L3(h2') as a diagnostic)
Correct answer = U_t, the union of e's answers under T_t.  Over the top-|U_t| beam candidates:
    exact-set@|U|  /  Target-only coverage@|U| (U_t \\ U_s covered)  /  Changed exact-set@|U| (differs from the unsteered set)
    strict target-only (exactly |U_t \\ U_s| candidates must equal that set)  /  chance (another entity's U under T_t)
Buckets: same_domain / cross_domain (cross_domain is the headline, as in the animal setting).
"""
import _bootstrap
import os as _os; DATA = _os.environ.get("HSAE_DATA", "data/generated")
import json, argparse, random
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from hsae.common import Tree, read_jsonl, match
from hsae.systems import build_system, add_system_args
from hsae.steering import beam_candidates, last_token_hidden
from hsae.rel import RelData as FBData

ap = argparse.ArgumentParser(); add_system_args(ap)
ap.add_argument("--lm", default="models/base_lm_goal/final"); ap.add_argument("--layer", type=int, default=9)
ap.add_argument("--alphas", default="20,50,100,200,500")
ap.add_argument("--steer_pos", default="last", choices=["last"], help="the edit is applied at the last prompt token")
ap.add_argument("--lm_input", default="h2", choices=["h2", "h3"])
ap.add_argument("--steer_form", default="slot_edit", choices=["slot_edit"],
                help="AlignSAE slot edit at level 2: zero the source class slot/features, set the target to alpha, decode h2")
ap.add_argument("--n_qa", type=int, default=150); ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--out", required=True); args = ap.parse_args()
DEV = "cuda"; random.seed(args.seed)
tree = Tree(args.tree); classes = tree.classes; fb = FBData(tree)
sysm = build_system(args, tree, DEV)
tok = AutoTokenizer.from_pretrained(args.lm); lm = AutoModelForCausalLM.from_pretrained(args.lm).to(DEV).eval()
def types_of(e): return sorted({tree.cls_of[r] for r in fb.facts[e]})
pool = [r for r in read_jsonl(f"{DATA}/qa_train.jsonl") if r["template_idx"] == 0 and len(types_of(r["e1"])) >= 2]
qa = random.sample(pool, min(args.n_qa, len(pool)))
by_type_entities = {t: [e for e in fb.facts if fb.split[e] == "train" and t in types_of(e)] for t in classes}
alphas = [float(a) for a in args.alphas.split(",")]
KEYS = ("cls_hit", "root_hit", "exact", "recall", "top1", "chance", "ex_new", "ex_changed", "ex_strict", "src_stay")
rec = {a: {"bucket": [], **{k: [] for k in KEYS}} for a in alphas}; cands0_cache = {}

for qi, r in enumerate(qa):
    e, src_rel = r["e1"], r["concept"]; src_t = tree.cls_of[src_rel]; src_c = classes.index(src_t)
    U_src, _ = fb.node_union(e, src_t, "class")
    inputs, h0 = last_token_hidden(lm, tok, r["question"], args.layer, DEV)
    for tgt_t in types_of(e):
        if tgt_t == src_t: continue
        U, _ = fb.node_union(e, tgt_t, "class")
        if sorted(U) == sorted(U_src): continue
        tgt_c = classes.index(tgt_t); Un = len(U); new_vals = [v for v in U if v not in U_src]
        same_dom = tree.root_of[tgt_t] == tree.root_of[src_t]; tgt_root = tree.root_idx_of_class(tgt_t)
        for a in alphas:
            box = {}
            def hk(mod, inp, out):
                hh = out[0] if isinstance(out, tuple) else out
                if hh.shape[1] > 1:
                    hh = hh.clone()
                    with torch.no_grad():
                        _, h1 = sysm.L1(hh[:, -1, :]); z2, h2 = sysm.L2(h1)
                        z2e = z2.clone(); z2e[:, sysm.class_feats(src_c)] = 0.0; z2e[:, sysm.class_feats(tgt_c)] = a; h2s = sysm.L2.decoder(z2e)
                        z3, h3 = sysm.L3(h2s); z2s, _ = sysm.L2(h2s)
                    box["reads"] = (sysm.read_class(z2s), sysm.read_root(z3))
                    hh[:, -1, :] = h2s if args.lm_input == "h2" else h3
                    return (hh,) + out[1:] if isinstance(out, tuple) else hh
                return out
            cands = beam_candidates(lm, tok, inputs, args.layer, hk, Un)
            if (qi, tgt_c) not in cands0_cache: cands0_cache[(qi, tgt_c)] = beam_candidates(lm, tok, inputs, args.layer, None, Un)
            cands0 = cands0_cache[(qi, tgt_c)]
            cov = [v for v in U if any(match(v, c) for c in cands)]; d = rec[a]
            d["bucket"].append("same" if same_dom else "cross")
            d["cls_hit"].append(box["reads"][0] == tgt_c); d["root_hit"].append(box["reads"][1] == tgt_root)
            d["exact"].append(len(cov) == Un); d["recall"].append(len(cov) / Un)
            d["top1"].append(any(match(v, cands[0]) for v in U) if cands else False)
            others = [o for o in by_type_entities[tgt_t] if o != e]
            if others:
                oU, _ = fb.node_union(random.choice(others), tgt_t, "class")
                d["chance"].append(all(any(match(v, c) for c in cands[:len(oU)]) for v in oU))
            d["ex_new"].append(all(any(match(v, c) for c in cands) for v in new_vals) if new_vals else float("nan"))
            d["ex_changed"].append(len(cov) == Un and set(cands) != set(cands0))
            k2 = len(new_vals); d["ex_strict"].append(all(any(match(v, c) for c in cands[:k2]) for v in new_vals) if new_vals else float("nan"))
            d["src_stay"].append(bool(cands) and any(match(v, cands[0]) for v in U_src))
    if (qi + 1) % 25 == 0: print(f"  {qi+1}/{len(qa)} questions", flush=True)

def mean(v): v = [x for x in v if x == x]; return sum(v) / max(1, len(v))
summary = {}
print(f"\nCLASS SWAP | system={args.system} k={args.k} steer_pos={args.steer_pos} lm_input={args.lm_input} form={args.steer_form}")
print(f"{'alpha':>6} {'bucket':>6} {'n':>4} {'class hit':>9} {'root hit':>8} {'exact-set':>9} {'tgt-only cov':>12} {'changed':>8} {'strict':>7} {'recall':>7} {'top-1':>6} {'chance':>7} {'stay-src':>8}")
for a in alphas:
    d = rec[a]
    for b in ("all", "same", "cross"):
        idx = [i for i, bb in enumerate(d["bucket"]) if b == "all" or bb == b]
        if not idx: continue
        m = lambda k: mean([d[k][i] for i in idx if i < len(d[k])])
        summary[f"alpha{a}_{b}"] = {"n": len(idx), **{k: m(k) for k in KEYS}}
        print(f"{a:6} {b:>6} {len(idx):4d} {m('cls_hit'):9.3f} {m('root_hit'):8.3f} {m('exact'):9.3f} {m('ex_new'):12.3f} "
              f"{m('ex_changed'):8.3f} {m('ex_strict'):7.3f} {m('recall'):7.3f} {m('top1'):6.3f} {m('chance'):7.3f} {m('src_stay'):8.3f}")
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump({"system": args.system, "k": args.k, "lm": args.lm, "steer_pos": args.steer_pos, "lm_input": args.lm_input,
           "steer_form": args.steer_form, "summary": summary}, open(args.out, "w"), indent=1); print("saved", args.out)
