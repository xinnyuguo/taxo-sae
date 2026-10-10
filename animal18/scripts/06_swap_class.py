"""Step 6 (Test D): class-level swap — steer level 2, LM answers with the target class's value set.

Test D — Class-level swap.

Leaf question; intervene one level up:
    h -> L1 -> h1 -> L2 -> z2 ; slot edit: z2[cls(s)] = 0, z2[c] = alpha, decode h2' ; LM continues from h2'  (--lm_input h2, paper)
    --steer_form slot_edit edits the last prompt token; slot_edit_all (paper, animal world) every prompt position.
    root is read from L3(h2') as a diagnostic (root kept on same-root swaps / followed on cross-root swaps).
The correct answer is the union U_c of the target class's descendant values.  Over the top-|U_c| beam
candidates we report (cross-root pairs are the headline):
    exact-set@|U|              candidates cover all of U_c
    Target-only coverage@|U|   candidates cover U_c \\ U_src  (values only the target class has)
    Changed exact-set@|U|      exact-set holds AND the candidate set differs from the unsteered one
    strict target-only         exactly |U_c \\ U_src| candidates, and they must equal that set
    chance                     same candidates scored against an unrelated class's union
"""
import _bootstrap
import json, argparse, random
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from hsae.common import Tree, read_jsonl, match
from hsae.systems import build_system, add_system_args
from hsae.steering import beam_candidates, last_token_hidden

ap = argparse.ArgumentParser()
add_system_args(ap)
ap.add_argument("--lm", default="models/base_lm_goal")
ap.add_argument("--layer", type=int, default=9)
ap.add_argument("--alphas", default="0.8")
ap.add_argument("--steer_pos", default="all", choices=["last", "all"], help="recorded in the output; the edited positions are set by --steer_form")
ap.add_argument("--lm_input", default="h2", choices=["h2", "h3"])
ap.add_argument("--steer_form", default="slot_edit_all_delta", choices=["slot_edit", "slot_edit_all", "slot_edit_all_delta"],
                help="slot_edit_all_delta = slot_edit_all but error-preserving: each position gets h + decode(z_edit) - decode(z), so the SAE reconstruction error is kept and only the edit is added. slot_edit = AlignSAE protocol at level 2: zero source class slot/features, set target to alpha, decode h2, LM continues from h2. slot_edit_all = the same edit at EVERY prompt position (each position encoded, source slot/features zeroed, target set to alpha, decoded and replaced); the last token is handled exactly as in slot_edit.")
ap.add_argument("--n_qa", type=int, default=60)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--qa", default="data/generated/qa_train.jsonl")
ap.add_argument("--out", required=True)
args = ap.parse_args()
DEV = "cuda"
tree = Tree(args.tree); leaves, classes, roots = tree.leaves, tree.classes, tree.roots
CLASS_VALS = json.load(open(args.property_labels))["layers"]["layer2"]
sysm = build_system(args, tree, DEV)
tok = AutoTokenizer.from_pretrained(args.lm)
lm = AutoModelForCausalLM.from_pretrained(args.lm).to(DEV).eval()

random.seed(args.seed)
qa = random.sample(read_jsonl(args.qa), args.n_qa)
alphas = [float(a) for a in args.alphas.split(",")]
KEYS = ("cls_hit", "root_hit", "exact", "recall", "top1", "chance", "ex_new", "ex_changed", "ex_strict", "src_stay")
rec = {a: {"bucket": [], **{k: [] for k in KEYS}} for a in alphas}
cands0_cache = {}

for qi, r in enumerate(qa):
    src_leaf, prop = r["concept"], r["property"]
    src_c = tree.class_idx_of_leaf(src_leaf)
    inputs, h0 = last_token_hidden(lm, tok, r["question"], args.layer, DEV)
    for tgt_c in range(len(classes)):
        if tgt_c == src_c: continue
        union = CLASS_VALS[classes[tgt_c]][prop]; src_union = CLASS_VALS[classes[src_c]][prop]
        if sorted(union) == sorted(src_union): continue                # undetectable pair
        U = len(union); new_vals = [v for v in union if v not in src_union]
        tgt_root = tree.root_idx_of_class(classes[tgt_c]); same_root = tgt_root == tree.root_idx_of_class(classes[src_c])
        for a in alphas:
            box = {}
            def hk(mod, inp, out):
                hh = out[0] if isinstance(out, tuple) else out
                if hh.shape[1] > 1:
                    hh = hh.clone()
                    with torch.no_grad():
                        _, h1 = sysm.L1(hh[:, -1, :]); z2, h2 = sysm.L2(h1)
                        delta = args.steer_form == "slot_edit_all_delta"
                        if args.steer_form != "slot_edit":       # every earlier prompt position, at level 2
                            P = hh[0, :-1, :]; _, h1p = sysm.L1(P); z2p, _ = sysm.L2(h1p); z2pe = z2p.clone()
                            z2pe[:, sysm.class_feats(src_c)] = 0.0; z2pe[:, sysm.class_feats(tgt_c)] = a; dec = sysm.L2.decoder(z2pe)
                            hh[0, :-1, :] = P + dec - sysm.L2.decoder(z2p) if delta else dec
                        z2e = z2.clone(); z2e[:, sysm.class_feats(src_c)] = 0.0; z2e[:, sysm.class_feats(tgt_c)] = a; h2s = sysm.L2.decoder(z2e)
                        z3, h3 = sysm.L3(h2s); z2s, _ = sysm.L2(h2s)
                        if delta: h_out = hh[:, -1, :] + h2s - sysm.L2.decoder(z2)
                    box["reads"] = (sysm.read_class(z2s), sysm.read_root(z3))
                    hh[:, -1, :] = h_out if delta else (h2s if args.lm_input == "h2" else h3)
                    return (hh,) + out[1:] if isinstance(out, tuple) else hh
                return out
            cands = beam_candidates(lm, tok, inputs, args.layer, hk, U)
            if (qi, tgt_c) not in cands0_cache:                        # unsteered reference candidates
                cands0_cache[(qi, tgt_c)] = beam_candidates(lm, tok, inputs, args.layer, None, U)
            cands0 = cands0_cache[(qi, tgt_c)]
            cov = [v for v in union if any(match(v, c) for c in cands)]
            d = rec[a]
            d["bucket"].append("same" if same_root else "cross")
            d["cls_hit"].append(box["reads"][0] == tgt_c); d["root_hit"].append(box["reads"][1] == tgt_root)
            d["exact"].append(len(cov) == U); d["recall"].append(len(cov) / U)
            d["top1"].append(any(match(v, cands[0]) for v in union) if cands else False)
            others = [classes[c] for c in range(len(classes)) if c not in (src_c, tgt_c)]
            ou = CLASS_VALS[random.choice(others)][prop]
            d["chance"].append(all(any(match(v, c) for c in cands[:len(ou)]) for v in ou))
            d["ex_new"].append(all(any(match(v, c) for c in cands) for v in new_vals) if new_vals else float("nan"))
            d["ex_changed"].append(len(cov) == U and set(cands) != set(cands0))
            k2 = len(new_vals)
            d["ex_strict"].append(all(any(match(v, c) for c in cands[:k2]) for v in new_vals) if new_vals else float("nan"))
            d["src_stay"].append(bool(cands) and any(match(v, cands[0]) for v in src_union))
    if (qi + 1) % 15 == 0: print(f"  {qi+1}/{len(qa)} questions", flush=True)

def mean(v):
    v = [x for x in v if x == x]; return sum(v) / max(1, len(v))
summary = {}
print(f"\nCLASS SWAP | system={args.system} k={args.k} steer_pos={args.steer_pos} lm_input={args.lm_input} form={args.steer_form}")
hdr = f"{'alpha':>6} {'bucket':>6} {'n':>4} {'class hit':>9} {'root hit':>8} {'exact-set':>9} {'tgt-only cov':>12} {'changed':>8} {'strict':>7} {'recall':>7} {'top-1':>6} {'chance':>7} {'stay-src':>8}"
print(hdr)
for a in alphas:
    d = rec[a]
    for b in ("all", "same", "cross"):
        idx = [i for i, bb in enumerate(d["bucket"]) if b == "all" or bb == b]
        if not idx: continue
        m = lambda k: mean([d[k][i] for i in idx])
        summary[f"alpha{a}_{b}"] = {"n": len(idx), **{k: m(k) for k in KEYS}}
        print(f"{a:6} {b:>6} {len(idx):4d} {m('cls_hit'):9.3f} {m('root_hit'):8.3f} {m('exact'):9.3f} {m('ex_new'):12.3f} "
              f"{m('ex_changed'):8.3f} {m('ex_strict'):7.3f} {m('recall'):7.3f} {m('top1'):6.3f} {m('chance'):7.3f} {m('src_stay'):8.3f}")
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump({"system": args.system, "k": args.k, "lm": args.lm, "steer_pos": args.steer_pos, "lm_input": args.lm_input,
           "steer_form": args.steer_form, "summary": summary}, open(args.out, "w"), indent=1)
print("saved", args.out)
