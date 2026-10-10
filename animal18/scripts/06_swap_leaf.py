"""Step 6 (Test C): leaf-level swap — steer level 1, propagate up, LM answers.

Test C — Leaf-level swap.

Leaf question with source leaf s, target leaf t != s.  Slot edit (AlignSAE): encode with SAE1, set the source leaf
slot (or its k features) to 0 and the target's to alpha, decode, and let the LM continue from the decoded state
(--lm_input h1, paper).  --steer_form slot_edit edits the last prompt token only; slot_edit_all (paper, animal world)
applies the same edit at every prompt position.  Swaps are grouped by the tree relation of (s, t) and scored on LM behaviour:
    answer (strict)      first answer == target leaf's value for the asked property
    answer (in-tgt-set)  first answer is ANY value the target leaf holds
Readouts (leaf / class / root hit after the swap) are printed for sanity only.
"""
import _bootstrap
import json, argparse, random
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from hsae.common import Tree, read_jsonl, first_word
from hsae.systems import build_system, add_system_args
from hsae.steering import generate_with_hook, last_token_hidden

ap = argparse.ArgumentParser()
add_system_args(ap)
ap.add_argument("--lm", default="models/base_lm_goal")
ap.add_argument("--layer", type=int, default=9)
ap.add_argument("--alphas", default="20,50,100,200,500")
ap.add_argument("--steer_pos", default="all", choices=["last", "all"], help="recorded in the output; the edited positions are set by --steer_form")
ap.add_argument("--lm_input", default="h3", choices=["h3", "h1"],
                help="what the LM sees at the last token: h3'' (top of the stack, paper) or h1' (skip levels 2-3)")
ap.add_argument("--steer_form", default="slot_edit_all_delta", choices=["slot_edit", "slot_edit_all", "slot_edit_all_delta"],
                help="slot_edit_all_delta = slot_edit_all but error-preserving: each position gets h + decode(z_edit) - decode(z). slot_edit = AlignSAE protocol: encode the last token, zero the source leaf slot/features, set the target to alpha, decode, replace the last token (use with --steer_pos last --lm_input h1). slot_edit_all = the same edit at EVERY prompt position (each position encoded, source slot/features zeroed, target set to alpha, decoded and replaced); the last token is handled exactly as in slot_edit.")
ap.add_argument("--n_qa", type=int, default=60)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--qa", default="data/generated/qa_train.jsonl")
ap.add_argument("--kg", default="data/generated/concept_kg_train.jsonl")
ap.add_argument("--out", required=True)
args = ap.parse_args()
DEV = "cuda"
tree = Tree(args.tree); leaves = tree.leaves
sysm = build_system(args, tree, DEV)
kg = {r["concept"]: r for r in read_jsonl(args.kg)}
props = json.load(open(args.property_labels))["properties"]
concept_vals = {c: {str(kg[c][p]).lower().strip() for p in props if p in kg[c]} for c in kg}
tok = AutoTokenizer.from_pretrained(args.lm)
lm = AutoModelForCausalLM.from_pretrained(args.lm).to(DEV).eval()

random.seed(args.seed)
qa = random.sample(read_jsonl(args.qa), args.n_qa)
alphas = [float(a) for a in args.alphas.split(",")]
buckets = ["within_class", "cross_class", "cross_root"]
KEYS = ("leaf", "class", "root", "ans", "tgtset", "tgtonly", "src")
rec = {a: {b: {k: [] for k in KEYS} for b in buckets} for a in alphas}

for qi, r in enumerate(qa):
    src, prop = r["concept"], r["property"]
    inputs, h0 = last_token_hidden(lm, tok, r["question"], args.layer, DEV)
    for tgt in leaves:
        if tgt == src: continue
        gold_t, gold_s = str(kg[tgt].get(prop, "")), str(kg[src].get(prop, ""))
        if not gold_t or not gold_s or gold_t == gold_s: continue        # undetectable pair
        if tree.cls_of[src] == tree.cls_of[tgt]: bucket = "within_class"
        elif tree.root_idx_of_leaf(src) == tree.root_idx_of_leaf(tgt): bucket = "cross_class"
        else: bucket = "cross_root"
        for a in alphas:
            box = {}
            def hk(mod, inp, out):
                hh = out[0] if isinstance(out, tuple) else out
                if hh.shape[1] > 1:                                   # prompt pass only
                    hh = hh.clone(); h_last = hh[:, -1, :].clone()
                    with torch.no_grad():
                        if args.steer_form != "slot_edit":            # also edit every earlier prompt position
                            P = hh[0, :-1, :]; zp, _ = sysm.L1(P); zpe = zp.clone()
                            zpe[:, sysm.leaf_feats(leaves.index(src))] = 0.0; zpe[:, sysm.leaf_feats(leaves.index(tgt))] = a
                            dec = sysm.L1.decoder(zpe)
                            hh[0, :-1, :] = P + dec - sysm.L1.decoder(zp) if args.steer_form == "slot_edit_all_delta" else dec
                        z1o, _ = sysm.L1(h_last); z1e = z1o.clone()
                        z1e[:, sysm.leaf_feats(leaves.index(src))] = 0.0; z1e[:, sysm.leaf_feats(leaves.index(tgt))] = a
                        h1s = sysm.L1.decoder(z1e); z2, h2 = sysm.L2(h1s); z3, h3 = sysm.L3(h2); z1, _ = sysm.L1(h1s)
                    box["reads"] = (sysm.read_leaf(z1), sysm.read_class(z2), sysm.read_root(z3))
                    hh[:, -1, :] = (h_last + h1s - sysm.L1.decoder(z1o)) if args.steer_form == "slot_edit_all_delta" else (h3 if args.lm_input == "h3" else h1s)
                    return (hh,) + out[1:] if isinstance(out, tuple) else hh
                return out
            ans = first_word(generate_with_hook(lm, tok, inputs, args.layer, hk)).lower().strip()
            lf, c, rt = box["reads"]
            d = rec[a][bucket]
            d["leaf"].append(lf == leaves.index(tgt))
            d["class"].append(c == tree.class_idx_of_leaf(tgt)); d["root"].append(rt == tree.root_idx_of_leaf(tgt))
            gt, gs = gold_t.lower().strip(), gold_s.lower().strip()
            hit_t = bool(ans) and (gt in ans or ans in gt); hit_s = bool(ans) and (gs in ans or ans in gs)
            in_t = any(v and ans and (v in ans or ans in v) for v in concept_vals[tgt])
            in_s = any(v and ans and (v in ans or ans in v) for v in concept_vals[src])
            d["ans"].append(hit_t); d["src"].append(hit_s and not hit_t)
            d["tgtset"].append(in_t); d["tgtonly"].append(in_t and not in_s)
    if (qi + 1) % 20 == 0: print(f"  {qi+1}/{len(qa)} questions", flush=True)

summary = {}
print(f"\nLEAF SWAP | system={args.system} k={args.k} steer_pos={args.steer_pos} lm_input={args.lm_input} form={args.steer_form}")
for a in alphas:
    print(f"alpha={a}")
    for b in buckets:
        d = rec[a][b]; n = len(d["ans"])
        if not n: continue
        m = lambda k: sum(d[k]) / n
        summary[f"alpha{a}_{b}"] = {"n": n, **{k: m(k) for k in KEYS}}
        print(f"  {b:13s} n={n:4d}  leaf {m('leaf'):.3f}  class {m('class'):.3f}  root {m('root'):.3f}  "
              f"answer(strict) {m('ans'):.3f}  answer(in-tgt-set) {m('tgtset'):.3f}  tgt-only {m('tgtonly'):.3f}  src {m('src'):.3f}")
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump({"system": args.system, "k": args.k, "lm": args.lm, "steer_pos": args.steer_pos, "lm_input": args.lm_input,
           "steer_form": args.steer_form, "summary": summary}, open(args.out, "w"), indent=1)
print("saved", args.out)
