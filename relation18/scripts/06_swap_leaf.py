"""
Step 6 (Test C): relation (leaf-level) swap — synthetic biography world.

Leaf question (entity e, source relation s); target relation t is another relation e has a (different) answer for.
Slot edit (AlignSAE) at the last prompt token: encode with SAE1, set the source slot (or its k features) to 0 and the
target's to alpha, decode, and let the LM continue from the decoded state (--lm_input h1, paper).
Buckets by tree relation of (s, t): within_type / cross_type (same domain) / cross_domain.
    answer (strict)      first answer == e's answer for t
    answer (in-tgt-set)  first answer is ANY answer relation t takes in the KG (relation switched, entity may be off)
"""
import _bootstrap
import os as _os; DATA = _os.environ.get("HSAE_DATA", "data/generated")
import json, argparse, random
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from hsae.common import Tree, read_jsonl, first_word
from hsae.systems import build_system, add_system_args
from hsae.steering import generate_with_hook, last_token_hidden
from hsae.rel import RelData as FBData

ap = argparse.ArgumentParser(); add_system_args(ap)
ap.add_argument("--lm", default="models/base_lm_goal/final"); ap.add_argument("--layer", type=int, default=9)
ap.add_argument("--alphas", default="20,50,100,200,500")
ap.add_argument("--steer_pos", default="last", choices=["last"], help="the edit is applied at the last prompt token")
ap.add_argument("--lm_input", default="h1", choices=["h3", "h1"], help="what the LM continues from: h1 (decoded leaf state, paper) or h3")
ap.add_argument("--steer_form", default="slot_edit", choices=["slot_edit"], help="AlignSAE slot edit")
ap.add_argument("--n_qa", type=int, default=150); ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--out", required=True); args = ap.parse_args()
DEV = "cuda"; random.seed(args.seed)
tree = Tree(args.tree); leaves = tree.leaves; fb = FBData(tree)
sysm = build_system(args, tree, DEV)
tok = AutoTokenizer.from_pretrained(args.lm); lm = AutoModelForCausalLM.from_pretrained(args.lm).to(DEV).eval()
def answer_of(e, r): return fb.facts[e][r]
# questions whose entity has >= 2 relations with different answers (otherwise nothing to swap to)
pool = [r for r in read_jsonl(f"{DATA}/qa_train.jsonl") if r["template_idx"] == 0
        and any(answer_of(r["e1"], x) != r["answer"] for x in fb.facts[r["e1"]] if x != r["concept"])]
qa = random.sample(pool, min(args.n_qa, len(pool)))
alphas = [float(a) for a in args.alphas.split(",")]
buckets = ["within_class", "cross_class", "cross_root"]
KEYS = ("leaf", "class", "root", "ans", "tgtset", "src")
rec = {a: {b: {k: [] for k in KEYS} for b in buckets} for a in alphas}
def ans_hit(ans, gold): g = gold.lower().strip(); return bool(ans) and (g in ans or ans in g)

for qi, r in enumerate(qa):
    e, src = r["e1"], r["concept"]
    inputs, h0 = last_token_hidden(lm, tok, r["question"], args.layer, DEV)
    for tgt in fb.facts[e]:
        if tgt == src or answer_of(e, tgt) == answer_of(e, src): continue
        if tree.cls_of[src] == tree.cls_of[tgt]: bucket = "within_class"
        elif tree.root_idx_of_leaf(src) == tree.root_idx_of_leaf(tgt): bucket = "cross_class"
        else: bucket = "cross_root"
        for a in alphas:
            box = {}
            def hk(mod, inp, out):
                hh = out[0] if isinstance(out, tuple) else out
                if hh.shape[1] > 1:
                    hh = hh.clone(); h_last = hh[:, -1, :].clone()
                    with torch.no_grad():
                        z1e, _ = sysm.L1(h_last); z1e = z1e.clone()
                        z1e[:, sysm.leaf_feats(leaves.index(src))] = 0.0; z1e[:, sysm.leaf_feats(leaves.index(tgt))] = a
                        h1s = sysm.L1.decoder(z1e); z2, h2 = sysm.L2(h1s); z3, h3 = sysm.L3(h2); z1, _ = sysm.L1(h1s)
                    box["reads"] = (sysm.read_leaf(z1), sysm.read_class(z2), sysm.read_root(z3))
                    hh[:, -1, :] = h3 if args.lm_input == "h3" else h1s
                    return (hh,) + out[1:] if isinstance(out, tuple) else hh
                return out
            raw = generate_with_hook(lm, tok, inputs, args.layer, hk).strip().split("\n")[0].lower().strip()
            ans = raw.split(".")[0].strip()                      # entity names contain underscores, keep the whole first line
            lf, c, rt = box["reads"]; d = rec[a][bucket]
            d["leaf"].append(lf == leaves.index(tgt)); d["class"].append(c == tree.class_idx_of_leaf(tgt)); d["root"].append(rt == tree.root_idx_of_leaf(tgt))
            hit_t, hit_s = ans_hit(ans, answer_of(e, tgt)), ans_hit(ans, answer_of(e, src))
            d["ans"].append(hit_t); d["src"].append(hit_s and not hit_t)
            d["tgtset"].append(any(ans_hit(ans, v) for v in fb.rel_answers[tgt]))
    if (qi + 1) % 25 == 0: print(f"  {qi+1}/{len(qa)} questions", flush=True)

summary = {}
print(f"\nLEAF SWAP | system={args.system} k={args.k} steer_pos={args.steer_pos} lm_input={args.lm_input} form={args.steer_form}")
for a in alphas:
    print(f"alpha={a}")
    for b in buckets:
        d = rec[a][b]; n = len(d["ans"])
        if not n: continue
        m = lambda k: sum(d[k]) / n; summary[f"alpha{a}_{b}"] = {"n": n, **{k: m(k) for k in KEYS}}
        print(f"  {b:13s} n={n:4d}  leaf {m('leaf'):.3f}  class {m('class'):.3f}  root {m('root'):.3f}  "
              f"answer(strict) {m('ans'):.3f}  answer(in-tgt-set) {m('tgtset'):.3f}  src {m('src'):.3f}")
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump({"system": args.system, "k": args.k, "lm": args.lm, "steer_pos": args.steer_pos, "lm_input": args.lm_input,
           "steer_form": args.steer_form, "summary": summary}, open(args.out, "w"), indent=1); print("saved", args.out)
