"""Step 5 (Test B): hierarchical readout on natural inputs, in-dist and OOD tiers.

Test B — Hierarchical readout on natural inputs (no intervention).

Leaf questions : read leaf / class / root from levels 1-3 under four tiers
                 (in-dist, A new templates, B new synonyms, C both new).
Class questions: read class (+ root) — tiers: seen template, new templates, new synonyms, both new.
Root questions : read root — same tiers.
Readout = argmax over slots (hier) or over per-node mean feature activation (baselines, top-k features).
"""
import _bootstrap
import json, argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from hsae.common import Tree, CLASS_SURFACE, ROOT_SURFACE, CLASS_SYN, ROOT_SYN, art, read_jsonl
from hsae.systems import build_system, add_system_args
from hsae.steering import last_token_hidden

ap = argparse.ArgumentParser()
add_system_args(ap)
ap.add_argument("--lm", default="models/base_lm_goal")
ap.add_argument("--layer", type=int, default=9)
ap.add_argument("--qa", default="data/generated/qa_train.jsonl")
ap.add_argument("--ood_dir", default="data/generated", help="holds qa_ood_{a,b,c}.jsonl")
ap.add_argument("--abstain_tau", type=float, default=0.0,
                help="> 0: abstain (scored wrong) when the top node beats the runner-up by less than tau * |top|")
ap.add_argument("--avg", default="micro", choices=["micro", "macro"],
                help="micro: share of correct questions; macro: AlignSAE diagonal accuracy (mean per-class accuracy)")
ap.add_argument("--extra_surfaces", default="", help="json of extra class / root surface forms -> tier E_extra_surfaces")
ap.add_argument("--per_surface", type=int, default=40, help="questions per (node, surface) in tier E")
ap.add_argument("--out", required=True)
args = ap.parse_args()
DEV = "cuda"
tree = Tree(args.tree); leaves, classes, roots = tree.leaves, tree.classes, tree.roots
sysm = build_system(args, tree, DEV)
tok = AutoTokenizer.from_pretrained(args.lm)
lm = AutoModelForCausalLM.from_pretrained(args.lm).to(DEV).eval()

def slots(q):
    _, h = last_token_hidden(lm, tok, q, args.layer, DEV)
    return sysm.read_all_abstain(h, args.abstain_tau) if args.abstain_tau > 0 else sysm.read_all(h)

def bank(path):
    d = {}
    for r in read_jsonl(path):
        if r.get("synonym_idx", 0) != 0: continue
        d.setdefault(r["property_idx"], {}).setdefault(r["concept"], []).append(r["question"])
    return d
ind, ood = bank(args.qa), bank(f"{args.ood_dir}/qa_ood_a.jsonl")
ref = leaves[0]
def subst(q, surf): return q.replace(f"{art(ref)} {ref}", f"{art(surf)} {surf}").replace(ref, surf)

class Diag:
    """Readout accuracy per level. micro: share of correct questions. macro (AlignSAE's diagonal accuracy):
    Diag = (1/R) sum_i C_ii with C_ii = #{true i, read i} / #{true i}, averaged over the classes present."""
    def __init__(self): self.c = {}
    def add(self, true, pred):
        k = self.c.setdefault(int(true), [0, 0]); k[0] += int(pred == true); k[1] += 1
    def value(self):
        if not self.c: return 0.0
        if args.avg == "macro": return sum(a / n for a, n in self.c.values()) / len(self.c)
        return sum(a for a, _ in self.c.values()) / sum(n for _, n in self.c.values())

results = {}
# ---- 1. leaf questions: read all three levels ----
for tier, src in [("in-dist", args.qa), ("A_new_templates", f"{args.ood_dir}/qa_ood_a.jsonl"),
                  ("B_new_synonyms", f"{args.ood_dir}/qa_ood_b.jsonl"), ("C_both_new", f"{args.ood_dir}/qa_ood_c.jsonl")]:
    rows = read_jsonl(src)
    if tier == "in-dist":
        rows = [r for r in rows if r.get("synonym_idx", 0) == 0 and r["template_idx"] < 3]
    LF, CL, RT = Diag(), Diag(), Diag()
    for r in rows:
        a, b, c = slots(r["question"])
        LF.add(leaves.index(r["concept"]), a); CL.add(tree.class_idx_of_leaf(r["concept"]), b)
        RT.add(tree.root_idx_of_leaf(r["concept"]), c)
    n = len(rows); lf, cl, rt = LF.value() * n, CL.value() * n, RT.value() * n; results[f"leafQ_{tier}"] = {"n": n, "leaf": lf / n, "class": cl / n, "root": rt / n}
    print(f"leaf question / {tier:16s} n={n:4d}  leaf {lf/n:.3f}  class {cl/n:.3f}  root {rt/n:.3f}")
# ---- 2. class questions ----
for tier, tb, smap in [("A_seen_template", ind, CLASS_SURFACE), ("B_new_templates", ood, CLASS_SURFACE),
                       ("C_new_synonyms", ind, CLASS_SYN), ("D_both_new", ood, CLASS_SYN)]:
    CL, RT = Diag(), Diag(); fam = n = 0
    for c in classes:
        for pi, byc in tb.items():
            for q in byc.get(ref, []):
                a, b, r_ = slots(subst(q, smap[c]))
                CL.add(classes.index(c), b); RT.add(tree.root_idx_of_class(c), r_)
                fam += a >= 0 and tree.cls_of[leaves[a]] == c; n += 1
    cl, rt = CL.value() * n, RT.value() * n
    results[f"classQ_{tier}"] = {"n": n, "class": cl / n, "root": rt / n, "leaf_in_family": fam / n}
    print(f"class question / {tier:16s} n={n:4d}  class {cl/n:.3f}  root {rt/n:.3f}  (leaf in-family {fam/n:.3f})")
# ---- 3. root questions ----
for tier, tb, smap in [("A_seen_template", ind, ROOT_SURFACE), ("B_new_templates", ood, ROOT_SURFACE),
                       ("C_new_synonyms", ind, ROOT_SYN), ("D_both_new", ood, ROOT_SYN)]:
    RT = Diag(); fam = n = 0
    for rn in roots:
        for pi, byc in tb.items():
            for q in byc.get(ref, []):
                a, b, r_ = slots(subst(q, smap[rn]))
                RT.add(roots.index(rn), r_); fam += b >= 0 and tree.root_of[classes[b]] == rn; n += 1
    rt = RT.value() * n
    results[f"rootQ_{tier}"] = {"n": n, "root": rt / n, "class_in_family": fam / n}
    print(f"root  question / {tier:16s} n={n:4d}  root {rt/n:.3f}  (class in-family {fam/n:.3f})")

# ---- 4. extra surface forms (tier E): every class / root named in several new ways, seen + new templates ----
if args.extra_surfaces:
    import random as _rnd
    XS = json.load(open(args.extra_surfaces)); _g = _rnd.Random(1234)
    pool = [q for tb in (ind, ood) for byc in tb.values() for q in byc.get(ref, [])]
    CL, RT = Diag(), Diag(); fam = n = 0
    for c in classes:
        for surf in XS["class"][c]:
            for q in _g.sample(pool, min(args.per_surface, len(pool))):
                a, b, r_ = slots(subst(q, surf))
                CL.add(classes.index(c), b); RT.add(tree.root_idx_of_class(c), r_)
                fam += a >= 0 and tree.cls_of[leaves[a]] == c; n += 1
    cl, rt = CL.value() * n, RT.value() * n
    results["classQ_E_extra_surfaces"] = {"n": n, "class": cl / n, "root": rt / n, "leaf_in_family": fam / n}
    print(f"class question / E_extra_surfaces  n={n:4d}  class {cl/n:.3f}  root {rt/n:.3f}")
    RT = Diag(); fam = n = 0
    for rn in roots:
        for surf in XS["root"][rn]:
            for q in _g.sample(pool, min(args.per_surface, len(pool))):
                a, b, r_ = slots(subst(q, surf))
                RT.add(roots.index(rn), r_); fam += b >= 0 and tree.root_of[classes[b]] == rn; n += 1
    rt = RT.value() * n
    results["rootQ_E_extra_surfaces"] = {"n": n, "root": rt / n, "class_in_family": fam / n}
    print(f"root  question / E_extra_surfaces  n={n:4d}  root {rt/n:.3f}")

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump({"system": args.system, "k": args.k, "lm": args.lm, "abstain_tau": args.abstain_tau, "avg": args.avg, "results": results}, open(args.out, "w"), indent=1)
print("saved", args.out)
