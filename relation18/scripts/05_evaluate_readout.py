"""
Step 5 (Test B): hierarchical readout on natural inputs (no intervention).

Leaf questions (read relation / type / domain from levels 1-3):
    in-dist = training rows, A = held-out templates, B = held-out entities, C = both.
Type questions (read type + domain) and domain questions (read domain):
    A = training templates, B = held-out templates, C = held-out entities, D = both.
"""
import _bootstrap
import os as _os; DATA = _os.environ.get("HSAE_DATA", "data/generated")
import json, argparse, random
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from hsae.common import Tree, read_jsonl
from hsae.systems import build_system, add_system_args
from hsae.steering import last_token_hidden

ap = argparse.ArgumentParser(); add_system_args(ap)
ap.add_argument("--lm", default="models/base_lm_goal/final"); ap.add_argument("--layer", type=int, default=9)
ap.add_argument("--n_per_tier", type=int, default=800); ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--abstain_tau", type=float, default=0.0,
                help="> 0: abstain (scored wrong) when the top node beats the runner-up by less than tau * |top|")
ap.add_argument("--avg", default="micro", choices=["micro", "macro"],
                help="micro: share of correct questions; macro: AlignSAE diagonal accuracy (mean per-class accuracy)")
ap.add_argument("--node_qa_extra", default="", help="extra class / root questions (make_node_qa_extra.py) -> tier C_extra_templates")
ap.add_argument("--out", required=True); args = ap.parse_args()
DEV = "cuda"; random.seed(args.seed)
tree = Tree(args.tree); leaves, classes, roots = tree.leaves, tree.classes, tree.roots
sysm = build_system(args, tree, DEV)
tok = AutoTokenizer.from_pretrained(args.lm); lm = AutoModelForCausalLM.from_pretrained(args.lm).to(DEV).eval()
def slots(q):
    h = last_token_hidden(lm, tok, q, args.layer, DEV)[1]
    return sysm.read_all_abstain(h, args.abstain_tau) if args.abstain_tau > 0 else sysm.read_all(h)
def sample(rows): return random.sample(rows, min(args.n_per_tier, len(rows)))

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
import os as _o
leaf_tiers = [("in-dist", "qa_train", None), ("A_new_templates", "qa_test_ood", None)] + [(t, f, None) for t, f in [("B_new_synonyms", "qa_ood_b"), ("C_both_new", "qa_ood_c")] if _o.path.exists(f"{DATA}/{f}.jsonl")]
for tier, src, filt in leaf_tiers:
    rows = sample([r for r in read_jsonl(f"{DATA}/{src}.jsonl") if filt is None or filt(r)])
    LF, CL, RT = Diag(), Diag(), Diag()
    for r in rows:
        a, b, c = slots(r["question"]); rel = leaves[r["concept_idx"]]
        LF.add(r["concept_idx"], a); CL.add(tree.class_idx_of_leaf(rel), b); RT.add(tree.root_idx_of_leaf(rel), c)
    n = len(rows); lf, cl, rt = LF.value() * n, CL.value() * n, RT.value() * n
    results[f"leafQ_{tier}"] = {"n": n, "leaf": lf / n, "class": cl / n, "root": rt / n}
    print(f"leaf question / {tier:16s} n={n:4d}  leaf {lf/n:.3f}  class {cl/n:.3f}  root {rt/n:.3f}")
tl = read_jsonl(f"{DATA}/qa_node_level.jsonl")
tiers = [("A_train_templates", "train", range(0, 2)), ("B_new_templates", "train", range(2, 4))]
if args.node_qa_extra:
    tl = tl + read_jsonl(args.node_qa_extra); tiers.append(("C_extra_templates", "train", range(4, 14)))
for tier, es, ts in tiers:
    rows = sample([r for r in tl if r["kind"] == "class" and r["template_idx"] in ts])
    CL, RT = Diag(), Diag(); fam = 0
    for r in rows:
        a, b, c = slots(r["question"]); CL.add(r["node_idx"], b); RT.add(r["root_idx"], c); fam += a >= 0 and tree.cls_of[leaves[a]] == r["node"]
    n = max(1, len(rows)); cl, rt = CL.value() * n, RT.value() * n; results[f"classQ_{tier}"] = {"n": len(rows), "class": cl / n, "root": rt / n, "leaf_in_family": fam / n}
    print(f"class question / {tier:16s} n={len(rows):4d}  class {cl/n:.3f}  root {rt/n:.3f}  (leaf in-family {fam/n:.3f})")
for tier, es, ts in tiers:
    rows = sample([r for r in tl if r["kind"] == "root" and r["template_idx"] in ts])
    RT = Diag(); fam = 0
    for r in rows:
        a, b, c = slots(r["question"]); RT.add(r["node_idx"], c); fam += b >= 0 and tree.root_of[classes[b]] == r["node"]
    n = max(1, len(rows)); rt = RT.value() * n; results[f"rootQ_{tier}"] = {"n": len(rows), "root": rt / n, "class_in_family": fam / n}
    print(f"root  question / {tier:16s} n={len(rows):4d}  root {rt/n:.3f}  (class in-family {fam/n:.3f})")
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump({"system": args.system, "k": args.k, "lm": args.lm, "abstain_tau": args.abstain_tau, "avg": args.avg, "results": results}, open(args.out, "w"), indent=1); print("saved", args.out)
