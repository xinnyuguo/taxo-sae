"""Loss ablation table (relation world): the full system and the six stage-2 runs that drop one loss each.

Same metrics and conventions as make_tables.py (our model only):
  A  class / root QA exact-set@|U| through the leaf module's reconstruction (membership, leaf stay at 1)
  B  readout, diagonal accuracy with 10% abstention: leaf questions (four tiers, question-weighted) -> leaf / class / root;
     class questions -> class; root questions -> root (10 unseen templates per node)
  C  leaf swap strict / in-target-set at the best alpha;  D  class swap Target-only coverage@|U| (cross-root)
  E  Top1C at leaf / class / root;  F  Own / Other at the leaf and class levels
Writes results/ablation_table.md and prints it.  Run after `bash run.sh test` and `bash run.sh ablation`.
"""
import json, os, sys
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.parent)
sys.path.insert(0, "scripts")
from make_tables import C_best, D_best, J, f3, T                     # noqa: E402  (make_tables prints its tables on import)

ROWS = [("full system", "results"), ("− concept alignment", "results/ablation/no_concept"),
        ("− hierarchical CE (class/root)", "results/ablation/no_hier"), ("− property alignment", "results/ablation/no_property"),
        ("− value head", "results/ablation/no_value"), ("− concept/property orthogonality", "results/ablation/no_ortho_cp"),
        ("− concept/free orthogonality", "results/ablation/no_ortho_cf")]
LEAF_TIERS = ("leafQ_in-dist", "leafQ_A_new_templates", "leafQ_B_new_synonyms", "leafQ_C_both_new")


def row(R):
    a = J(f"{R}/A/A_hier.json")["results"]
    b = J(f"{R}/B/B_hier.json")["results"]; n = sum(b[k]["n"] for k in LEAF_TIERS)
    leaf = [sum(b[k][x] * b[k]["n"] for k in LEAF_TIERS) / n for x in ("leaf", "class", "root")]
    _, cs, ct = C_best(f"{R}/C/C_hier.json"); _, dt, _ = D_best(f"{R}/D/D_hier.json")
    e = J(f"{R}/E/efffeat.json")[T]
    F = J(f"{R}/F/selectivity_models.json")[T]
    oo = lambda k: (F[k]["curve"][str(F[k]["s0"] or 160)]["self_exact"], F[k]["curve"][str(F[k]["s0"] or 160)]["other_exact"])
    return ([f3(a["class"]["exact_set"]), f3(a["root"]["exact_set"])] + [f3(x) for x in leaf]
            + [f3(b["classQ_C_extra_templates"]["class"]), f3(b["rootQ_C_extra_templates"]["root"]), f3(cs), f3(ct), f3(dt)]
            + [f3(e[f"hier_concept_{lv}"]["Top1C"]) for lv in ("leaf", "class", "root")]
            + [f"{f3(o)} / {f3(t)}" for o, t in (oo("hier_leaf"), oo("hier_class"))])


head = ["", "A class", "A root", "B leafQ leaf", "B leafQ class", "B leafQ root", "B classQ", "B rootQ",
        "C strict", "C in-tgt", "D target-only", "E Top1C leaf", "E Top1C class", "E Top1C root", "F leaf Own/Other", "F class Own/Other"]
lines = ["### Loss ablation (relation world, layer 6; our model only)", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
lines += ["| " + " | ".join([name] + row(R)) + " |" for name, R in ROWS]
text = "\n".join(lines) + "\n"
open("results/ablation_table.md", "w").write(text); print("\n" + text)
