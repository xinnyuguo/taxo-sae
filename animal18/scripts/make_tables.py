"""Tables A-F of the 18-animal world from results/ (written by `run.sh test`), in the paper layout.

Conventions
  A  QA through the leaf module's reconstruction; membership / leaf: accuracy, class / root: exact-set@|U|.
     Baseline2 shares our leaf module (row = ours); Baseline0 shares Baseline1's dictionary (row = Baseline1).
  B  readout = argmax over a level's concept slots (ours) or the nodes' selected features (baselines), abstaining
     (scored wrong) when the top node beats the runner-up by <= 10% of the top value; AlignSAE diagonal accuracy
     (mean per-concept accuracy). Leaf questions: four tiers; class questions: unseen templates; root questions: 9 new descriptions per root that avoid the
     words vertebrate / invertebrate (e.g. 'animal lacking a backbone'; data/00_taxonomy/extra_surfaces.json).
  C  leaf swap (edit at every prompt position): best alpha by overall strict accuracy over all 1,745 source-target pairs.
  D  class swap (edit at every prompt position): cross-root pairs; best alpha by (cross-root + same-root) Target-only coverage@|U|.
  E  EffFeat / Top1C; ours over its concept slots, baselines over their full dictionaries.
  F  selectivity: Own / Other exact match at the smallest ablation strength with Own <= 0.05 (else s = 160).
Baseline0 = mean over its four feature assignments. Writes results/tables.md and prints it.
"""
import json, os, statistics as st
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.parent)
R, T, B0 = "results", "L9", ("id", "s1", "s2", "s3")
BUCKETS = ("within_class", "cross_class", "cross_root")
J = lambda p: json.load(open(p))
f3 = lambda v: "–" if v is None else ("1" if round(v, 3) == 1 else f"{v:.3f}")
f2 = lambda v: f"{v:.2f}"
mean = lambda xs: st.mean(xs)


def alphas(s): return sorted({k.split("_")[0][5:] for k in s if k.startswith("alpha")}, key=float)


def C_best(p):
    s = J(p)["summary"]; best = None
    for a in alphas(s):
        rows = [s[f"alpha{a}_{b}"] for b in BUCKETS]; n = sum(r["n"] for r in rows)
        v = (int(float(a)), sum(r["ans"] * r["n"] for r in rows) / n, sum(r["tgtset"] * r["n"] for r in rows) / n)
        if best is None or v[1] > best[1]: best = v
    return best


def D_best(p):
    s = J(p)["summary"]
    a = max(alphas(s), key=lambda a: s[f"alpha{a}_cross"]["ex_new"] + s[f"alpha{a}_same"]["ex_new"])
    x = s[f"alpha{a}_cross"]; return int(float(a)), x["ex_new"], x["ex_changed"]


def table(title, header, rows):
    out = [f"### {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


md = []
# ---- A ----
A = {n: J(f"{R}/A/A_{n}.json")["results"] for n in ("none", "hier", "full")}
rowA = lambda r: [f3(r["membership"]["acc"]), f3(r["leaf"].get("acc", r["leaf"].get("exact_set"))), f3(r["class"]["exact_set"]), f3(r["root"]["exact_set"])]
md.append(table("A: QA through reconstruction", ["", "Membership", "leaf (exact-set@|U|)", "class (exact-set@|U|)", "root (exact-set@|U|)"],
                [["base model + without SAE"] + rowA(A["none"]), ["base model + Baseline1"] + rowA(A["full"]),
                 ["base model + Baseline2"] + rowA(A["hier"]), ["base model + TaxoSAE"] + rowA(A["hier"])]))
# ---- B ----
TIERS = [("Leaf-level", "in-dist", "leafQ_in-dist"), ("", "A · new templates", "leafQ_A_new_templates"),
         ("", "B · new synonyms", "leafQ_B_new_synonyms"), ("", "C · both new", "leafQ_C_both_new"),
         ("Class-level", "new templates", "classQ_B_new_templates"), ("Root-level", "extra surfaces", "rootQ_E_extra_surfaces")]


def b_rows(name):
    if name == "b0":
        Rs = [J(f"{R}/B/B_b0_{v}.json")["results"] for v in B0]
        get = lambda k, lv: None if lv not in Rs[0][k] else mean(r[k][lv] for r in Rs)
    else:
        Rr = J(f"{R}/B/B_{name}.json")["results"]; get = lambda k, lv: Rr[k].get(lv)
    return [[lvl, tier] + [f3(get(k, lv)) for lv in ("leaf", "class", "root")] for lvl, tier, k in TIERS]


rows = []
for label, name in (("TaxoSAE", "hier"), ("Baseline1", "full"), ("Baseline2", "hybrid"), ("Baseline0", "b0")):
    if name == "hybrid" and not Path(f"{R}/B/B_hybrid.json").exists(): continue
    for i, r in enumerate(b_rows(name)): rows.append([label if i == 0 else ""] + r)
md.append(table("B: readout (diagonal accuracy, abstain 10%)", ["", "Question level", "Tier", "leaf", "class", "root"], rows))
# ---- C ----
rows = []
for label, name in (("TaxoSAE", "hier"), ("Baseline1", "full"), ("Baseline2", "hier")):
    a, s_, t_ = C_best(f"{R}/C/C_{name}.json"); rows.append([label, a, f3(s_), f3(t_)])
c0 = [C_best(f"{R}/C/C_b0_{v}.json") for v in B0]
rows.append(["Baseline0", c0[0][0] if len({x[0] for x in c0}) == 1 else "–", f3(mean(x[1] for x in c0)), f3(mean(x[2] for x in c0))])
md.append(table("C: leaf swap", ["", "α", "answer (strict)", "answer (in-tgt-set)"], rows))
# ---- D ----
rows = []
for label, name, k in (("TaxoSAE", "hier", ""), ("Baseline1", "full", 1), ("Baseline2", "hybrid", 1)):
    if not Path(f"{R}/D/D_{name}.json").exists(): continue
    a, t_, c_ = D_best(f"{R}/D/D_{name}.json"); rows.append([label, k, a, f3(t_), f3(c_)])
d0 = [D_best(f"{R}/D/D_b0_{v}.json") for v in B0]
rows.append(["Baseline0", 1, d0[0][0] if len({x[0] for x in d0}) == 1 else "–", f3(mean(x[1] for x in d0)), f3(mean(x[2] for x in d0))])
md.append(table("D: class swap (cross-root)", ["", "k", "α", "Target-only coverage@|U|", "Changed exact-set@|U|"], rows))
# ---- E ----
if Path(f"{R}/E/efffeat.json").exists():
    E = J(f"{R}/E/efffeat.json")[T]; rows = []
    for label, keys in (("TaxoSAE", ("hier_concept_leaf", "hier_concept_class", "hier_concept_root")),
                        ("Baseline1", ("full_all_leaf", "full_all_class", "full_all_root")),
                        ("Baseline2", ("hybrid_concept_leaf", "hybrid_all_class", "hybrid_all_root")),
                        ("Baseline0", ("full_all_leaf", "full_all_class", "full_all_root"))):
        for i, (lv, k) in enumerate(zip(("Leaf-level", "Class-level", "Root-level"), keys)):
            rows.append([label if i == 0 else "", lv, f2(E[k]["EffFeat"]), f3(E[k]["Top1C"])])
    md.append(table("E: EffFeat (↓) / Top1C (↑)", ["", "Level", "EffFeat", "Top1C"], rows))
# ---- F ----
def own_other(d, key):
    x = d[T][key]; c = x["curve"][str(x["s0"] or 160)]; return c["self_exact"], c["other_exact"]


if Path(f"{R}/F/selectivity_models.json").exists():
    F = J(f"{R}/F/selectivity_models.json"); rows = []
    rows.append(["TaxoSAE"] + [f3(v) for k in ("hier_leaf", "hier_class") for v in own_other(F, k)])
    rows.append(["Baseline1"] + [f3(v) for k in ("full_k1_leaf", "full_k1_class") for v in own_other(F, k)])
    rows.append(["Baseline2", "–", "–"] + [f3(v) for v in own_other(F, "hybrid_k1_class")])
    Fs = [J(f"{R}/F/selectivity_b0_{v}.json") for v in B0 if Path(f"{R}/F/selectivity_b0_{v}.json").exists()]
    if len(Fs) == len(B0):
        rows.append(["Baseline0"] + [f3(mean(own_other(f, k)[i] for f in Fs)) for k in ("full_k1_leaf", "full_k1_class") for i in (0, 1)])
    md.append(table("F: selectivity", ["", "leaf Own ↓", "leaf Other ↑", "class Own ↓", "class Other ↑"], rows))
text = "\n".join(md); open(f"{R}/tables.md", "w").write(text); print(text)
