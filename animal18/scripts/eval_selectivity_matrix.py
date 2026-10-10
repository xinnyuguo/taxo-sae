"""Concept x concept selectivity matrix: ablate concept i (row), exact match on the questions of concept j (column).

Same questions, ablation and strength rule as eval_selectivity.py; the strength s of each system/level is the s0 stored
in <sel>/selectivity_<tag>.json (smallest s with own exact <= 0.05, else 160).
Ideal matrix: 0 on the diagonal (own concept removed), 1 elsewhere (other concepts kept).

Usage: python scripts/eval_selectivity_matrix.py --tag L9 --k 1 --lm models/base_lm --sel results/testF --out <json>
"""
import sys, json
argv = sys.argv[1:]
get = lambda f, d: argv[argv.index(f) + 1] if f in argv else d
tag, K, SEL, OUT = get("--tag", "L9"), int(get("--k", "1")), get("--sel", "results/testF"), get("--out", None)
sys.argv = [sys.argv[0], "--tags", tag, "--lm", get("--lm", "models/base_lm"), "--positions", "last"]
src = open(__file__.replace("eval_selectivity_matrix.py", "eval_selectivity.py")).read().split("out = {}")[0]
exec(src)  # defines tree, lm, System, build_batch, score, make_delta, args, MODELS, ACTS, ...

layer = int(re.search(r"L(\d+)$", tag).group(1))
sel = json.load(open(f"{SEL}/selectivity_{tag}.json"))[tag]
rows = pickle.load(open(f"{ACTS}/{tag}_ooda_layer{layer}.pkl", "rb"))
by = collections.defaultdict(list)
for r in rows: by[r["concept_idx"]].append(r)
rng = random.Random(args.seed)
qs = [r for c in range(n_leaf) for r in rng.sample(by[c], min(args.per_leaf, len(by[c])))]
leaf_of = torch.tensor([r["concept_idx"] for r in qs])
cls_of = torch.tensor([tree.class_idx_of_leaf(tree.leaves[i]) for i in leaf_of.tolist()])
batch = build_batch(qs)
H = f"{MODELS}/hier_sae_{tag}/hier_value_final.pt"
systems = {
    "hier": System("hier", tree, DEV, hier_ckpt=H, property_labels=args.property_labels),
    f"full_k{K}": System("full", tree, DEV, unsup_ckpt=f"{MODELS}/baseline1_{tag}/unsup_full_final.pt", k=K),
    f"hybrid_k{K}": System("hybrid", tree, DEV, hier_ckpt=H, unsup_ckpt=f"{MODELS}/baseline2_{tag}/unsup_anc_final.pt",
                           property_labels=args.property_labels, k=K),
}
_, clean = score(batch, layer)
out = {"tag": tag, "layer": layer, "k": K, "leaves": tree.leaves, "classes": tree.classes,
       "clean_exact": [clean[leaf_of == j].mean().item() for j in range(n_leaf)]}
for name, S in systems.items():
    for level, n_con, lab, feats_of in (("leaf", n_leaf, leaf_of, S.leaf_feats), ("class", n_cls, cls_of, S.class_feats)):
        if name.startswith("hybrid") and level == "leaf":
            continue
        s = sel[f"{name}_{level}"]["s0"] or 160
        M = []
        for i in range(n_con):
            _, ex = score(batch, layer, make_delta(S, level, feats_of(i), s))
            M.append([ex[lab == j].mean().item() for j in range(n_con)])
        out[f"{name}_{level}"] = {"s": s, "matrix": M}
        diag = sum(M[i][i] for i in range(n_con)) / n_con
        off = sum(M[i][j] for i in range(n_con) for j in range(n_con) if i != j) / (n_con * (n_con - 1))
        print(f"{name:10s} {level:5s} s={s:<4} diag {diag:.3f}  off-diag {off:.3f}", flush=True)
json.dump(out, open(OUT or f"{RES}/testF/selectivity_matrix_{tag}_k{K}.json", "w"), indent=1)
print("saved")
