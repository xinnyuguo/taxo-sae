"""Baseline0 (lower bound): the unsupervised Baseline1 / Baseline2 dictionaries with NO feature selection.
Each node gets a fixed feature index instead of the differential-activation choice:
  --mapping id      leaf i -> feature i of SAE1, class c -> feature c of SAE2, root r -> feature r of SAE3
  --mapping rand    distinct random indices per level (--seed)
Writes <out>/baseline1_<tag>/unsup_full_final.pt and <out>/baseline2_<tag>/unsup_anc_final.pt (same weights,
rewritten *_features), and links hier_sae_<tag> / base_lm so the test scripts can run with HSAE_MODELS=<out>.
"""
import argparse, os, random, torch
ap = argparse.ArgumentParser()
ap.add_argument("--models", default="models"); ap.add_argument("--tag", default="L6")
ap.add_argument("--mapping", choices=["id", "rand"], required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True); args = ap.parse_args()


def fixed(n_nodes, n_feat, top_k, rng):
    if args.mapping == "id":
        return {i: {"feature_indices": [(i + j * n_nodes) % n_feat for j in range(top_k)], "differential_scores": None}
                for i in range(n_nodes)}
    idx = rng.sample(range(n_feat), n_nodes * top_k)
    return {i: {"feature_indices": idx[i * top_k:(i + 1) * top_k], "differential_scores": None} for i in range(n_nodes)}


for sub, f, keys in (("baseline1", "unsup_full_final.pt", ("leaf_features", "class_features", "root_features")),
                     ("baseline2", "unsup_anc_final.pt", ("class_features", "root_features"))):
    src = f"{args.models}/{sub}_{args.tag}/{f}"
    if not os.path.exists(src): print("skip (not trained yet):", src); continue
    ck = torch.load(src, map_location="cpu", weights_only=False)
    rng = random.Random(args.seed * 1000 + len(keys))
    for k in keys:
        ck[k] = fixed(len(ck[k]), ck["n_features"], ck["top_k"], rng)
        print(sub, k, {i: v["feature_indices"][0] for i, v in ck[k].items()})
    ck["baseline0"] = {"mapping": args.mapping, "seed": args.seed}
    os.makedirs(f"{args.out}/{sub}_{args.tag}", exist_ok=True); torch.save(ck, f"{args.out}/{sub}_{args.tag}/{f}")
for link in (f"hier_sae_{args.tag}", "base_lm"):
    if not os.path.exists(f"{args.out}/{link}"): os.symlink(os.path.abspath(f"{args.models}/{link}"), f"{args.out}/{link}")
