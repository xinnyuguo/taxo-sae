"""Relation world (18 relations / 6 types / 3 domains, GPT-2 layer 6): the paper system end to end.

System = Hierarchical SAE with 10k free slots and AlignSAE's concept/free orthogonality (raw cross-covariance, lambda 1e-2),
trained as AlignSAE: 100 epochs of reconstruction + sparsity only, then 500 epochs with every loss (warm start).
Baselines: Baseline1 (three unsupervised SAEs), Baseline2 (our frozen leaf module + unsupervised upper levels),
Baseline0 (Baseline1's dictionary with fixed, unselected features; four assignments). All baselines use k = 1.

Stages (python scripts/pipeline.py <stage>, or bash run.sh <stage>):
  act        collect layer-6 activations (skipped when data/01_activations already holds them)
  train      stage 1 (100 ep) -> stage 2 (500 ep)                       -> models/stage1_L6, models/hier_sae_L6
  baselines  Baseline1, Baseline2 (on our leaf module), Baseline0         -> models/baseline{1,2}_L6, models/baseline0/*
  test       Tests A-F for every system, in parallel on $GPUS            -> results/{A..F}
  ablation   six stage-2 runs, each without one loss (shared stage 1), + Tests A-F of each -> models/ablation, results/ablation
  all        act, train, baselines, test
Every stage skips outputs that already exist, so a stage can be re-run after an interruption.
"""
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
PY = sys.executable
GPUS = [int(g) for g in os.environ.get("GPUS", "0").split(",")]
MAX_PER_GPU = int(os.environ.get("MAX_PER_GPU", "12"))

L, T = 6, "L6"
LM = "models/base_lm/final"
TREE, PL = "data/00_taxonomy/tree.json", "data/00_taxonomy/property_labels.json"
ACT = "data/01_activations"
TRAIN_ACT, OOD_ACT = f"{ACT}/{T}_train_layer{L}.pkl", f"{ACT}/{T}_ooda_layer{L}.pkl"
HIER = f"models/hier_sae_{T}/hier_value_final.pt"
B1, B2 = f"models/baseline1_{T}/unsup_full_final.pt", f"models/baseline2_{T}/unsup_anc_final.pt"
B0 = ("id", "s1", "s2", "s3")
ALPHAS = (20, 50, 100, 200, 500)

# the system's hyper-parameters (both stages share them; stage 1 switches every alignment loss off)
SAE = (f"--activation_file {TRAIN_ACT} --tree {TREE} --property_labels {PL} --n_free 10000 "
       "--lam_sparse 1e-3 --lam_sparse_c 1e-3 --lam_sparse_p 1e-3 --lam_palign 1.0 --lam_ortho_cp 0.5 "
       "--lr 3e-4 --lr_schedule cosine --lr_min 1e-6")
STAGE1 = "--epochs 100 --lam_align 0 --lam_hier 0 --lam_palign 0 --lam_ortho_cp 0 --lam_value 0"
STAGE2 = "--epochs 500 --lam_ortho_cf 1e-2"
READOUT = "--abstain_tau 0.1 --avg macro --node_qa_extra data/03_test_B/qa_node_level_extra.jsonl"
SWAP = "--steer_form slot_edit --steer_pos last"

LOG = ROOT / "logs"; LOG.mkdir(exist_ok=True)
running = []


def free_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return {int(i): int(t) - int(u) for i, t, u in (l.split(",") for l in out.strip().splitlines())}


def reap():
    global running
    keep = []
    for p, g, n in running:
        if p.poll() is None: keep.append((p, g, n))
        elif p.returncode: print(f"[FAIL] {n} (rc {p.returncode}), see logs/{n}.log", flush=True)
    running = keep


def run(cmd, name, out, env=None, need_mb=6000):
    """launch cmd on the first GPU with room unless `out` exists; returns immediately (use wait())."""
    if Path(out).exists(): return
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    while True:
        reap(); fm = free_mb()
        for g in GPUS:
            if fm.get(g, 0) > need_mb and sum(1 for _, gg, _ in running if gg == g) < MAX_PER_GPU:
                e = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTORCH_ALLOC_CONF="expandable_segments:True",
                         PYTHONUNBUFFERED="1", **(env or {}))
                p = subprocess.Popen(cmd, shell=True, env=e, stdout=open(LOG / f"{name}.log", "w"), stderr=subprocess.STDOUT)
                running.append((p, g, name)); print(f"[run] gpu{g} {name}", flush=True); time.sleep(10)
                return
        time.sleep(15)


def wait():
    while running:
        reap(); time.sleep(15)


def sys_args(system, models="models"):
    s = f"--tree {TREE} --property_labels {PL} --layer {L} --lm {models}/base_lm/final --hier_model {models}/hier_sae_{T}/hier_value_final.pt"
    if system == "full": s += f" --unsup_model {models}/baseline1_{T}/unsup_full_final.pt --k 1"
    if system == "hybrid": s += f" --unsup_model {models}/baseline2_{T}/unsup_anc_final.pt --k 1"
    return s


def stage_act():
    for qa, out in (("qa_train", TRAIN_ACT), ("qa_test_ood", OOD_ACT)):
        run(f"{PY} scripts/03_collect_activations.py --layer {L} --model_path {LM} --train_qa {ACT}/{qa}.jsonl --output {out}",
            f"act_{qa}", out)
    wait()


def stage_train():
    run(f"{PY} scripts/04_train_hierarchical_sae.py {SAE} {STAGE1} --output_dir models/stage1_{T}",
        "train_stage1", f"models/stage1_{T}/hier_value_final.pt", need_mb=16000)
    wait()
    run(f"{PY} scripts/04_train_hierarchical_sae.py {SAE} {STAGE2} --init_from models/stage1_{T}/hier_value_final.pt "
        f"--output_dir models/hier_sae_{T}", "train_stage2", HIER, need_mb=16000)
    wait()


def stage_baselines():
    run(f"{PY} scripts/04_train_baseline1_unsupervised.py --tree {TREE} --activation_file {TRAIN_ACT} --output_dir models/baseline1_{T}",
        "baseline1", B1, need_mb=16000)
    run(f"{PY} scripts/04_train_baseline2_hybrid.py --hier_model {HIER} --tree {TREE} --property_labels {PL} "
        f"--activation_file {TRAIN_ACT} --output_dir models/baseline2_{T}", "baseline2", B2, need_mb=16000)
    wait()
    for v in B0:
        have = [Path(f"models/baseline0/{v}/baseline{i}_{T}/{f}").exists() for i, f in ((1, "unsup_full_final.pt"), (2, "unsup_anc_final.pt"))]
        if not have[0] or (not have[1] and Path(B2).exists()):
            mapping = "--mapping id" if v == "id" else f"--mapping rand --seed {v[1:]}"
            subprocess.run(f"{PY} scripts/make_baseline0.py --models models --tag {T} {mapping} --out models/baseline0/{v}",
                           shell=True, check=True, stdout=subprocess.DEVNULL)


def stage_test():
    R = "results"
    envA, envB = {"HSAE_DATA": "data/02_test_A"}, {"HSAE_DATA": "data/03_test_B"}
    envC, envD = {"HSAE_DATA": "data/04_test_C"}, {"HSAE_DATA": "data/05_test_D"}
    # (name, system, models dir) ; Baseline0 = Baseline1's system with fixed features
    systems = [("hier", "hier", "models"), ("full", "full", "models"), ("hybrid", "hybrid", "models")] + \
              [(f"b0_{v}", "full", f"models/baseline0/{v}") for v in B0]
    for name, sysn in (("none", "none"), ("hier", "hier"), ("full", "full")):        # Baseline2 = hier, Baseline0 = full
        run(f"{PY} scripts/05_evaluate_qa_through_sae.py {sys_args(sysn)} --system {sysn} --out {R}/A/A_{name}.json",
            f"A_{name}", f"{R}/A/A_{name}.json", envA)
    merges = []
    ready = lambda sysn, M: Path(f"{M}/baseline2_{T}/unsup_anc_final.pt").exists() if sysn == "hybrid" else True
    for name, sysn, M in systems:
        if not ready(sysn, M): print(f"[skip] {name}: train Baseline2 first (stage baselines), then re-run this stage", flush=True); continue
        run(f"{PY} scripts/05_evaluate_readout.py {sys_args(sysn, M)} --system {sysn} {READOUT} --out {R}/B/B_{name}.json",
            f"B_{name}", f"{R}/B/B_{name}.json", envB)
        for test, script, lm_in, env in (("C", "06_swap_leaf.py", "h1", envC), ("D", "06_swap_class.py", "h2", envD)):
            if test == "C" and name == "hybrid": continue                              # Baseline2's leaf module = ours
            parts = []
            for a in ALPHAS:
                o = f"{R}/split/{test}_{name}_a{a}.json"; parts.append(o)
                run(f"{PY} scripts/{script} {sys_args(sysn, M)} --system {sysn} {SWAP} --alphas {a} --lm_input {lm_in} --out {o}",
                    f"{test}_{name}_a{a}", o, env)
            merges.append((f"{R}/{test}/{test}_{name}.json", parts))
    envE = {"HSAE_MODELS": "models", "HSAE_ACTS": "data/06_test_E"}
    if Path(B2).exists(): run(f"{PY} scripts/eval_efffeat.py --tags {T} --tree {TREE} --property_labels {PL} --out {R}/E/efffeat.json", "E", f"{R}/E/efffeat.json", envE)
    for name, M in [("models", "models")] + [(f"b0_{v}", f"models/baseline0/{v}") for v in B0]:
        out = f"{R}/F/selectivity_{name}.json"
        if not Path(f"{M}/baseline2_{T}/unsup_anc_final.pt").exists(): continue        # Test F also scores Baseline2
        run(f"{PY} scripts/eval_selectivity.py --tags {T} --ks 1 --lm {LM} --tree {TREE} --property_labels {PL} --out {out}",
            f"F_{name}", out, {"HSAE_MODELS": M, "HSAE_ACTS": "data/07_test_F"})
    wait()
    for target, parts in merges:
        if Path(target).exists(): continue
        if not all(Path(p).exists() for p in parts): print("[missing]", target, flush=True); continue
        d = json.load(open(parts[0]))
        for p in parts[1:]: d["summary"].update(json.load(open(p))["summary"])
        Path(target).parent.mkdir(parents=True, exist_ok=True); json.dump(d, open(target, "w"), indent=1)


# ---- loss ablation: drop one loss in stage 2 (stage 1 has every alignment loss off already, so it is shared) ----
ABLATIONS = {"no_concept": "--lam_align 0 --lam_hier 0",   # concept alignment at every level
             "no_hier": "--lam_hier 0",                     # derived class / root CE (leaf CE kept)
             "no_property": "--lam_palign 0",               # property-slot alignment
             "no_value": "--lam_value 0",                   # value head
             "no_ortho_cp": "--lam_ortho_cp 0",             # concept / property orthogonality
             "no_ortho_cf": "--lam_ortho_cf 0"}             # concept / free orthogonality (AlignSAE)


def stage_ablation():
    for v, flags in ABLATIONS.items():
        M = Path(f"models/ablation/{v}"); M.mkdir(parents=True, exist_ok=True)
        for link in ("base_lm", f"baseline1_{T}", f"baseline2_{T}"):
            if not os.path.lexists(M / link): os.symlink(f"../../{link}", M / link)
        run(f"{PY} scripts/04_train_hierarchical_sae.py {SAE} {STAGE2} {flags} --init_from models/stage1_{T}/hier_value_final.pt "
            f"--output_dir {M}/hier_sae_{T}", f"abl_train_{v}", f"{M}/hier_sae_{T}/hier_value_final.pt", need_mb=16000)
    wait()
    merges = []
    for v in ABLATIONS:
        M, R = f"models/ablation/{v}", f"results/ablation/{v}"
        if not Path(f"{M}/hier_sae_{T}/hier_value_final.pt").exists(): print("[missing model]", v, flush=True); continue
        merges += test_ours(M, R, v)
    wait()
    for target, parts in merges:
        if Path(target).exists(): continue
        if not all(Path(p).exists() for p in parts): print("[missing]", target, flush=True); continue
        d = json.load(open(parts[0]))
        for p in parts[1:]: d["summary"].update(json.load(open(p))["summary"])
        Path(target).parent.mkdir(parents=True, exist_ok=True); json.dump(d, open(target, "w"), indent=1)


def test_ours(M, R, tag):
    """Tests A-F of our model in models dir M (results to R); returns the swap merges"""
    S = sys_args("hier", M); merges = []
    run(f"{PY} scripts/05_evaluate_qa_through_sae.py {S} --system hier --out {R}/A/A_hier.json", f"abl_A_{tag}", f"{R}/A/A_hier.json", {"HSAE_DATA": "data/02_test_A"})
    run(f"{PY} scripts/05_evaluate_readout.py {S} --system hier {READOUT} --out {R}/B/B_hier.json", f"abl_B_{tag}", f"{R}/B/B_hier.json", {"HSAE_DATA": "data/03_test_B"})
    for test, script, lm_in, d in (("C", "06_swap_leaf.py", "h1", "data/04_test_C"), ("D", "06_swap_class.py", "h2", "data/05_test_D")):
        parts = []
        for a in ALPHAS:
            o = f"{R}/split/{test}_hier_a{a}.json"; parts.append(o)
            run(f"{PY} scripts/{script} {S} --system hier {SWAP} --alphas {a} --lm_input {lm_in} --out {o}", f"abl_{test}_{tag}_a{a}", o, {"HSAE_DATA": d})
        merges.append((f"{R}/{test}/{test}_hier.json", parts))
    run(f"{PY} scripts/eval_efffeat.py --tags {T} --tree {TREE} --property_labels {PL} --out {R}/E/efffeat.json", f"abl_E_{tag}",
        f"{R}/E/efffeat.json", {"HSAE_MODELS": M, "HSAE_ACTS": "data/06_test_E"})
    run(f"{PY} scripts/eval_selectivity.py --tags {T} --ks 1 --lm {LM} --tree {TREE} --property_labels {PL} --out {R}/F/selectivity_models.json",
        f"abl_F_{tag}", f"{R}/F/selectivity_models.json", {"HSAE_MODELS": M, "HSAE_ACTS": "data/07_test_F"})
    return merges


STAGES = {"act": [stage_act], "train": [stage_train], "baselines": [stage_baselines], "test": [stage_test], "ablation": [stage_ablation],
          "all": [stage_act, stage_train, stage_baselines, stage_test]}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in STAGES: sys.exit(__doc__)
    for f in STAGES[sys.argv[1]]: f()
    print("done:", sys.argv[1], flush=True)
