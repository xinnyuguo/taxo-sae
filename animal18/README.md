# TaxoSAE: 18-animal world (paper system)

The paper system on the 18-animal world: 18 animals (leaves), 6 classes (mammal, bird, amphibian, fish, insect, mollusc) and 2 roots (vertebrate, invertebrate), read from GPT-2 layer 9. Slot edits are applied at every prompt position.

## System

TaxoSAE (our model) has three stacked levels. Each level has 10,000 free slots, one concept slot per node, and one property slot per attribute. Training adds an orthogonality term between the concept slots and the free slots: the summed squared cross-covariance, weighted by λ = 1e-2.

Training follows a two-stage schedule:

| stage | epochs | losses |
|---|---|---|
| 1 | 100 | reconstruction + sparsity |
| 2 | 500, warm-started from stage 1 | reconstruction + sparsity + leaf CE + derived class/root CE (λ 5) + property alignment + concept/property orthogonality (0.5) + value head + concept/free orthogonality (1e-2) |

Both stages use lr 3e-4 with cosine decay to 1e-6. The hyper-parameters are fixed in `scripts/pipeline.py`.

The baselines all use one feature per node (k = 1):

- **Baseline1** (paper: DiffMean Unsup. SAE): three stacked unsupervised ReLU SAEs with 10k features each. Each node is assigned the feature with the largest gap between its mean activation on the node's questions and its mean on the strongest other node.
- **Baseline2** (paper: Hybrid SAE): our frozen leaf module, with unsupervised class and root SAEs on top. Nodes are selected the same way as in Baseline1.
- **Baseline0** (paper: Random Unsup. SAE): Baseline1's dictionary with fixed, unselected features. Feature i is assigned to node i, or the assignment is drawn from random seeds 1–3. Reported numbers are the mean over the four assignments.

## Run

Environment setup (conda, torch, requirements) and the full from-scratch recipe: see `../README.md`.

```bash
bash run.sh act        # collect layer-9 activations (already in data/01_activations)
bash run.sh train      # stage 1 + stage 2                 -> models/stage1_L9, models/hier_sae_L9
bash run.sh baselines  # Baseline1, Baseline2, Baseline0   -> models/baseline{1,2}_L9, models/baseline0/{id,s1,s2,s3}
bash run.sh test       # Tests A-F, all systems, parallel  -> results/{A,B,C,D,E,F}
bash run.sh tables     # tables A-F                        -> results/tables.md
bash run.sh figures    # paper figures                     -> figures/pdf, figures/png
```

- `GPUS="0,1,2" bash run.sh test` spreads the jobs over several GPUs. The default is GPU 0.
- Every stage skips outputs that already exist. This means the checkpoints shipped in `models/` are re-used, and an interrupted stage can be resumed by running it again.
- Logs go to `logs/<job>.log`.

## Layout

```
data/00_taxonomy     tree.json (leaf -> class -> root), property_labels.json (attribute values of every node),
                     extra_surfaces.json (9 descriptions per root / 6 per class without the node names, Test B)
data/01_activations  qa_train_x2.jsonl, qa_ood_a.jsonl -> L9_{train,ooda}_layer9.pkl (SAE training; Tests E, F)
data/02_test_A       qa_membership.jsonl, qa_train.jsonl
data/03_test_B       qa_train.jsonl (in-dist), qa_ood_a / qa_ood_b / qa_ood_c (new templates / new synonyms / both);
                     class and root questions are built from these by substituting the class / root name or description
data/04_test_C       qa_train.jsonl, concept_kg_train.jsonl (attribute values of every animal)
data/05_test_D       qa_train.jsonl (class answer sets come from property_labels.json)
data/06_test_E, data/07_test_F   links to the training / held-out activations
models/              base_lm (fine-tuned GPT-2), stage1_L9, hier_sae_L9, baseline1_L9, baseline2_L9, baseline0/*
scripts/             pipeline.py, make_tables.py, the training / test scripts and the hsae package
figures/             make_figures.py (+ data and drawing helpers)
results/             test outputs, tables.md
```

## Tests

| test | what it measures | setting |
|---|---|---|
| A | QA with the activation replaced by the leaf module's reconstruction | membership / leaf: accuracy; class / root: exact-set@\|U\| |
| B | readout of each level's concept: argmax over its slots (ours) or the nodes' features (baselines); root questions name the root by a description (no 'vertebrate' / 'invertebrate') | abstains (scored wrong) when top − runner-up ≤ 10% of top; diagonal accuracy (mean per-concept accuracy) |
| C | leaf swap: zero the source slot, set the target slot to α, decode, replace every prompt position | 1,745 pairs; α ∈ {20, 50, 100, 200, 500}, best by strict accuracy |
| D | class swap: same edit at the class level | 347 cross-root pairs; Target-only coverage@\|U\|, Changed exact-set@\|U\| |
| E | fragmentation: EffFeat / Top1C of each concept's activation mass | ours over its concept slots, baselines over the full dictionary |
| F | selectivity: ablate a concept and check exact match on its own vs. other concepts | Own / Other at the smallest strength with Own ≤ 0.05 (else s = 160) |

The expected numbers are in `results/tables.md`, which is produced by `bash run.sh tables`.
