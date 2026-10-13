# TaxoSAE: relation world (paper system)

The paper system on the relation world: 18 relations (leaves), 6 relation types (classes) and 3 domains (roots), read from GPT-2 layer 6 at the last prompt token.

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
bash run.sh act        # collect layer-6 activations (already in data/01_activations)
bash run.sh train      # stage 1 + stage 2                 -> models/stage1_L6, models/hier_sae_L6
bash run.sh baselines  # Baseline1, Baseline2, Baseline0   -> models/baseline{1,2}_L6, models/baseline0/{id,s1,s2,s3}
bash run.sh test       # Tests A-F, all systems, parallel  -> results/{A,B,C,D,E,F}
bash run.sh tables     # tables A-F                        -> results/tables.md
bash run.sh figures    # paper figures                     -> figures/pdf, figures/png
```

- `GPUS="0,1,2" bash run.sh test` spreads the jobs over several GPUs. The default is GPU 0.
- Every stage skips outputs that already exist. This means the checkpoints shipped in `models/` are re-used, and an interrupted stage can be resumed by running it again.
- Logs go to `logs/<job>.log`.

## Layout

```
data/00_taxonomy     tree.json (leaf -> class -> root), property_labels.json (attribute values of every node)
data/01_activations  qa_train.jsonl, qa_test_ood.jsonl -> L6_{train,ooda}_layer6.pkl (SAE training; Tests E, F)
data/02_test_A       qa_membership.jsonl, qa_train.jsonl, qa_node_level.jsonl, people.csv
data/03_test_B       leaf questions (qa_train, qa_test_ood = new templates, qa_ood_b = new names, qa_ood_c = both),
                     qa_node_level_extra.jsonl (class / root questions, 10 unseen templates per node; templates in
                     node_templates_extra/), qa_node_level.jsonl, people.csv
data/04_test_C, data/05_test_D   qa_train.jsonl, people.csv (questions, facts of every person)
data/06_test_E, data/07_test_F   links to the training / held-out activations
models/              base_lm (fine-tuned GPT-2), stage1_L6, hier_sae_L6, baseline1_L6, baseline2_L6, baseline0/*
scripts/             pipeline.py, make_tables.py, the training / test scripts and the hsae package
figures/             make_figures.py (+ data and drawing helpers)
results/             test outputs, tables.md
```

## Tests

| test | what it measures | setting |
|---|---|---|
| A | QA with the activation replaced by the leaf module's reconstruction | membership / leaf: accuracy; class / root: exact-set@\|U\| |
| B | readout of each level's concept: argmax over its slots (ours) or the nodes' features (baselines) | abstains (scored wrong) when top − runner-up ≤ 10% of top; diagonal accuracy (mean per-concept accuracy) |
| C | leaf swap: zero the source slot, set the target slot to α, decode, replace the last token | 2,550 pairs; α ∈ {20, 50, 100, 200, 500}, best by strict accuracy |
| D | class swap: same edit at the class level | 600 cross-root pairs; Target-only coverage@\|U\|, Changed exact-set@\|U\| |
| E | fragmentation: EffFeat / Top1C of each concept's activation mass | ours over its concept slots, baselines over the full dictionary |
| F | selectivity: ablate a concept and check exact match on its own vs. other concepts | Own / Other at the smallest strength with Own ≤ 0.05 (else s = 160) |

The expected numbers are in `results/tables.md`, which is produced by `bash run.sh tables`.
