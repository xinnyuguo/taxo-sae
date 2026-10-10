### A: QA through reconstruction

|  | Membership | leaf | class (exact-set@|U|) | root (exact-set@|U|) |
|---|---|---|---|---|
| base model + without SAE | 1 | 1 | 0.164 | 0.078 |
| base model + Baseline1 | 1 | 1 | 0.186 | 0.078 |
| base model + Baseline2 | 1 | 1 | 0.192 | 0.078 |
| base model + TaxoSAE | 1 | 1 | 0.192 | 0.078 |

### B: readout (diagonal accuracy, abstain 10%)

|  | Question level | Tier | leaf | class | root |
|---|---|---|---|---|---|
| TaxoSAE | Leaf-level | in-dist | 1 | 1 | 1 |
|  |  | A · new templates | 1 | 1 | 1 |
|  |  | B · new names | 1 | 1 | 1 |
|  |  | C · both new | 1 | 1 | 1 |
|  | Class-level | new templates | – | 0.698 | 0.920 |
|  | Root-level | new templates | – | – | 0.878 |
| Baseline1 | Leaf-level | in-dist | 0.789 | 0.828 | 0.671 |
|  |  | A · new templates | 0.789 | 0.842 | 0.745 |
|  |  | B · new names | 0.783 | 0.842 | 0.667 |
|  |  | C · both new | 0.779 | 0.821 | 0.754 |
|  | Class-level | new templates | – | 0.614 | 0.607 |
|  | Root-level | new templates | – | – | 0.637 |
| Baseline2 | Leaf-level | in-dist | 1 | 0.696 | 0.692 |
|  |  | A · new templates | 1 | 0.715 | 0.770 |
|  |  | B · new names | 1 | 0.708 | 0.732 |
|  |  | C · both new | 1 | 0.710 | 0.771 |
|  | Class-level | new templates | – | 0.549 | 0.545 |
|  | Root-level | new templates | – | – | 0.692 |
| Baseline0 | Leaf-level | in-dist | 0.044 | 0.040 | 0.017 |
|  |  | A · new templates | 0.042 | 0.040 | 0.034 |
|  |  | B · new names | 0.044 | 0.042 | 0.024 |
|  |  | C · both new | 0.043 | 0.042 | 0.039 |
|  | Class-level | new templates | – | 0.067 | 0.030 |
|  | Root-level | new templates | – | – | 0.014 |

### C: leaf swap

|  | α | answer (strict) | answer (in-tgt-set) |
|---|---|---|---|
| TaxoSAE | 100 | 0.847 | 0.881 |
| Baseline1 | 200 | 0.818 | 0.830 |
| Baseline2 | 100 | 0.847 | 0.881 |
| Baseline0 | 500 | 0.022 | 0.040 |

### D: class swap (cross-root)

|  | k | α | Target-only coverage@|U| | Changed exact-set@|U| |
|---|---|---|---|---|
| TaxoSAE |  | 100 | 0.418 | 0.418 |
| Baseline1 | 1 | 200 | 0.122 | 0.122 |
| Baseline2 | 1 | 200 | 0.163 | 0.163 |
| Baseline0 | 1 | – | 0.000 | 0.000 |

### E: EffFeat (↓) / Top1C (↑)

|  | Level | EffFeat | Top1C |
|---|---|---|---|
| TaxoSAE | Leaf-level | 3.21 | 0.748 |
|  | Class-level | 1.47 | 0.919 |
|  | Root-level | 1.06 | 0.991 |
| Baseline1 | Leaf-level | 360.80 | 0.035 |
|  | Class-level | 298.03 | 0.035 |
|  | Root-level | 283.28 | 0.031 |
| Baseline2 | Leaf-level | 3.21 | 0.748 |
|  | Class-level | 188.22 | 0.044 |
|  | Root-level | 184.87 | 0.035 |
| Baseline0 | Leaf-level | 360.80 | 0.035 |
|  | Class-level | 298.03 | 0.035 |
|  | Root-level | 283.28 | 0.031 |

### F: selectivity

|  | leaf Own ↓ | leaf Other ↑ | class Own ↓ | class Other ↑ |
|---|---|---|---|---|
| TaxoSAE | 0.003 | 1 | 0.000 | 1 |
| Baseline1 | 0.000 | 0.929 | 0.028 | 0.600 |
| Baseline2 | – | – | 0.119 | 0.598 |
| Baseline0 | 0.985 | 0.998 | 0.994 | 0.995 |
