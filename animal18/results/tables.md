### A: QA through reconstruction

|  | Membership | leaf (exact-set@|U|) | class (exact-set@|U|) | root (exact-set@|U|) |
|---|---|---|---|---|
| base model + without SAE | 1 | 1 | 0.708 | 0.358 |
| base model + Baseline1 | 1 | 1 | 0.711 | 0.375 |
| base model + Baseline2 | 0.971 | 1 | 0.722 | 0.392 |
| base model + TaxoSAE | 0.971 | 1 | 0.722 | 0.392 |

### B: readout (diagonal accuracy, abstain 10%)

|  | Question level | Tier | leaf | class | root |
|---|---|---|---|---|---|
| TaxoSAE | Leaf-level | in-dist | 1 | 1 | 1 |
|  |  | A · new templates | 0.993 | 1 | 0.988 |
|  |  | B · new synonyms | 0.851 | 0.879 | 0.915 |
|  |  | C · both new | 0.843 | 0.869 | 0.903 |
|  | Class-level | new templates | – | 0.479 | 0.866 |
|  | Root-level | extra surfaces | – | – | 0.506 |
| Baseline1 | Leaf-level | in-dist | 0.688 | 0.547 | 0.565 |
|  |  | A · new templates | 0.689 | 0.582 | 0.525 |
|  |  | B · new synonyms | 0.608 | 0.526 | 0.498 |
|  |  | C · both new | 0.607 | 0.531 | 0.505 |
|  | Class-level | new templates | – | 0.225 | 0.647 |
|  | Root-level | extra surfaces | – | – | 0.432 |
| Baseline2 | Leaf-level | in-dist | 1 | 0.188 | 0.569 |
|  |  | A · new templates | 0.993 | 0.183 | 0.561 |
|  |  | B · new synonyms | 0.851 | 0.190 | 0.553 |
|  |  | C · both new | 0.843 | 0.182 | 0.553 |
|  | Class-level | new templates | – | 0.163 | 0.506 |
|  | Root-level | extra surfaces | – | – | 0.454 |
| Baseline0 | Leaf-level | in-dist | 0.016 | 0.036 | 0.125 |
|  |  | A · new templates | 0.016 | 0.034 | 0.124 |
|  |  | B · new synonyms | 0.021 | 0.037 | 0.125 |
|  |  | C · both new | 0.019 | 0.035 | 0.124 |
|  | Class-level | new templates | – | 0.029 | 0.125 |
|  | Root-level | extra surfaces | – | – | 0.125 |

### C: leaf swap

|  | α | answer (strict) | answer (in-tgt-set) |
|---|---|---|---|
| TaxoSAE | 150 | 0.436 | 0.593 |
| Baseline1 | 150 | 0.405 | 0.547 |
| Baseline2 | 150 | 0.436 | 0.593 |
| Baseline0 | – | 0.053 | 0.244 |

### D: class swap (cross-root)

|  | k | α | Target-only coverage@|U| | Changed exact-set@|U| |
|---|---|---|---|---|
| TaxoSAE |  | 200 | 0.276 | 0.182 |
| Baseline1 | 1 | 100 | 0.255 | 0.184 |
| Baseline2 | 1 | 150 | 0.214 | 0.153 |
| Baseline0 | 1 | – | 0.094 | 0.052 |

### E: EffFeat (↓) / Top1C (↑)

|  | Level | EffFeat | Top1C |
|---|---|---|---|
| TaxoSAE | Leaf-level | 4.67 | 0.637 |
|  | Class-level | 1.68 | 0.890 |
|  | Root-level | 1.00 | 1 |
| Baseline1 | Leaf-level | 644.82 | 0.015 |
|  | Class-level | 415.58 | 0.013 |
|  | Root-level | 346.92 | 0.017 |
| Baseline2 | Leaf-level | 4.67 | 0.637 |
|  | Class-level | 95.29 | 0.074 |
|  | Root-level | 86.41 | 0.047 |
| Baseline0 | Leaf-level | 644.82 | 0.015 |
|  | Class-level | 415.58 | 0.013 |
|  | Root-level | 346.92 | 0.017 |

### F: selectivity

|  | leaf Own ↓ | leaf Other ↑ | class Own ↓ | class Other ↑ |
|---|---|---|---|---|
| TaxoSAE | 0.000 | 0.905 | 0.022 | 0.913 |
| Baseline1 | 0.142 | 0.617 | 0.139 | 0.434 |
| Baseline2 | – | – | 0.086 | 0.444 |
| Baseline0 | 0.929 | 0.929 | 0.933 | 0.936 |
