# mCREAM Project Overview

## 1. Main Entry Points (Scripts)

| Script | Purpose |
|--------|---------|
| `simple_main.py` | **Main script** — trains all model types, runs evaluation |
| `training_and_propagating_interventions.py` | Trains with **propagating interventions** (FashionMNIST only) |
| `training_with_removed_concepts.py` | Trains with **concept subsets removed** (CUB only) |
| `sage_importance.py` | **Standalone SAGE** calculation on pre-trained model |

---

## 2. Training Modes (via `mode` in YAML)

| Mode | Description | Script |
|------|-------------|--------|
| `train_cbm` | Full pipeline: Image → Concepts → Task (CREAM/CBM) | `simple_main.py` |
| `train_x2y` | Black-box: Image → Task (no concepts) | `simple_main.py` |
| `train_c2y` | Oracle: True Concepts → Task | `simple_main.py` |
| `train_x2c` | Concept extractor only: Image → Concepts | `simple_main.py` |
| `train_backbone` | Backbone only: Image → Embedding | `simple_main.py` |

---

## 3. Model Variants (via config)

| Model Type | Key Config Fields |
|------------|-------------------|
| **CREAM** (main) | `masking_algorithm: zuko`, `last_layer_mask: true` |
| **CREAM no side-channel** | `num_side_channel: 0`, `side_dropout: false` |
| **Vanilla CBM** | `masking_algorithm: none`, `last_layer_mask: false` |
| **CBM + side-channel** | `masking_algorithm: none`, `side_dropout: true` |
| **Black-box** | `mode: train_x2y` |

---

## 4. Datasets

| Dataset | Config Name | Concepts |
|---------|-------------|----------|
| iFMNIST (incomplete) | `Concept_FMNIST` | 8 |
| cFMNIST (complete) | `Complete_Concept_FMNIST` | 11 |
| CUB | `CUB` | 112 |
| CelebA | `CelebA` | 7 |

---

## 5. Ablation Studies (in `all_configs/others/ablations/`)

| Ablation | What it Tests |
|----------|---------------|
| `dropout_ablations/` | Different `dropout_prob` values (concept reliance) |
| `input_multiplicity_ablation/` | Different `num_exogenous` sizes |
| `maskedmlp_depth_ablation/` | Different `num_hidden_layers_in_maskedmlp` |

---

## 6. Special Experiments

| Experiment | Location | Description |
|------------|----------|-------------|
| **Dropping Concepts** | `all_configs/others/dropping_concepts/` + `training_with_removed_concepts.py` | Train with subsets of concepts removed |
| **Hard Concepts** | `all_configs/others/hard/` | Binary (0/1) concepts instead of soft |
| **Propagating Interventions** | `training_and_propagating_interventions.py` | Interventions cascade through DAG |

---

## 7. Evaluation Metrics (computed automatically)

| Metric | What it Measures |
|--------|------------------|
| Task Accuracy | Final prediction accuracy |
| Concept Accuracy | How well concepts are predicted |
| **PFI** | Permutation Feature Importance |
| **CCI** | Concept Contribution Index (from SAGE) |
| **Intervention Curve** | Accuracy vs # fixed concepts |
| Benchmark (CPU/GPU time) | Efficiency metrics |

---

## Visual Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         mCREAM EXPERIMENT TYPES                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ENTRY POINTS                                                               │
│  ┌─────────────────┐  ┌──────────────────────────┐  ┌───────────────────┐  │
│  │ simple_main.py  │  │ training_and_propagating │  │ training_with_    │  │
│  │ (main script)   │  │ _interventions.py        │  │ removed_concepts  │  │
│  └────────┬────────┘  └────────────┬─────────────┘  └─────────┬─────────┘  │
│           │                        │                          │             │
│           ▼                        ▼                          ▼             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         TRAINING MODES                               │   │
│  │  • train_cbm (CREAM/CBM)    • train_x2y (blackbox)                  │   │
│  │  • train_c2y (oracle)       • train_x2c (concept extractor)         │   │
│  │  • train_backbone                                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         MODEL VARIANTS                               │   │
│  │  • CREAM (structured + side-channel)                                │   │
│  │  • CREAM no side-channel                                            │   │
│  │  • Vanilla CBM (no structure)                                       │   │
│  │  • CBM + side-channel                                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         DATASETS                                     │   │
│  │  • iFMNIST (K=8)    • cFMNIST (K=11)                                │   │
│  │  • CUB (K=112)      • CelebA (K=7)                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         ABLATIONS                                    │   │
│  │  • Dropout (dropout_prob: 0.5, 0.7, 0.9, 0.95)                      │   │
│  │  • Depth (num_hidden_layers: 0, 1, 2)                               │   │
│  │  • Input size (num_exogenous: varies)                               │   │
│  │  • Hard vs Soft concepts                                            │   │
│  │  • Dropping concepts (CUB only)                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         EVALUATION                                   │   │
│  │  • Task/Concept Accuracy    • PFI importance                        │   │
│  │  • SAGE → CCI               • Intervention curve                    │   │
│  │  • CPU/GPU benchmarks                                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Count

| Category | Count |
|----------|-------|
| Entry point scripts | 4 |
| Training modes | 5 |
| Model variants | 5 |
| Datasets | 4 |
| Ablation types | 3+ |
| Evaluation metrics | 6+ |

**Total unique experiment configurations:** Hundreds (combinations of above)
