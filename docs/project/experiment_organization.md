# mCREAM Experiment Organization

## Overview

This document explains how experiments are organized for systematic evaluation of mCREAM.

---

## 1. Existing Baselines (Already Configured)

### Baseline 1: Vanilla CBM (No Graph)
- **Config**: `all_configs/best_hparams/CBM/CBM_cfmnist.yaml`
- **Key Settings**: `masking_algorithm: none`, `last_layer_mask: false`
- **Run**: `python simple_main.py --config all_configs/best_hparams/CBM/CBM_cfmnist.yaml`

### Baseline 2: Single-Graph CREAM (Original)
- **Config**: `all_configs/best_hparams/CREAM/CREAM_best_cfmnist_soft_config.yaml`
- **Key Settings**: `masking_algorithm: zuko`, `last_layer_mask: true`
- **Run**: `python simple_main.py --config all_configs/best_hparams/CREAM/CREAM_best_cfmnist_soft_config.yaml`

---

## 2. mCREAM Baselines & Methods (NEW)

### Folder Structure
```
all_configs/mcream_configs/
├── cfmnist/                          # Dataset: Complete FashionMNIST
│   ├── baselines/                    # Non-learnable aggregation
│   │   ├── union_M5_low.yaml
│   │   ├── union_M5_medium.yaml
│   │   ├── union_M5_high.yaml
│   │   ├── intersection_M5_low.yaml
│   │   ├── intersection_M5_medium.yaml
│   │   ├── intersection_M5_high.yaml
│   │   ├── majority_M5_low.yaml
│   │   ├── majority_M5_medium.yaml
│   │   └── majority_M5_high.yaml
│   │
│   ├── edge/                         # Edge-level reliability learning
│   │   ├── edge_M1_medium.yaml
│   │   ├── edge_M2_medium.yaml
│   │   ├── edge_M5_low.yaml
│   │   ├── edge_M5_medium.yaml
│   │   ├── edge_M5_high.yaml
│   │   └── edge_M10_medium.yaml
│   │
│   ├── graph/                        # Graph-level attention
│   │   ├── graph_M2_medium.yaml
│   │   ├── graph_M5_low.yaml
│   │   ├── graph_M5_medium.yaml
│   │   ├── graph_M5_high.yaml
│   │   └── graph_M10_medium.yaml
│   │
│   └── combined/                     # Combined edge + graph
│       ├── combined_M5_low.yaml
│       ├── combined_M5_medium.yaml
│       └── combined_M5_high.yaml
│
├── cub/                              # Dataset: CUB-200-2011
│   └── (same structure)
│
└── celeba/                           # Dataset: CelebA
    └── (same structure)
```

---

## 3. Expert Graphs Storage

```
data/
├── FashionMNIST/
│   └── expert_graphs/
│       ├── M1/                       # Single expert (degenerate case)
│       │   └── medium/
│       ├── M2/
│       │   └── medium/
│       ├── M5/
│       │   ├── low/                  # Low disagreement
│       │   │   ├── expert_0_u2c.pt
│       │   │   ├── expert_0_c2y.pt
│       │   │   ├── ...
│       │   │   ├── expert_4_u2c.pt
│       │   │   ├── expert_4_c2y.pt
│       │   │   ├── ground_truth/
│       │   │   │   ├── u2c_star.pt
│       │   │   │   └── c2y_star.pt
│       │   │   └── config.yaml
│       │   ├── medium/               # Medium disagreement
│       │   └── high/                 # High disagreement
│       └── M10/
│           └── medium/
│
├── CUB/
│   └── expert_graphs/
│       └── (same structure)
│
└── CelebA/
    └── expert_graphs/
        └── (same structure)
```

---

## 4. Results Storage

```
experiments/
├── Complete_Concept_FMNIST/          # Dataset name
│   └── train_cbm/                    # Mode
│       ├── CBM/                      # Vanilla CBM baseline
│       │   └── last_metrics/
│       │
│       ├── CREAM/                    # Single-graph CREAM baseline
│       │   └── last_metrics/
│       │
│       └── mCREAM/                   # Our multi-expert models
│           ├── baselines/
│           │   ├── union_M5_low/
│           │   │   └── last_metrics/
│           │   │       └── results.csv
│           │   ├── union_M5_medium/
│           │   ├── intersection_M5_medium/
│           │   └── majority_M5_medium/
│           │
│           ├── edge/                 # Edge-level reliability
│           │   ├── edge_M1_medium/
│           │   ├── edge_M2_medium/
│           │   ├── edge_M5_low/
│           │   ├── edge_M5_medium/
│           │   │   └── last_metrics/
│           │   │       └── results.csv  # Contains:
│           │   │           # - test_task_accuracy
│           │   │           # - test_concept_accuracy
│           │   │           # - u2c_f1, u2c_precision, u2c_recall
│           │   │           # - c2y_f1, c2y_precision, c2y_recall
│           │   │           # - expert_weights (if graph/combined)
│           │   │           # - training_time_min
│           │   ├── edge_M5_high/
│           │   └── edge_M10_medium/
│           │
│           ├── graph/                # Graph-level attention
│           │   └── ...
│           │
│           └── combined/             # Combined method
│               └── ...
│
├── CUB/
│   └── (same structure)
│
└── CelebA/
    └── (same structure)
```

---

## 5. Naming Convention

### Config Files
```
{method}_{M}{num_experts}_{noise_level}.yaml

Examples:
- edge_M5_medium.yaml      → Edge reliability, 5 experts, medium noise
- graph_M10_high.yaml      → Graph attention, 10 experts, high noise
- combined_M5_low.yaml     → Combined, 5 experts, low noise
- union_M5_medium.yaml     → Union baseline, 5 experts, medium noise
```

### Experiment Folders
```
experiments/{dataset}/train_cbm/mCREAM/{method}/{config_name}/
```

---

## 6. Experimental Variables

### A. Number of Experts (M)
| M | Use Case |
|---|----------|
| 1 | Degenerate (single corrupted graph) |
| 2 | Minimal ensemble |
| 5 | Default (moderate diversity) |
| 10 | High diversity |

### B. Disagreement Level (Noise Severity)
| Level | p_del | p_add | p_rev | Description |
|-------|-------|-------|-------|-------------|
| low | 0.05 | 0.05 | 0.02 | Minor errors |
| medium | 0.15 | 0.10 | 0.05 | Moderate errors |
| high | 0.30 | 0.20 | 0.10 | Major errors |

### C. Aggregation Method
| Method | Learnable | Parameters |
|--------|-----------|------------|
| union | ❌ | None |
| intersection | ❌ | None |
| majority | ❌ | None |
| edge | ✅ | α (per-edge reliability) |
| graph | ✅ | π (per-expert weight) |
| combined | ✅ | α + π |

### D. Datasets
| Dataset | K (concepts) | T (classes) | DAG Size |
|---------|--------------|-------------|----------|
| Complete_Concept_FMNIST | 11 | 10 | 21×21 |
| CUB | 112 | 200 | 312×312 |
| CelebA | 39 | 2 | 41×41 |

---

## 7. Running Experiments

### Step 1: Generate Expert Graphs
```bash
# Generate for all M and noise levels
python scripts/generate_all_expert_graphs.py --dataset cfmnist
```

### Step 2: Run Baselines (CBM, CREAM)
```bash
python simple_main.py --config all_configs/best_hparams/CBM/CBM_cfmnist.yaml
python simple_main.py --config all_configs/best_hparams/CREAM/CREAM_best_cfmnist_soft_config.yaml
```

### Step 3: Run mCREAM Experiments
```bash
# Run all mCREAM configs for a dataset
python scripts/run_mcream_experiments.py --dataset cfmnist

# Or run individual config
python mcream_main.py --config all_configs/mcream_configs/cfmnist/edge/edge_M5_medium.yaml
```

### Step 4: Aggregate Results
```bash
python scripts/aggregate_results.py --dataset cfmnist --output results/cfmnist_summary.csv
```

---

## 8. Key Metrics to Compare

### Task Performance
- `test_task_accuracy` - Main metric for task prediction
- `test_concept_accuracy` - How well concepts are predicted

### Graph Recovery
- `u2c_f1`, `u2c_precision`, `u2c_recall` - Concept→concept graph recovery
- `c2y_f1`, `c2y_precision`, `c2y_recall` - Concept→task graph recovery

### Efficiency
- `training_time_min` - Training time
- `num_params` - Model parameters

### Learned Parameters (for analysis)
- `expert_weights_u2c` - Which experts are trusted for u2c
- `expert_weights_c2y` - Which experts are trusted for c2y

