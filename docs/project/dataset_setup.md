# Dataset Setup

## Overview

CREAM uses four datasets. FashionMNIST **auto-downloads**. CUB and CelebA require manual setup.

---

## 1. FashionMNIST (iFMNIST / cFMNIST) — Auto-download

**No manual steps needed.** Downloaded automatically on first run to `./data/FashionMNIST/`.

### Two variants

| Variant | `dataset_name` | Concepts (K) | Description |
|---|---|---|---|
| iFMNIST | `Concept_FMNIST` | 8 | Incomplete — hierarchical clothing categories (Tops, Bottoms, etc.) |
| cFMNIST | `Complete_Concept_FMNIST` | 11 | Complete — adds 3 seasonal concepts (Summer, Winter, Mild Seasons) |

### Included concept files
- `data/FashionMNIST/Concept_FMNIST_DAG.csv` — reasoning graph for iFMNIST
- `data/FashionMNIST/Complete_Concept_FMNIST_DAG.csv` — reasoning graph for cFMNIST
- `data/FashionMNIST/mutually_exclusive_relationships.json` — mutex groups for iFMNIST
- `data/FashionMNIST/mutually_exclusive_relationships_COMPLETE.json` — mutex groups for cFMNIST
- `data/FashionMNIST/concept_vectors/` — precomputed concept annotations

### DataModule
`data/fashionmnist_loader.py` → `ConceptFashionMNISTDataModule`

---

## 2. CUB (Caltech-UCSD Birds-200-2011)

**Requires manual download.** 200 bird species, 112 expert concept attributes.

### Download
Get the preprocessed version used by CBM papers:
- Dataset: [CUB_200_2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) (original)
- Preprocessed splits: follow the preprocessing from [CBM (Koh et al. 2020)](https://github.com/yewsiang/ConceptBottleneck)

Place at:
```
data/CUB/
  CUB_200_2011/        ← image data
  class_attr_data_10/  ← concept annotations (train/val/test splits)
```

### Included concept files (already in repo)
- `data/CUB/CUB_DAG_only_Gc.csv` — concept-concept reasoning graph
- `data/CUB/CUB_mutually_exclusive_concepts.json` — mutex groups
- `data/CUB/concept_weights_cub.pt` — class imbalance weights for concept loss

### Key stats
- 200 classes, 112 binary concepts, 11,788 images
- Backbone: ResNet18 (pretrained at `pretrained_models/CUB/version_1/`)

### DataModule
`data/CUB_loader.py` → `CUBDataModule`

---

## 3. CelebA

**Requires manual download.** Task: predict "Smiling". 7 facial concept attributes.

### Download
```bash
# Option 1: via torchvision (downloads ~1.4GB)
python -c "import torchvision; torchvision.datasets.CelebA('./data/CelebA', download=True)"

# Option 2: manual from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
# Requires: img_align_celeba.zip, list_attr_celeba.txt, list_eval_partition.txt
```

Place at:
```
data/CelebA/
  img_align_celeba/    ← ~200k face images
  list_attr_celeba.txt
  list_eval_partition.txt
```

### Included concept files (already in repo)
- `data/CelebA/final_DAG_unfair.csv` — reasoning graph (7 facial concepts → smiling)
- `data/CelebA/DAG_CGM_42.csv` — alternative graph

### Key stats
- Binary classification (smiling vs not), 7 concepts, ~200k images
- Backbone: ResNet18 (pretrained at `pretrained_models/CelebA/version_11/`)

### DataModule
`data/celeba_loader.py` → `CelebADataModule`

---

## Pretrained Backbones

All `train_cbm` configs assume a **frozen pretrained backbone**. Download from:

```
https://cloud.mi.uni-saarland.de/s/RYStkW2TgJoJPKr
```

Extract to `./pretrained_models/` so the structure matches:
```
pretrained_models/
  resnet18.pth
  FMNIST/version_0/checkpoints/epoch=49-step=10750.ckpt
  CUB/version_1/checkpoints/epoch=49-step=3750.ckpt
  CelebA/version_11/checkpoints/epoch=89-step=6840.ckpt
```

These were trained with `train_backbone` or `train_x2y` mode.

---

## DAG Files (Reasoning Graphs)

DAG CSV files are adjacency matrices where:
- Rows = source nodes (concepts or tasks)
- Columns = target nodes

The code reads them with `pd.read_csv(..., index_col=0)` and converts to a boolean tensor.

---

## Quick Readiness Checklist

| Dataset | Ready to run? | What's needed |
|---|---|---|
| iFMNIST / cFMNIST | ✅ Auto-downloads | Just pretrained backbone |
| CUB | ❌ Manual | Download + preprocess images, + backbone |
| CelebA | ❌ Manual | Download images, + backbone |

**Recommended first experiment:** iFMNIST — requires only the pretrained backbone download.
