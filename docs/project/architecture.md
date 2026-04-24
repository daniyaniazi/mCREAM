# CREAM Architecture

## Overview

**CREAM (Concept REAsoning Models)** is a family of Concept Bottleneck Models (CBMs) that:
1. Explicitly encodes concept-concept (C-C) and concept-task (C→Y) relationships via a reasoning graph
2. Uses a **regularized side-channel** (black-box component) to recover task performance when concepts are incomplete

Paper: [arXiv:2506.05014](https://arxiv.org/abs/2506.05014)

---

## High-Level Data Flow

```
Image (X)
  └─► Backbone (frozen ResNet18 or custom CNN)
        └─► Feature vector z
              └─► Representation Splitter (linear layer)
                    ├─► zC  (concept exogenous vars)  ──► Concept-Concept Block ──► ĉ (concept preds)
                    └─► zY  (side-channel)             ──► Concept-Task Block   ──► ŷ (task preds)
                                                               ▲
                                                               └── ĉ also feeds here
```

---

## Three Core Blocks

### 1. Representation Splitter
- A single learnable **linear layer** that partitions the backbone output `z` into:
  - `zC ∈ R^(dC × K)` — per-concept exogenous embeddings (K = num concepts, dC = latent dim per concept)
  - `zY ∈ R^(|z| - dC×K)` — side-channel representation, projected via MLP to `ẑY ∈ R^L` (L = num classes)
- `dC` is a key hyperparameter (`num_exogenous` in configs controls total size)

### 2. Concept-Concept Block (`src/models.py`)
- Enforces the **concept adjacency matrix** `AC ∈ {0,1}^(K×K)` using a **Structured Neural Network (StrNN)** from `zuko`
- Binary masks are computed as `MC = AC^T ⊗ 1_(1×dC)` (Kronecker product)
- Each concept receives input only from its parents' exogenous vectors
- Activations:
  - **Mutex groups** → `softmax` (mutually exclusive concepts, e.g., "Clothes" vs "Goods")
  - **Non-mutex concepts** → `sigmoid` (independent binary concepts)
- Output: concept logits `l̂C ∈ R^K`, then probabilities `ĉ`

### 3. Concept-Task Block (`src/models.py`)
- Enforces the **task adjacency matrix** `AY ∈ {0,1}^(K×L)` via another StrNN
- Combined mask: `MY = [AY^T ; IL]` — concepts connect to assigned classes, side-channel connects each element to one class
- Task prediction: `ŷ_j = f([ĉ_Pa_j, ẑY_j])` where `f` is a single linear layer
- **Side-channel dropout**: during training, the entire side-channel is dropped with probability `p` (hyperparameter `dropout_prob`) to force concept-based reasoning

---

## Model Classes (src/models.py)

| Class | Role |
|---|---|
| `Template_CBM_MultiClass` | Main wrapper: glues backbone (`x_to_u`) + bottleneck (`u_to_CY`). Handles train/val/test steps, concept+task loss, interventions |
| `UtoY_model` | The concept-concept + concept-task bottleneck (CREAM's structured part) |
| `X2C_model` | Concept extractor only (image → concepts), used in `train_x2c` mode |
| `C2Y_model` | Concept-to-task head only, used for the Bayes-optimal `Ctrue→Y` baseline |
| `Standard_resnet18` | Frozen ResNet18 backbone for CelebA and CUB |
| `FashionMNIST_for_CBM` | Custom small CNN backbone for FashionMNIST |
| `Template_MultiClass` | Abstract base for standard classifiers |

---

## Training Objective

```
L = L_Y(ŷ, y) + λ · Σ_k L_C_k(ĉ, c)
```

- `L_Y`: CrossEntropyLoss (multiclass) or BCEWithLogitsLoss (binary like CelebA)
- `L_C`: BCELoss or BCEWithLogitsLoss depending on `concept_representation`
- `λ` = `lambda_weight` in config (typically 1.0)
- Joint bottleneck training — both losses optimized simultaneously

---

## Key Hyperparameters (in configs under `hyperparameters_model2`)

| Parameter | Meaning |
|---|---|
| `num_concepts` (K) | Number of binary concepts |
| `num_classes` (L) | Number of output classes |
| `num_exogenous` | Total size of split representation = `dC×K + num_side_channel` |
| `num_side_channel` | Side-channel dimension (0 = no side channel) |
| `dropout_prob` (p) | Side-channel dropout rate (higher = more concept-reliant) |
| `side_dropout` | Whether to use side-channel dropout |
| `concept_representation` | `group_soft` (mutex+sigmoid), `soft` (sigmoid only), `hard`, `logits` |
| `masking_algorithm` | `zuko` (StrNN masking) or `none` (vanilla CBM) |
| `num_hidden_layers_in_maskedmlp` | Depth of StrNN (0 = linear) |
| `last_layer_mask` | Whether to apply concept mask on final prediction layer |

---

## Structured Neural Networks (StrNNs)

- Implemented via [`zuko`](https://zuko.readthedocs.io/) library (`MaskedMLP`, `MaskedLinear`)
- Binary masks enforce that `∂x_i/∂z_j = 0` when `x_j ∉ pa(x_i)` (no gradient flow from non-parents)
- Mask construction follows the algorithm in Zuko, which uses binary matrix factorization

---

## Concept Channel Importance (CCI)

CCI measures how much the model relies on concepts vs. side-channel:

```
CCI = φ_c / (φ_c + φ_y)
```

- `φ_c`, `φ_y` = SAGE values of concept channel and side-channel respectively
- CCI > 0.5 → model relies primarily on concepts (desired for interpretability)
- Computed in `src/sage_importance_functions.py` and `sage_importance.py`

---

## Inference Modes

- **With side-channel** (default): uses both concept and side-channel paths
- **Without side-channel** (interpretable mode): zero out `zY` to get purely concept-based predictions; can be toggled at inference time
