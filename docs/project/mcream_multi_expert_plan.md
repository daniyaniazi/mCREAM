# mCREAM: Multi-Expert Graph Extension Plan

## Overview

**Goal:** Extend CREAM to handle multiple expert graphs with learned edge reliability.

**Key Idea:** Instead of one fixed graph G, we have M expert graphs G^(1), G^(2), ..., G^(M) and learn which edges/experts to trust.

---

## Part 1: Expert Graph Generation

### 1.1 Starting Point: Ground Truth Graph

```
G* = Original CREAM graph (from DAG CSV files)
     - data/FashionMNIST/Concept_FMNIST_DAG.csv
     - data/CUB/CUB_DAG_only_Gc.csv
     - data/CelebA/final_DAG_unfair.csv
```

### 1.1.1 Understanding the DAG Structure

The DAG CSV contains ONE matrix with both concept→concept AND concept→task edges:

```
Full DAG (N x N) where N = num_concepts + num_tasks:

                    ┌─────────────────────┬─────────────────────┐
                    │   K Concepts        │    T Tasks          │
                    │                     │   (class labels)    │
┌───────────────────┼─────────────────────┼─────────────────────┤
│  K Concepts       │        A_C          │        A_Y          │
│                   │  (concept→concept)  │   (concept→task)    │
├───────────────────┼─────────────────────┼─────────────────────┤
│  T Tasks          │         0           │         I           │
│                   │                     │   (self-identity)   │
└───────────────────┴─────────────────────┴─────────────────────┘
```

**For Complete_Concept_FMNIST (21x21 matrix):**
- K = 11 concepts: Clothes, Tops, Bottoms, Dresses, Outers, Goods, Accessories, Shoes, Summer, Winter, Mild Seasons
- T = 10 tasks: T-Shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

**We split into two separate graphs:**
- `A_C` = concept→concept (11 x 11) — How concepts relate to each other
- `A_Y` = concept→task (11 x 10) — How concepts cause task predictions

```python
# Splitting code
A_C = full_dag[:num_concepts, :num_concepts]  # [K, K]
A_Y = full_dag[:num_concepts, num_concepts:]  # [K, T]
```

### 1.2 Corruption Operations

Generate expert graphs by corrupting G* with three operations:

| Operation | Symbol | Meaning |
|-----------|--------|---------|
| **Edge Deletion** | p_del | Expert missed a real relationship |
| **Edge Addition** | p_add | Expert believes a spurious relation |
| **Edge Reversal** | p_rev | Expert got direction wrong |

```python
def generate_expert_graph(G_star, p_del, p_add, p_rev, seed):
    """
    Generate one expert graph by corrupting ground truth.
    
    Args:
        G_star: Ground truth adjacency matrix [n_nodes, n_nodes]
        p_del: Probability of deleting existing edge
        p_add: Probability of adding non-existing edge
        p_rev: Probability of reversing edge direction
        seed: Random seed for reproducibility
    
    Returns:
        G_expert: Corrupted expert graph
    """
    np.random.seed(seed)
    G_expert = G_star.copy()
    
    for i in range(n_nodes):
        for j in range(n_nodes):
            if G_star[i, j] == 1:  # Edge exists
                # Deletion
                if np.random.random() < p_del:
                    G_expert[i, j] = 0
                # Reversal
                elif np.random.random() < p_rev:
                    G_expert[i, j] = 0
                    G_expert[j, i] = 1
            else:  # Edge doesn't exist
                # Addition
                if np.random.random() < p_add:
                    G_expert[i, j] = 1
    
    return G_expert
```

### 1.3 Disagreement Levels

| Level | p_del | p_add | p_rev | Description |
|-------|-------|-------|-------|-------------|
| **Low** | 0.05 | 0.05 | 0.02 | Experts mostly agree |
| **Medium** | 0.15 | 0.15 | 0.08 | Moderate disagreement |
| **High** | 0.40 | 0.40 | 0.20 | Significant disagreement |

### 1.4 Structured Expert Bias Types

| Expert Type | p_del | p_add | Behavior |
|-------------|-------|-------|----------|
| **Conservative** | 0.30 | 0.05 | Few edges, high precision, low recall |
| **Liberal** | 0.05 | 0.30 | Many edges, low precision, high recall |
| **Balanced** | 0.15 | 0.15 | Moderate both ways |
| **Domain-shifted** | varies | varies | Correct for subset, wrong for others |

### 1.5 Storage Structure

```
data/
├── FashionMNIST/
│   ├── Concept_FMNIST_DAG.csv              # Ground truth G*
│   └── expert_graphs/
│       ├── config.yaml                      # Generation parameters
│       ├── low_disagreement/
│       │   ├── expert_1.csv
│       │   ├── expert_2.csv
│       │   └── expert_3.csv
│       ├── medium_disagreement/
│       │   └── ...
│       ├── high_disagreement/
│       │   └── ...
│       └── structured_bias/
│           ├── conservative/
│           ├── liberal/
│           └── domain_shifted/
```

**config.yaml example:**
```yaml
ground_truth: Concept_FMNIST_DAG.csv
num_experts: 5
generation_seed: 42

low_disagreement:
  p_del: 0.05
  p_add: 0.05
  p_rev: 0.02
  
medium_disagreement:
  p_del: 0.15
  p_add: 0.15
  p_rev: 0.08

high_disagreement:
  p_del: 0.40
  p_add: 0.40
  p_rev: 0.20

structured_bias:
  conservative:
    p_del: 0.30
    p_add: 0.05
    p_rev: 0.05
  liberal:
    p_del: 0.05
    p_add: 0.30
    p_rev: 0.05
```

---

## Part 2: Baseline Merging Strategies

### 2.1 Union Graph (OR)

```python
# All edges that ANY expert includes
A_union = A_1 | A_2 | ... | A_M

# In code:
A_union = torch.any(torch.stack(expert_graphs), dim=0).float()
```

**Properties:**
- ✅ Unlikely to miss true edges
- ❌ Many spurious edges (false positives)
- ❌ Possible information leakage

### 2.2 Intersection Graph (AND)

```python
# Only edges that ALL experts agree on
A_inter = A_1 & A_2 & ... & A_M

# In code:
A_inter = torch.all(torch.stack(expert_graphs), dim=0).float()
```

**Properties:**
- ✅ Very clean graph (high precision)
- ❌ Misses useful edges (low recall)

### 2.3 Majority Vote

```python
# Edges that >50% of experts include
vote_count = torch.stack(expert_graphs).sum(dim=0)
A_majority = (vote_count > M/2).float()
```

### 2.4 Weighted Average (Fixed Weights)

```python
# Simple average
A_avg = torch.stack(expert_graphs).mean(dim=0)
# Result: soft values in [0, 1]
```

---

## Part 3: mCREAM Learning Approaches

### 3.1 Edge-Level Reliability (α)

Learn a reliability parameter for each possible edge:

```python
class EdgeReliabilityModule(nn.Module):
    def __init__(self, expert_graphs):
        """
        Args:
            expert_graphs: List of M adjacency matrices [n_nodes, n_nodes]
        """
        super().__init__()
        
        # Initialize with expert voting
        stacked = torch.stack(expert_graphs)  # [M, n_nodes, n_nodes]
        vote_score = stacked.mean(dim=0)       # [n_nodes, n_nodes]
        
        # Learnable parameters (logits, will apply sigmoid)
        # Initialize so sigmoid(alpha_init) ≈ vote_score
        alpha_init = torch.logit(vote_score.clamp(0.01, 0.99))
        self.alpha = nn.Parameter(alpha_init)
    
    def forward(self):
        """Returns soft adjacency matrix."""
        return torch.sigmoid(self.alpha)
```

**How learning happens:**
```
Forward: ŷ = f(ĉ, A_soft)  where A_soft = sigmoid(α)
Loss: L = CrossEntropy(ŷ, y)
Backward: ∂L/∂α → update α

If edge helps prediction → gradient increases α → edge becomes stronger
If edge hurts prediction → gradient decreases α → edge becomes weaker
```

### 3.2 Graph-Level Attention (π)

Learn a weight for each expert:

```python
class GraphAttentionModule(nn.Module):
    def __init__(self, num_experts):
        super().__init__()
        # Learnable expert weights (logits)
        self.pi_logits = nn.Parameter(torch.zeros(num_experts))
    
    def forward(self, expert_graphs):
        """
        Args:
            expert_graphs: [M, n_nodes, n_nodes]
        Returns:
            Weighted average graph [n_nodes, n_nodes]
        """
        pi = F.softmax(self.pi_logits, dim=0)  # [M], sums to 1
        # Weighted sum: Σ π_m * A^(m)
        A_soft = torch.einsum('m,mij->ij', pi, expert_graphs)
        return A_soft, pi
```

### 3.3 Combined: Edge + Graph Level

```python
class CombinedReliabilityModule(nn.Module):
    def __init__(self, expert_graphs):
        super().__init__()
        M = len(expert_graphs)
        n_nodes = expert_graphs[0].shape[0]
        
        # Graph-level weights
        self.pi_logits = nn.Parameter(torch.zeros(M))
        
        # Edge-level refinement
        stacked = torch.stack(expert_graphs)
        vote_score = stacked.mean(dim=0)
        self.alpha = nn.Parameter(torch.logit(vote_score.clamp(0.01, 0.99)))
    
    def forward(self, expert_graphs):
        # Step 1: Graph-level aggregation
        pi = F.softmax(self.pi_logits, dim=0)
        A_weighted = torch.einsum('m,mij->ij', pi, expert_graphs)
        
        # Step 2: Edge-level refinement
        A_soft = A_weighted * torch.sigmoid(self.alpha)
        
        return A_soft, pi, torch.sigmoid(self.alpha)
```

---

## Part 4: Model Architecture Integration

### 4.1 Where Graphs Are Used in CREAM

```
Current CREAM (src/models.py):

1. Load DAG from CSV → boolean adjacency matrix
2. Build mask for concept reasoning (A_C)
3. Build mask for task reasoning (A_Y)
4. Use masks in MaskedLinear/MaskedMLP layers
```

### 4.2 mCREAM Modifications

```python
class mCREAM_UtoC_Y(nn.Module):
    def __init__(
        self,
        expert_graphs_C,      # List of M concept graphs
        expert_graphs_Y,      # List of M task graphs
        reliability_type,     # 'edge', 'graph', or 'combined'
        # ... other CREAM params
    ):
        super().__init__()
        
        # NEW: Graph aggregation modules
        if reliability_type == 'edge':
            self.graph_agg_C = EdgeReliabilityModule(expert_graphs_C)
            self.graph_agg_Y = EdgeReliabilityModule(expert_graphs_Y)
        elif reliability_type == 'graph':
            self.graph_agg_C = GraphAttentionModule(len(expert_graphs_C))
            self.graph_agg_Y = GraphAttentionModule(len(expert_graphs_Y))
        else:
            self.graph_agg_C = CombinedReliabilityModule(expert_graphs_C)
            self.graph_agg_Y = CombinedReliabilityModule(expert_graphs_Y)
        
        # Store expert graphs as buffers (not parameters)
        self.register_buffer('expert_graphs_C', torch.stack(expert_graphs_C))
        self.register_buffer('expert_graphs_Y', torch.stack(expert_graphs_Y))
        
        # ... rest of CREAM initialization
    
    def forward(self, x):
        # 1. Get soft adjacency matrices
        A_soft_C = self.graph_agg_C(self.expert_graphs_C)
        A_soft_Y = self.graph_agg_Y(self.expert_graphs_Y)
        
        # 2. Feature extraction
        u = self.u2u_model(x)
        Uc = u[:, :self.num_exogenous - self.num_side_channel]
        Uy = u[:, self.num_exogenous - self.num_side_channel:]
        
        # 3. Concept prediction with soft mask
        c = self.u2c_model(Uc, mask=A_soft_C)  # Modified to use soft mask
        c = self.concept_activation_function(c)
        
        # 4. Side channel
        s = self.side_channel(Uy) if self.num_side_channel > 0 else None
        
        # 5. Task prediction with soft mask
        if s is not None:
            last_input = torch.cat([c, s], dim=1)
        else:
            last_input = c
        y = self.last_layer(last_input, mask=A_soft_Y)  # Modified
        
        return y, c
```

### 4.3 Modified MaskedLinear for Soft Masks

```python
class SoftMaskedLinear(nn.Module):
    """Linear layer with learnable soft mask."""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x, mask):
        """
        Args:
            x: Input [batch, in_features]
            mask: Soft mask [out_features, in_features], values in [0,1]
        """
        # Apply soft mask to weights
        masked_weight = self.weight * mask
        return F.linear(x, masked_weight, self.bias)
```

---

## Part 5: Loss Functions

### 5.1 Base Loss (Same as CREAM)

```python
L_base = L_task + λ * L_concept

# L_task: CrossEntropy for classification
# L_concept: BCE for concept prediction
```

### 5.2 Graph Regularization Losses

#### Prior Consistency Loss
Keep learned graph close to expert consensus:

```python
def prior_loss(alpha, expert_graphs):
    """
    Args:
        alpha: Learned edge reliabilities [n_nodes, n_nodes]
        expert_graphs: [M, n_nodes, n_nodes]
    """
    # Expert voting score
    vote_score = expert_graphs.mean(dim=0)  # [n_nodes, n_nodes]
    
    # MSE between learned and voted
    return F.mse_loss(torch.sigmoid(alpha), vote_score)
```

#### Sparsity Loss
Encourage sparse graph:

```python
def sparsity_loss(alpha):
    """Penalize too many edges."""
    return torch.sigmoid(alpha).sum()
```

#### Acyclicity Loss (Optional)
Ensure DAG property:

```python
def acyclicity_loss(A_soft):
    """Penalize cycles using trace of matrix exponential."""
    # tr(e^A) - n should be 0 for DAG
    n = A_soft.shape[0]
    return torch.trace(torch.matrix_exp(A_soft)) - n
```

### 5.3 Total Loss

```python
L_total = L_task + λ * L_concept + β * L_prior + γ * L_sparse

# Hyperparameters:
# λ: concept loss weight (from CREAM)
# β: prior consistency weight
# γ: sparsity weight
```

---

## Part 6: Config Structure

### 6.1 New YAML Fields

```yaml
mode: train_cbm
dataset_name: Concept_FMNIST

# NEW: Multi-expert settings
multi_expert:
  enabled: true
  num_experts: 5
  disagreement_level: medium  # low, medium, high
  expert_type: mixed          # conservative, liberal, balanced, mixed
  
  reliability_learning:
    type: edge                # edge, graph, combined
    
  graph_regularization:
    prior_weight: 0.1         # β
    sparsity_weight: 0.01     # γ
    acyclicity_weight: 0.0    # optional

paths:
  DAG_file: ./data/FashionMNIST/Concept_FMNIST_DAG.csv  # Ground truth
  expert_graphs_dir: ./data/FashionMNIST/expert_graphs/medium_disagreement/

# ... rest of CREAM config
```

---

## Part 7: Experiment Plan

### 7.1 Baselines

| Experiment | Description |
|------------|-------------|
| Vanilla CBM | No graph structure |
| CREAM (single graph) | Use ground truth G* |
| CREAM (noisy single) | Use one corrupted expert |
| Union baseline | A_union = OR of all experts |
| Intersection baseline | A_inter = AND of all experts |
| Majority vote | Edges with >50% agreement |

### 7.2 mCREAM Variants

| Experiment | Description |
|------------|-------------|
| mCREAM (edge-level) | Learn α per edge |
| mCREAM (graph-level) | Learn π per expert |
| mCREAM (combined) | Both α and π |

### 7.3 Ablations

| Ablation | What to Test |
|----------|--------------|
| Disagreement level | Low vs Medium vs High |
| Number of experts | M = 3, 5, 10 |
| Expert types | Conservative vs Liberal vs Mixed |
| Regularization | With/without prior, sparsity |
| Initialization | Vote-init vs random |

### 7.4 Evaluation Metrics

| Metric | What it Measures |
|--------|------------------|
| Task Accuracy | Main performance |
| Concept Accuracy | Concept prediction |
| CCI | Concept contribution |
| **Graph Recovery** | How close learned Ã is to G* |
| **Edge Precision/Recall** | Quality of learned edges |
| **Expert Weight Distribution** | Which experts trusted |

---

## Part 8: Implementation Checklist

### Implemented Files

```
mCREAM/
├── src/
│   ├── expert_graphs/                    # NEW: Expert graph module
│   │   ├── __init__.py                   # Module exports
│   │   ├── generation.py                 # Graph generation & corruption
│   │   ├── aggregation.py                # Aggregation methods (Union, Edge, Graph, Combined)
│   │   └── losses.py                     # Graph regularization losses
│   │
│   └── mcream_model.py                   # NEW: mCREAM model class
│
├── all_configs/
│   └── mcream_configs/                   # NEW: mCREAM experiment configs
│       └── mCREAM_cfmnist_edge_medium.yaml
│
├── generate_expert_graphs.py             # NEW: Script to generate expert graphs
├── mcream_main.py                        # NEW: mCREAM training script
│
└── data/
    └── FashionMNIST/
        └── expert_graphs/                # Generated expert graphs
            └── medium_disagreement/
                ├── config.yaml
                ├── u2c/
                │   ├── expert_0.pt
                │   └── ...
                ├── c2y/
                │   ├── expert_0.pt
                │   └── ...
                └── ground_truth/
                    ├── u2c_star.pt
                    └── c2y_star.pt
```

### Phase 1: Data Generation ✅
- [x] Create `generate_expert_graphs.py` script
- [x] Implement `generation.py` with corruption functions
- [x] Support disagreement levels (low/medium/high)
- [x] Support structured bias types (conservative/liberal)
- [x] Create storage structure

### Phase 2: Baseline Models ✅
- [x] Implement UnionAggregation
- [x] Implement IntersectionAggregation
- [x] Implement MajorityVoteAggregation
- [x] Implement AverageAggregation

### Phase 3: mCREAM Core ✅
- [x] Implement `EdgeReliabilityModule`
- [x] Implement `GraphAttentionModule`
- [x] Implement `CombinedReliabilityModule`
- [x] Implement `SoftMaskedLinear` for soft masks
- [x] Implement factory function `create_aggregation_module()`

### Phase 4: Model Integration ✅
- [x] Create `mCREAM_UtoC_Y` class
- [x] Integrate graph aggregation into forward pass
- [x] Add graph regularization losses
- [x] Create `mcream_main.py` training script

### Phase 5: Evaluation ✅
- [x] Add graph recovery metrics (precision/recall/F1)
- [x] Add expert weight logging
- [x] Integrate with existing CSV output

### Phase 6: Experiments (TODO)
- [ ] Run all baselines
- [ ] Run mCREAM variants
- [ ] Run ablations
- [ ] Aggregate and analyze results

---

## Part 9: Visual Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              mCREAM ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  INPUT                                                                          │
│  ┌─────────┐    ┌──────────────────────────────────────────────────────────┐   │
│  │  Image  │    │  Expert Graphs: G^(1), G^(2), ..., G^(M)                 │   │
│  │   (x)   │    │  (loaded from expert_graphs_dir)                         │   │
│  └────┬────┘    └──────────────────────────┬───────────────────────────────┘   │
│       │                                     │                                   │
│       ▼                                     ▼                                   │
│  ┌─────────┐                    ┌───────────────────────────────┐              │
│  │Backbone │                    │   GRAPH AGGREGATION MODULE    │              │
│  │(frozen) │                    │                               │              │
│  └────┬────┘                    │  ┌─────────────────────────┐  │              │
│       │                         │  │  Option A: Edge-level   │  │              │
│       ▼                         │  │  α_ij = learnable       │  │              │
│  ┌─────────┐                    │  │  Ã = sigmoid(α)         │  │              │
│  │Embedding│                    │  └─────────────────────────┘  │              │
│  │  (u)    │                    │            OR                 │              │
│  └────┬────┘                    │  ┌─────────────────────────┐  │              │
│       │                         │  │  Option B: Graph-level  │  │              │
│       ▼                         │  │  π_m = learnable        │  │              │
│  ┌─────────┐                    │  │  Ã = Σ π_m * A^(m)      │  │              │
│  │Splitter │                    │  └─────────────────────────┘  │              │
│  │ u→Uc,Uy │                    │                               │              │
│  └────┬────┘                    └───────────────┬───────────────┘              │
│       │                                         │                               │
│       ├──────────────┐                          │                               │
│       │              │                          │                               │
│       ▼              ▼                          ▼                               │
│  ┌─────────┐    ┌─────────┐              ┌───────────┐                         │
│  │   Uc    │    │   Uy    │              │  Ã_C, Ã_Y │                         │
│  └────┬────┘    └────┬────┘              │(soft masks)│                         │
│       │              │                    └─────┬─────┘                         │
│       │              │                          │                               │
│       ▼              │                          │                               │
│  ┌──────────────┐    │                          │                               │
│  │ Concept Block│◄───┼──────────────────────────┤                               │
│  │ (with Ã_C)   │    │                          │                               │
│  └──────┬───────┘    │                          │                               │
│         │            │                          │                               │
│         ▼            ▼                          │                               │
│  ┌──────────┐   ┌──────────┐                    │                               │
│  │Concepts ĉ│   │Side-chan │                    │                               │
│  │  (K-dim) │   │ s (S-dim)│                    │                               │
│  └────┬─────┘   └────┬─────┘                    │                               │
│       │              │                          │                               │
│       └──────┬───────┘                          │                               │
│              │                                  │                               │
│              ▼                                  │                               │
│       ┌─────────────┐                           │                               │
│       │ Task Block  │◄──────────────────────────┘                               │
│       │ (with Ã_Y)  │                                                           │
│       └──────┬──────┘                                                           │
│              │                                                                  │
│              ▼                                                                  │
│       ┌─────────────┐                                                           │
│       │   Task ŷ    │                                                           │
│       └─────────────┘                                                           │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  LOSS                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  L = L_task + λ*L_concept + β*L_prior + γ*L_sparse                      │   │
│  │      ├────────────────────┤ ├─────────────────────────┤                 │   │
│  │      Original CREAM loss    Graph regularization                        │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  LEARNED PARAMETERS                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  • Concept network weights                                               │   │
│  │  • Task network weights                                                  │   │
│  │  • Splitter weights                                                      │   │
│  │  • α (edge reliabilities) ← NEW                                         │   │
│  │  • π (expert weights) ← NEW                                             │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 10: Key Design Decisions Summary

| Decision | Recommendation | Reason |
|----------|----------------|--------|
| **Start with** | Edge-level (α) | More flexible, precise |
| **Initialize α** | Expert voting | Better than random |
| **Regularization** | Prior + Sparsity | Prevents overfitting |
| **First dataset** | FashionMNIST | Fast iteration |
| **Number of experts** | 5 | Good balance |
| **First disagreement** | Medium | Not too easy/hard |
