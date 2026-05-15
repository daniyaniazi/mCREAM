"""
Expert Graph Generation Module for mCREAM.

This module handles:
1. Loading and splitting DAG files (CREAM-style)
2. Generating corrupted expert graphs
3. Saving/loading expert graph sets
"""

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import yaml
import os


# =============================================================================
# DAG Loading and Splitting (CREAM-compatible)
# =============================================================================

def load_and_split_dag(
    dag_path: str | Path,
    num_classes: int,
) -> Tuple[Tensor, Tensor]:
    """
    Load DAG CSV and split into u2c and c2y graphs exactly like CREAM does.
    
    CREAM splits the full DAG as follows:
    - u2c_graph: concept→concept relationships (for MaskedMLP in concept prediction)
    - c2y_graph: task rows with all columns (for MaskedLinear in task prediction)
    
    Args:
        dag_path: Path to DAG CSV file (e.g., Complete_Concept_FMNIST_DAG.csv)
        num_classes: Number of task classes (e.g., 10 for FMNIST)
    
    Returns:
        u2c_graph: Concept-to-concept adjacency [K x K] where K = num_concepts
        c2y_graph: Task adjacency [T x (K+T)] where T = num_classes
    """
    import numpy as np
    
    df = pd.read_csv(dag_path, index_col=0)
    
    # Convert to boolean tensor - handle various formats
    values = df.values
    
    # Check if already boolean (handles numpy.bool_)
    if np.issubdtype(values.dtype, np.bool_):
        bool_values = values
    elif values.dtype == object:
        # String values - could be 'True'/'False' or 'true'/'false'
        bool_values = (values == True) | (values == 'True') | (values == 'true') | (values == '1')
    else:
        # Numeric - treat non-zero as True
        bool_values = values != 0
    
    full_graph = torch.tensor(bool_values, dtype=torch.bool)
    
    # CREAM's exact splitting logic (from src/models.py lines 816-817):
    # self.u2c_graph = self.causal_graph[:-self.num_classes, :-self.num_classes]
    # self.c2y_graph = self.causal_graph[-self.num_classes:, :]
    
    u2c_graph = full_graph[:-num_classes, :-num_classes]  # [K x K]
    c2y_graph = full_graph[-num_classes:, :]              # [T x (K+T)]
    
    return u2c_graph.float(), c2y_graph.float()


def get_node_names(dag_path: str | Path, num_classes: int) -> Tuple[List[str], List[str]]:
    """
    Extract concept and task names from DAG CSV.
    
    Returns:
        concept_names: List of concept names
        task_names: List of task/class names
    """
    df = pd.read_csv(dag_path, index_col=0)
    all_names = list(df.columns)
    
    concept_names = all_names[:-num_classes]
    task_names = all_names[-num_classes:]
    
    return concept_names, task_names


# =============================================================================
# Expert Graph Corruption
# =============================================================================

def generate_expert_graph(
    G_star: Tensor,
    p_del: float = 0.1,
    p_add: float = 0.1,
    p_rev: float = 0.05,
    seed: Optional[int] = None,
    preserve_diagonal: bool = True,
) -> Tensor:
    """
    Generate one expert graph by corrupting ground truth.
    
    Args:
        G_star: Ground truth adjacency matrix [n_rows, n_cols]
        p_del: Probability of deleting existing edge
        p_add: Probability of adding non-existing edge
        p_rev: Probability of reversing edge direction (only for square matrices)
        seed: Random seed for reproducibility
        preserve_diagonal: If True, don't modify diagonal entries (self-loops)
    
    Returns:
        G_expert: Corrupted expert graph with same shape as G_star
    """
    if seed is not None:
        np.random.seed(seed)
    
    G_expert = G_star.clone()
    n_rows, n_cols = G_star.shape
    is_square = (n_rows == n_cols)
    
    for i in range(n_rows):
        for j in range(n_cols):
            # Skip diagonal if preserving
            if preserve_diagonal and i == j and is_square:
                continue
            
            if G_star[i, j] == 1:  # Edge exists
                rand_val = np.random.random()
                if rand_val < p_del:
                    # Deletion: remove edge
                    G_expert[i, j] = 0
                elif is_square and rand_val < p_del + p_rev:
                    # Reversal: flip direction (only for square matrices)
                    G_expert[i, j] = 0
                    G_expert[j, i] = 1
            else:  # Edge doesn't exist
                if np.random.random() < p_add:
                    # Addition: add spurious edge
                    G_expert[i, j] = 1
    
    return G_expert


def generate_expert_graphs_from_dag(
    dag_path: str | Path,
    num_classes: int,
    num_experts: int,
    p_del: float = 0.15,
    p_add: float = 0.15,
    p_rev: float = 0.08,
    base_seed: int = 42,
) -> Tuple[List[Tensor], List[Tensor], Tensor, Tensor]:
    """
    Generate M expert graphs for both u2c and c2y components.
    
    Args:
        dag_path: Path to ground truth DAG CSV
        num_classes: Number of task classes
        num_experts: Number of expert graphs to generate
        p_del: Deletion probability
        p_add: Addition probability
        p_rev: Reversal probability
        base_seed: Base random seed (each expert uses base_seed + m)
    
    Returns:
        expert_u2c_graphs: List of M concept→concept graphs [K x K]
        expert_c2y_graphs: List of M task graphs [T x (K+T)]
        u2c_star: Ground truth u2c graph
        c2y_star: Ground truth c2y graph
    """
    # Load and split ground truth
    u2c_star, c2y_star = load_and_split_dag(dag_path, num_classes)
    
    expert_u2c_graphs = []
    expert_c2y_graphs = []
    
    for m in range(num_experts):
        # Generate corrupted u2c graph
        u2c_expert = generate_expert_graph(
            u2c_star, p_del, p_add, p_rev, 
            seed=base_seed + m
        )
        expert_u2c_graphs.append(u2c_expert)
        
        # Generate corrupted c2y graph (different seed offset)
        c2y_expert = generate_expert_graph(
            c2y_star, p_del, p_add, p_rev,
            seed=base_seed + m + 10000  # Offset to ensure different randomness
        )
        expert_c2y_graphs.append(c2y_expert)
    
    return expert_u2c_graphs, expert_c2y_graphs, u2c_star, c2y_star


# =============================================================================
# Structured Expert Bias
# =============================================================================

DISAGREEMENT_LEVELS = {
    "low": {"p_del": 0.25, "p_add": 0.25, "p_rev": 0.10},
    "medium": {"p_del": 0.55, "p_add": 0.55, "p_rev": 0.25},
    "high": {"p_del": 0.85, "p_add": 0.85, "p_rev": 0.40},
}

EXPERT_BIAS_TYPES = {
    "conservative": {"p_del": 0.60, "p_add": 0.10, "p_rev": 0.10},
    "liberal": {"p_del": 0.10, "p_add": 0.60, "p_rev": 0.10},
    "balanced": {"p_del": 0.35, "p_add": 0.35, "p_rev": 0.15},
}


def generate_structured_experts(
    dag_path: str | Path,
    num_classes: int,
    expert_types: List[str],
    base_seed: int = 42,
) -> Tuple[List[Tensor], List[Tensor], Tensor, Tensor]:
    """
    Generate experts with specific bias types.
    
    Args:
        dag_path: Path to ground truth DAG
        num_classes: Number of task classes
        expert_types: List of expert types, e.g., ["conservative", "liberal", "balanced"]
        base_seed: Base random seed
    
    Returns:
        expert_u2c_graphs, expert_c2y_graphs, u2c_star, c2y_star
    """
    u2c_star, c2y_star = load_and_split_dag(dag_path, num_classes)
    
    expert_u2c_graphs = []
    expert_c2y_graphs = []
    
    for m, expert_type in enumerate(expert_types):
        if expert_type not in EXPERT_BIAS_TYPES:
            raise ValueError(f"Unknown expert type: {expert_type}. "
                           f"Choose from {list(EXPERT_BIAS_TYPES.keys())}")
        
        params = EXPERT_BIAS_TYPES[expert_type]
        
        u2c_expert = generate_expert_graph(
            u2c_star, 
            params["p_del"], params["p_add"], params["p_rev"],
            seed=base_seed + m
        )
        expert_u2c_graphs.append(u2c_expert)
        
        c2y_expert = generate_expert_graph(
            c2y_star,
            params["p_del"], params["p_add"], params["p_rev"],
            seed=base_seed + m + 10000
        )
        expert_c2y_graphs.append(c2y_expert)
    
    return expert_u2c_graphs, expert_c2y_graphs, u2c_star, c2y_star


# =============================================================================
# Save / Load Expert Graphs
# =============================================================================

def save_expert_graphs(
    expert_u2c_graphs: List[Tensor],
    expert_c2y_graphs: List[Tensor],
    output_dir: str | Path,
    config: Dict[str, Any],
    concept_names: Optional[List[str]] = None,
    task_names: Optional[List[str]] = None,
) -> None:
    """
    Save expert graphs and generation config to disk.
    
    Directory structure:
        output_dir/
            config.yaml
            u2c/
                expert_0.pt
                expert_1.pt
                ...
            c2y/
                expert_0.pt
                expert_1.pt
                ...
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Save u2c graphs
    u2c_dir = output_dir / "u2c"
    u2c_dir.mkdir(exist_ok=True)
    for i, graph in enumerate(expert_u2c_graphs):
        torch.save(graph, u2c_dir / f"expert_{i}.pt")
    
    # Save c2y graphs
    c2y_dir = output_dir / "c2y"
    c2y_dir.mkdir(exist_ok=True)
    for i, graph in enumerate(expert_c2y_graphs):
        torch.save(graph, c2y_dir / f"expert_{i}.pt")
    
    # Optionally save node names
    if concept_names or task_names:
        names = {"concept_names": concept_names, "task_names": task_names}
        with open(output_dir / "node_names.yaml", "w") as f:
            yaml.dump(names, f)
    
    print(f"Saved {len(expert_u2c_graphs)} expert graphs to {output_dir}")


def load_expert_graphs(
    input_dir: str | Path,
) -> Tuple[List[Tensor], List[Tensor], Dict[str, Any]]:
    """
    Load expert graphs from disk.
    
    Args:
        input_dir: Directory containing saved expert graphs
    
    Returns:
        expert_u2c_graphs: List of u2c graphs
        expert_c2y_graphs: List of c2y graphs
        config: Generation configuration
    """
    input_dir = Path(input_dir)
    
    # Load config
    with open(input_dir / "config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Load u2c graphs
    u2c_dir = input_dir / "u2c"
    expert_u2c_graphs = []
    for pt_file in sorted(u2c_dir.glob("expert_*.pt")):
        expert_u2c_graphs.append(torch.load(pt_file, weights_only=True))
    
    # Load c2y graphs
    c2y_dir = input_dir / "c2y"
    expert_c2y_graphs = []
    for pt_file in sorted(c2y_dir.glob("expert_*.pt")):
        expert_c2y_graphs.append(torch.load(pt_file, weights_only=True))
    
    return expert_u2c_graphs, expert_c2y_graphs, config


# =============================================================================
# Utility Functions
# =============================================================================

def compute_edge_statistics(
    expert_graphs: List[Tensor],
    ground_truth: Tensor,
) -> Dict[str, float]:
    """
    Compute statistics about expert graphs vs ground truth.
    
    Returns:
        Dictionary with precision, recall, agreement metrics
    """
    stacked = torch.stack(expert_graphs)
    M = len(expert_graphs)
    
    # Vote score (agreement)
    vote_score = stacked.mean(dim=0)
    
    # Binarize at 0.5 threshold
    majority_vote = (vote_score > 0.5).float()
    
    # Compare to ground truth
    gt = ground_truth.bool()
    mv = majority_vote.bool()
    
    true_positives = (gt & mv).sum().item()
    false_positives = (~gt & mv).sum().item()
    false_negatives = (gt & ~mv).sum().item()
    true_negatives = (~gt & ~mv).sum().item()
    
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # Expert agreement (how much experts agree with each other)
    agreement = vote_score[vote_score > 0].mean().item() if (vote_score > 0).any() else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "expert_agreement": agreement,
        "num_experts": M,
        "num_edges_gt": gt.sum().item(),
        "num_edges_majority": mv.sum().item(),
    }
