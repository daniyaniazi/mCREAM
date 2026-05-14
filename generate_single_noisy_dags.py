"""
Generate single noisy DAG CSV files from expert_0 of M=5 expert graphs.
This converts the .pt tensor format back to CSV for use with standard CREAM.

The generated CSV matches the ground-truth DAG format exactly (same node names,
same matrix layout) so that CREAM can load it without issues.

Usage:
    python generate_single_noisy_dags.py --dataset cfmnist --level medium
    python generate_single_noisy_dags.py --dataset celeba --level high
"""

import argparse
import os
import torch
import pandas as pd
import numpy as np


# Dataset configurations
# ground_truth_dag is used to read the real node names and concept/class split
DATASET_CONFIGS = {
    "cfmnist": {
        "expert_graphs_base": "data/FashionMNIST/expert_graphs/M5",
        "output_dir": "data/FashionMNIST/noisy_dags",
        "ground_truth_dag": "data/FashionMNIST/Complete_Concept_FMNIST_DAG.csv",
        "num_concepts": 11,   # first 11 nodes are concepts
        "num_classes": 10,    # last 10 nodes are classes
    },
    "celeba": {
        "expert_graphs_base": "data/CelebA/expert_graphs/M5",
        "output_dir": "data/CelebA/noisy_dags",
        "ground_truth_dag": "data/CelebA/final_DAG_unfair.csv",
        "num_concepts": 7,    # first 7 nodes are concepts
        "num_classes": 1,     # last node is class (Smiling)
    },
}


def load_ground_truth_dag(path: str):
    """Load the ground-truth DAG to get the node names and structure."""
    df = pd.read_csv(path, index_col=0)
    return df


def load_expert_tensors(expert_graphs_base: str, level: str, expert_idx: int = 0):
    """Load u2c and c2y tensors for a specific expert."""
    u2c_path = os.path.join(expert_graphs_base, level, "u2c", f"expert_{expert_idx}.pt")
    c2y_path = os.path.join(expert_graphs_base, level, "c2y", f"expert_{expert_idx}.pt")
    
    if not os.path.exists(u2c_path):
        raise FileNotFoundError(f"Expert u2c tensor not found: {u2c_path}")
    if not os.path.exists(c2y_path):
        raise FileNotFoundError(f"Expert c2y tensor not found: {c2y_path}")
    
    u2c = torch.load(u2c_path, map_location="cpu", weights_only=True)
    c2y = torch.load(c2y_path, map_location="cpu", weights_only=True)
    
    return u2c, c2y


def tensors_to_dag_csv(u2c: torch.Tensor, c2y: torch.Tensor,
                       gt_dag: pd.DataFrame, num_concepts: int,
                       num_classes: int) -> pd.DataFrame:
    """
    Convert u2c and c2y tensors back to full DAG adjacency matrix CSV format,
    using the ground-truth DAG as a template for node names and concept-concept
    edges.

    Strategy:
      - Start from the ground-truth DAG (preserves concept-concept hierarchy).
      - Overwrite only the concept->class block with the noisy expert's c2y.
    """
    node_names = list(gt_dag.index)
    total_nodes = len(node_names)

    # Start from ground-truth adjacency (keeps concept-concept edges intact)
    adj = gt_dag.values.copy().astype(object)
    # Convert True/False to int if needed
    if adj.dtype == object or adj.dtype == bool:
        adj = np.where(adj == True, 1, np.where(adj == False, 0, adj))  # noqa: E712
    adj = adj.astype(int)

    # Overwrite u2c block (concept-concept, top-left K×K) with noisy expert
    u2c_np = u2c.numpy() if isinstance(u2c, torch.Tensor) else u2c
    # u2c shape: [K × K]  (from load_and_split_dag: full_dag[:-T, :-T])
    for r in range(min(u2c_np.shape[0], num_concepts)):
        for c in range(min(u2c_np.shape[1], num_concepts)):
            adj[r, c] = 1 if u2c_np[r, c] > 0.5 else 0

    # Overwrite c2y block (last T rows, full width) with noisy expert
    # c2y shape: [T × (K+T)]  (from load_and_split_dag: full_dag[-T:, :])
    c2y_np = c2y.numpy() if isinstance(c2y, torch.Tensor) else c2y
    for y_idx in range(min(c2y_np.shape[0], num_classes)):
        for col in range(min(c2y_np.shape[1], total_nodes)):
            adj[num_concepts + y_idx, col] = 1 if c2y_np[y_idx, col] > 0.5 else 0

    # Convert back to True/False to match ground-truth format
    bool_adj = adj.astype(bool)
    df = pd.DataFrame(bool_adj, index=node_names, columns=node_names)

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate noisy DAG CSV from expert_0")
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=["cfmnist", "celeba"],
                        help="Dataset name")
    parser.add_argument("--level", type=str, required=True,
                        choices=["low", "medium", "high", "structured_bias"],
                        help="Disagreement level")
    parser.add_argument("--expert_idx", type=int, default=0,
                        help="Which expert to use (default: 0)")
    args = parser.parse_args()
    
    config = DATASET_CONFIGS[args.dataset]
    num_concepts = config["num_concepts"]
    num_classes = config["num_classes"]

    # Load ground-truth DAG for node names and concept-concept structure
    gt_dag = load_ground_truth_dag(config["ground_truth_dag"])
    print(f"Ground-truth DAG: {gt_dag.shape}  nodes: {list(gt_dag.index)}")

    print(f"Loading expert_{args.expert_idx} from {args.dataset}/{args.level}...")
    
    # Load tensors
    u2c, c2y = load_expert_tensors(
        config["expert_graphs_base"], 
        args.level, 
        args.expert_idx
    )
    
    print(f"  u2c shape: {u2c.shape}")
    print(f"  c2y shape: {c2y.shape}")
    
    # Convert to DAG CSV
    dag_df = tensors_to_dag_csv(u2c, c2y, gt_dag, num_concepts, num_classes)
    
    # Save
    os.makedirs(config["output_dir"], exist_ok=True)
    output_path = os.path.join(config["output_dir"], f"noisy_dag_{args.level}.csv")
    dag_df.to_csv(output_path)
    
    print(f"Saved noisy DAG to: {output_path}")
    print(f"  Shape: {dag_df.shape}")
    print(f"  Total edges (True): {dag_df.values.sum()}")
    
    # Show diff from ground truth
    gt_bool = gt_dag.values.astype(bool) if gt_dag.values.dtype != bool else gt_dag.values
    noisy_bool = dag_df.values
    diff_count = (gt_bool != noisy_bool).sum()
    print(f"  Edges changed vs ground truth: {diff_count}")


if __name__ == "__main__":
    main()
