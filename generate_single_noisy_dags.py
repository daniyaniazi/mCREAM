"""
Generate single noisy DAG CSV files from expert_0 of M=5 expert graphs.
This converts the .pt tensor format back to CSV for use with standard CREAM.

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
DATASET_CONFIGS = {
    "cfmnist": {
        "expert_graphs_base": "data/FashionMNIST/expert_graphs/M5",
        "output_dir": "data/FashionMNIST/noisy_dags",
        "num_concepts": 12,
        "num_classes": 10,
        "concept_names": [f"c{i}" for i in range(12)],
        "class_names": [f"y{i}" for i in range(10)],
    },
    "celeba": {
        "expert_graphs_base": "data/CelebA/expert_graphs/M5",
        "output_dir": "data/CelebA/noisy_dags",
        "num_concepts": 39,
        "num_classes": 2,
        "concept_names": [f"c{i}" for i in range(39)],
        "class_names": ["y0", "y1"],
    },
}


def load_expert_tensors(expert_graphs_base: str, level: str, expert_idx: int = 0):
    """Load u2c and c2y tensors for a specific expert."""
    u2c_path = os.path.join(expert_graphs_base, level, "u2c", f"expert_{expert_idx}.pt")
    c2y_path = os.path.join(expert_graphs_base, level, "c2y", f"expert_{expert_idx}.pt")
    
    if not os.path.exists(u2c_path):
        raise FileNotFoundError(f"Expert u2c tensor not found: {u2c_path}")
    if not os.path.exists(c2y_path):
        raise FileNotFoundError(f"Expert c2y tensor not found: {c2y_path}")
    
    u2c = torch.load(u2c_path, map_location="cpu")
    c2y = torch.load(c2y_path, map_location="cpu")
    
    return u2c, c2y


def tensors_to_dag_csv(u2c: torch.Tensor, c2y: torch.Tensor, 
                       concept_names: list, class_names: list) -> pd.DataFrame:
    """
    Convert u2c and c2y tensors back to full DAG adjacency matrix CSV format.
    
    The DAG format expected by CREAM:
    - Rows and columns are [concepts..., classes...]
    - Entry (i,j) = 1 means edge from node i to node j
    - u2c has shape [num_concepts, 1] - edges from input U to concepts
    - c2y has shape [num_classes, num_concepts] - edges from concepts to classes
    
    For the CSV, we need concept-to-concept and concept-to-class edges.
    Since u2c represents which concepts are "active" (have edge from U),
    we'll focus on c2y which shows concept->class relationships.
    """
    num_concepts = len(concept_names)
    num_classes = len(class_names)
    total_nodes = num_concepts + num_classes
    
    # Create full adjacency matrix
    all_node_names = concept_names + class_names
    adj_matrix = np.zeros((total_nodes, total_nodes), dtype=int)
    
    # Fill in c2y edges (concept -> class)
    # c2y shape: [num_classes, num_concepts], entry (y, c) = 1 means c -> y
    c2y_np = c2y.numpy() if isinstance(c2y, torch.Tensor) else c2y
    
    for y_idx in range(num_classes):
        for c_idx in range(num_concepts):
            if c2y_np[y_idx, c_idx] > 0.5:  # Edge exists
                # In adjacency matrix: row=source, col=target
                # concept c_idx -> class y_idx
                adj_matrix[c_idx, num_concepts + y_idx] = 1
    
    # Create DataFrame
    df = pd.DataFrame(adj_matrix, index=all_node_names, columns=all_node_names)
    
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
    dag_df = tensors_to_dag_csv(
        u2c, c2y,
        config["concept_names"],
        config["class_names"]
    )
    
    # Save
    os.makedirs(config["output_dir"], exist_ok=True)
    output_path = os.path.join(config["output_dir"], f"noisy_dag_{args.level}.csv")
    dag_df.to_csv(output_path)
    
    print(f"Saved noisy DAG to: {output_path}")
    print(f"  Shape: {dag_df.shape}")
    print(f"  Total edges: {dag_df.values.sum()}")


if __name__ == "__main__":
    main()
