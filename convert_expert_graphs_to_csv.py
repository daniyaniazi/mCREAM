"""
Convert ensemble expert graphs (.pt files) to DAG CSV files
so that simple_main.py (standalone CREAM) can use them.

This allows running CREAM on the SAME expert graphs used in the ensemble,
giving a fair comparison:
  CREAM (one expert graph, fully trained)  vs  mCREAM ensemble (M expert graphs)

Usage:
    python generate_cream_noisy_dags.py --dataset cfmnist
    python generate_cream_noisy_dags.py --dataset celeba
    python generate_cream_noisy_dags.py --dataset all

Output structure:
    data/FashionMNIST/expert_graphs/ensemble/deletion_medium/
        cream_noisy_dags/
            expert_0.csv   ← full (K+T)x(K+T) DAG, same format as ground truth
            expert_1.csv
            ...
            expert_4.csv
"""

import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.expert_graphs.generation import SINGLE_ACTION_NOISE_LEVELS

DATASETS = {
    "cfmnist": {
        "dag_path": "data/FashionMNIST/Complete_Concept_FMNIST_DAG.csv",
        "num_classes": 10,
        "graphs_base": "data/FashionMNIST/expert_graphs/ensemble",
        "num_experts": 5,
    },
    "celeba": {
        "dag_path": "data/CelebA/final_DAG_unfair.csv",
        "num_classes": 1,
        "graphs_base": "data/CelebA/expert_graphs/ensemble",
        "num_experts": 5,
    },
}


def convert_expert_graphs_to_csv(dataset_key: str):
    cfg = DATASETS[dataset_key]
    dag_path = Path(cfg["dag_path"])
    graphs_base = Path(cfg["graphs_base"])
    num_classes = cfg["num_classes"]
    num_experts = cfg["num_experts"]

    # Load ground truth DAG to get node names
    gt_df = pd.read_csv(dag_path, index_col=0)
    node_names = list(gt_df.index)
    K = len(node_names) - num_classes   # num concepts
    T = num_classes

    print(f"\n{'='*55}")
    print(f"Dataset: {dataset_key}  |  K={K} concepts, T={T} tasks")
    print(f"{'='*55}")

    actions = list(SINGLE_ACTION_NOISE_LEVELS.keys())
    levels  = ["low", "medium", "high"]

    for action in actions:
        for level in levels:
            graph_dir = graphs_base / f"{action}_{level}"
            if not graph_dir.exists():
                print(f"  [SKIP] {graph_dir} not found")
                continue

            out_dir = graph_dir / "cream_noisy_dags"
            out_dir.mkdir(exist_ok=True)

            for m in range(num_experts):
                u2c_path = graph_dir / "u2c" / f"expert_{m}.pt"
                c2y_path = graph_dir / "c2y" / f"expert_{m}.pt"

                if not u2c_path.exists():
                    print(f"  [SKIP] {u2c_path} not found")
                    continue

                u2c = torch.load(u2c_path, weights_only=True).float().numpy()  # [K, K]
                c2y = torch.load(c2y_path, weights_only=True).float().numpy()  # [T, K+T]

                # Reconstruct full (K+T)x(K+T) DAG
                full = np.zeros((K + T, K + T), dtype=bool)
                full[:K, :K] = u2c.astype(bool)   # concept→concept block
                full[K:, :]  = c2y.astype(bool)   # task rows

                # Save as CSV with same node names as ground truth
                df = pd.DataFrame(full, index=node_names, columns=node_names)
                out_path = out_dir / f"expert_{m}.csv"
                df.to_csv(out_path)

            print(f"  {action}/{level}: saved {num_experts} DAG CSVs to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cfmnist", "celeba", "all"], default="all")
    args = parser.parse_args()

    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        convert_expert_graphs_to_csv(ds)

    print("\n=== DAG CSV conversion complete ===")


if __name__ == "__main__":
    main()
