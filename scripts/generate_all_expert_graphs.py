"""
Generate expert graphs for all experimental conditions.

This script generates corrupted expert graphs for:
- Multiple expert counts: M = 1, 2, 5, 10
- Multiple noise levels: low, medium, high
- All datasets: cfmnist, cub, celeba

Usage:
    python scripts/generate_all_expert_graphs.py --dataset cfmnist
    python scripts/generate_all_expert_graphs.py --all
"""

import argparse
import torch
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.expert_graphs.generation import (
    generate_expert_graphs_from_dag,
    save_expert_graphs,
    DISAGREEMENT_LEVELS,
)


# Dataset configurations
DATASETS = {
    "cfmnist": {
        "dag_path": "data/FashionMNIST/Complete_Concept_FMNIST_DAG.csv",
        "num_classes": 10,
        "output_base": "data/FashionMNIST/expert_graphs",
    },
    "cub": {
        "dag_path": "data/CUB/CUB_DAG_only_Gc.csv",
        "num_classes": 200,
        "output_base": "data/CUB/expert_graphs",
    },
    "celeba": {
        "dag_path": "data/CelebA/final_DAG_unfair.csv",
        "num_classes": 2,
        "output_base": "data/CelebA/expert_graphs",
    },
}

# Expert counts to generate
EXPERT_COUNTS = [1, 2, 5, 10]

# Base seed for reproducibility
BASE_SEED = 42


def generate_for_dataset(dataset_key: str):
    """Generate all expert graph combinations for a dataset."""
    
    ds = DATASETS[dataset_key]
    dag_path = Path(ds["dag_path"])
    
    if not dag_path.exists():
        print(f"WARNING: DAG file not found: {dag_path}")
        print(f"  Skipping {dataset_key}")
        return
    
    output_base = Path(ds["output_base"])
    
    print(f"\n{'='*60}")
    print(f"Generating expert graphs for: {dataset_key}")
    print(f"DAG: {dag_path}")
    print(f"{'='*60}")
    
    for m in EXPERT_COUNTS:
        for noise_level, params in DISAGREEMENT_LEVELS.items():
            # Skip M=1 with non-medium noise (not meaningful)
            if m == 1 and noise_level != "medium":
                continue
            
            # For M<=2, only generate medium noise
            if m <= 2 and noise_level != "medium":
                continue
            
            output_dir = output_base / f"M{m}" / noise_level
            
            # Skip if already exists
            if (output_dir / "config.yaml").exists():
                print(f"  [SKIP] Already exists: M={m}, noise={noise_level}")
                continue
            
            print(f"\n  Generating: M={m}, noise={noise_level}")
            print(f"    p_del={params['p_del']}, p_add={params['p_add']}, p_rev={params['p_rev']}")
            
            # Generate expert graphs
            expert_u2c, expert_c2y, u2c_star, c2y_star = generate_expert_graphs_from_dag(
                dag_path=str(dag_path),
                num_classes=ds["num_classes"],
                num_experts=m,
                p_del=params["p_del"],
                p_add=params["p_add"],
                p_rev=params["p_rev"],
                base_seed=BASE_SEED,
            )
            
            # Save
            save_config = {
                "dag_path": str(dag_path),
                "num_classes": ds["num_classes"],
                "num_experts": m,
                "disagreement_level": noise_level,
                **params,
                "seed": BASE_SEED,
            }
            
            save_expert_graphs(expert_u2c, expert_c2y, output_dir, save_config)
            
            # Also save ground truth
            gt_dir = output_dir / "ground_truth"
            gt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(u2c_star, gt_dir / "u2c_star.pt")
            torch.save(c2y_star, gt_dir / "c2y_star.pt")
            
            print(f"    Saved to: {output_dir}")
            print(f"    u2c shape: {expert_u2c[0].shape}, c2y shape: {expert_c2y[0].shape}")


def main():
    parser = argparse.ArgumentParser(description="Generate expert graphs for mCREAM experiments")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASETS.keys()),
        help="Dataset to generate for"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate for all datasets"
    )
    args = parser.parse_args()
    
    if args.all:
        for dataset_key in DATASETS.keys():
            generate_for_dataset(dataset_key)
    elif args.dataset:
        generate_for_dataset(args.dataset)
    else:
        print("Please specify --dataset or --all")
        parser.print_help()
        return
    
    print("\n" + "=" * 60)
    print("Expert graph generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
