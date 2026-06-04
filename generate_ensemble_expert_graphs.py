"""
Generate expert graphs for mCREAM Ensemble experiments.

Single-action noise: all M experts share ONE action type at ONE noise level.
Run this ONCE before submitting any ensemble experiment jobs.

Usage:
    python generate_ensemble_expert_graphs.py --dataset cfmnist
    python generate_ensemble_expert_graphs.py --dataset celeba
    python generate_ensemble_expert_graphs.py --dataset all
"""

import argparse
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.expert_graphs.generation import (
    generate_single_action_experts,
    load_and_split_dag,
    save_expert_graphs,
    SINGLE_ACTION_NOISE_LEVELS,
)

DATASETS = {
    "cfmnist": {
        "dag_path": "data/FashionMNIST/Complete_Concept_FMNIST_DAG.csv",
        "num_classes": 10,
        "num_experts": 5,
        "output_base": "data/FashionMNIST/expert_graphs/ensemble",
    },
    "celeba": {
        "dag_path": "data/CelebA/final_DAG_unfair.csv",
        "num_classes": 1,
        "num_experts": 5,
        "output_base": "data/CelebA/expert_graphs/ensemble",
    },
}


def generate_for_dataset(dataset_name: str, num_experts: int = None):
    cfg = DATASETS[dataset_name]
    dag_path = cfg["dag_path"]
    num_classes = cfg["num_classes"]
    M = num_experts or cfg["num_experts"]
    output_base = cfg["output_base"]

    print(f"\n{'='*55}")
    print(f"Dataset: {dataset_name}  |  M={M}  |  DAG: {dag_path}")
    print(f"{'='*55}")

    actions = list(SINGLE_ACTION_NOISE_LEVELS.keys())   # deletion, addition, reversal
    levels  = ["low", "medium", "high"]

    for action in actions:
        for level in levels:
            output_dir = Path(output_base) / f"{action}_{level}"

            # Skip if already generated
            if (output_dir / "config.yaml").exists():
                print(f"  [SKIP] {action}/{level} — already exists at {output_dir}")
                continue

            print(f"  Generating {action}/{level} (M={M})...")

            expert_u2c, expert_c2y, u2c_star, c2y_star = generate_single_action_experts(
                dag_path=dag_path,
                num_classes=num_classes,
                num_experts=M,
                action=action,
                noise_level=level,
                base_seed=42,
            )

            params = SINGLE_ACTION_NOISE_LEVELS[action][level]
            save_config = {
                "dag_path": dag_path,
                "num_classes": num_classes,
                "num_experts": M,
                "noise_type": "single_action",
                "action": action,
                "noise_level": level,
                **params,
                "seed": 42,
            }

            # Save expert graphs
            save_expert_graphs(expert_u2c, expert_c2y, output_dir, save_config)

            # Save ground truth alongside for easy comparison
            gt_dir = output_dir / "ground_truth"
            gt_dir.mkdir(exist_ok=True)
            torch.save(u2c_star, gt_dir / "u2c_star.pt")
            torch.save(c2y_star, gt_dir / "c2y_star.pt")

            # Print corruption stats
            for m, (u2c, c2y) in enumerate(zip(expert_u2c, expert_c2y)):
                pct_u2c = (u2c.bool() != u2c_star.bool()).sum().item() / u2c_star.numel() * 100
                pct_c2y = (c2y.bool() != c2y_star.bool()).sum().item() / c2y_star.numel() * 100
                print(f"    expert_{m}: u2c {pct_u2c:.1f}% changed, c2y {pct_c2y:.1f}% changed")

            print(f"    Saved to: {output_dir}")

    print(f"\nDone: {dataset_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cfmnist", "celeba", "all"], default="all")
    parser.add_argument("--num_experts", type=int, default=None,
                        help="Override default M (default: 5 for both datasets)")
    args = parser.parse_args()

    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        generate_for_dataset(ds, args.num_experts)

    print("\n=== All expert graphs generated ===")


if __name__ == "__main__":
    main()
