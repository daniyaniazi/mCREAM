"""
Script to generate expert graphs for mCREAM experiments.

Usage:
    python generate_expert_graphs.py --config path/to/config.yaml
    
Or with command line arguments:
    python generate_expert_graphs.py \
        --dag_path data/FashionMNIST/Complete_Concept_FMNIST_DAG.csv \
        --num_classes 10 \
        --num_experts 5 \
        --disagreement_level medium \
        --output_dir data/FashionMNIST/expert_graphs/medium_disagreement
"""

import argparse
from pathlib import Path
import yaml
import torch
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.expert_graphs.generation import (
    generate_expert_graphs_from_dag,
    generate_structured_experts,
    save_expert_graphs,
    load_and_split_dag,
    get_node_names,
    compute_edge_statistics,
    DISAGREEMENT_LEVELS,
    EXPERT_BIAS_TYPES,
)


def main():
    parser = argparse.ArgumentParser(description="Generate expert graphs for mCREAM")
    
    # Config file option
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    
    # Command line options (override config)
    parser.add_argument("--dag_path", type=str, help="Path to ground truth DAG CSV")
    parser.add_argument("--num_classes", type=int, help="Number of task classes")
    parser.add_argument("--num_experts", type=int, default=5, help="Number of experts")
    parser.add_argument("--disagreement_level", type=str, default="medium",
                       choices=["low", "medium", "high", "structured_bias"],
                       help="Disagreement level preset")
    parser.add_argument("--expert_types", type=str, nargs="+",
                       help="List of expert types (e.g., conservative liberal balanced)")
    parser.add_argument("--p_del", type=float, help="Custom deletion probability")
    parser.add_argument("--p_add", type=float, help="Custom addition probability")
    parser.add_argument("--p_rev", type=float, help="Custom reversal probability")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    
    args = parser.parse_args()
    
    # Load config if provided
    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    
    # Override with command line args
    dag_path = args.dag_path or config.get("dag_path")
    num_classes = args.num_classes or config.get("num_classes")
    num_experts = args.num_experts or config.get("num_experts", 5)
    disagreement_level = args.disagreement_level or config.get("disagreement_level", "medium")
    expert_types = args.expert_types or config.get("expert_types")
    seed = args.seed or config.get("seed", 42)
    output_dir = args.output_dir or config.get("output_dir")
    
    # Validate required args
    if not dag_path:
        raise ValueError("--dag_path is required")
    if not num_classes:
        raise ValueError("--num_classes is required")
    if not output_dir:
        raise ValueError("--output_dir is required")
    
    print(f"Generating expert graphs...")
    print(f"  DAG: {dag_path}")
    print(f"  Classes: {num_classes}")
    print(f"  Experts: {num_experts}")
    print(f"  Output: {output_dir}")
    
    # Get corruption parameters
    if args.p_del is not None and args.p_add is not None:
        # Custom parameters
        p_del = args.p_del
        p_add = args.p_add
        p_rev = args.p_rev or 0.05
        print(f"  Custom params: p_del={p_del}, p_add={p_add}, p_rev={p_rev}")
    else:
        # Use preset
        params = DISAGREEMENT_LEVELS[disagreement_level]
        p_del = params["p_del"]
        p_add = params["p_add"]
        p_rev = params["p_rev"]
        print(f"  Disagreement: {disagreement_level} (p_del={p_del}, p_add={p_add}, p_rev={p_rev})")
    
    # Generate experts
    if expert_types:
        # Structured expert bias
        print(f"  Expert types: {expert_types}")
        expert_u2c, expert_c2y, u2c_star, c2y_star = generate_structured_experts(
            dag_path=dag_path,
            num_classes=num_classes,
            expert_types=expert_types,
            base_seed=seed,
        )
        num_experts = len(expert_types)
    else:
        # Random corruption
        expert_u2c, expert_c2y, u2c_star, c2y_star = generate_expert_graphs_from_dag(
            dag_path=dag_path,
            num_classes=num_classes,
            num_experts=num_experts,
            p_del=p_del,
            p_add=p_add,
            p_rev=p_rev,
            base_seed=seed,
        )
    
    # Get node names
    concept_names, task_names = get_node_names(dag_path, num_classes)
    
    # Compute per-expert corruption statistics
    print("\n" + "="*60)
    print("CORRUPTION STATISTICS")
    print("="*60)
    
    K_u2c = u2c_star.shape[0] * u2c_star.shape[1]
    K_c2y = c2y_star.shape[0] * c2y_star.shape[1]
    gt_edges_u2c = u2c_star.bool().sum().item()
    gt_edges_c2y = c2y_star.bool().sum().item()
    gt_noedges_u2c = K_u2c - gt_edges_u2c
    gt_noedges_c2y = K_c2y - gt_edges_c2y
    
    print(f"\n  Ground truth u2c: {gt_edges_u2c} edges / {K_u2c} cells ({gt_edges_u2c/K_u2c*100:.1f}% density)")
    print(f"  Ground truth c2y: {gt_edges_c2y} edges / {K_c2y} cells ({gt_edges_c2y/K_c2y*100:.1f}% density)")
    
    print(f"\n  Per-expert corruption (vs ground truth):")
    print(f"  {'Expert':<10} {'u2c_diff':>10} {'u2c_%':>8} {'u2c_del':>8} {'u2c_add':>8} {'c2y_diff':>10} {'c2y_%':>8} {'c2y_del':>8} {'c2y_add':>8}")
    print(f"  {'-'*80}")
    
    for m in range(len(expert_u2c)):
        # u2c
        diff_u2c = (expert_u2c[m].bool() != u2c_star.bool())
        deleted_u2c = (u2c_star.bool() & ~expert_u2c[m].bool()).sum().item()
        added_u2c = (~u2c_star.bool() & expert_u2c[m].bool()).sum().item()
        total_diff_u2c = diff_u2c.sum().item()
        
        # c2y
        diff_c2y = (expert_c2y[m].bool() != c2y_star.bool())
        deleted_c2y = (c2y_star.bool() & ~expert_c2y[m].bool()).sum().item()
        added_c2y = (~c2y_star.bool() & expert_c2y[m].bool()).sum().item()
        total_diff_c2y = diff_c2y.sum().item()
        
        print(f"  expert_{m:<4} {total_diff_u2c:>10} {total_diff_u2c/K_u2c*100:>7.1f}% {deleted_u2c:>8} {added_u2c:>8} {total_diff_c2y:>10} {total_diff_c2y/K_c2y*100:>7.1f}% {deleted_c2y:>8} {added_c2y:>8}")
    
    # Majority vote statistics
    print(f"\n  Majority vote (aggregated) vs ground truth:")
    stats_u2c = compute_edge_statistics(expert_u2c, u2c_star)
    print(f"    u2c: edges_gt={stats_u2c['num_edges_gt']}, edges_majority={stats_u2c['num_edges_majority']}, P={stats_u2c['precision']:.3f}, R={stats_u2c['recall']:.3f}, F1={stats_u2c['f1']:.3f}")
    
    stats_c2y = compute_edge_statistics(expert_c2y, c2y_star)
    print(f"    c2y: edges_gt={stats_c2y['num_edges_gt']}, edges_majority={stats_c2y['num_edges_majority']}, P={stats_c2y['precision']:.3f}, R={stats_c2y['recall']:.3f}, F1={stats_c2y['f1']:.3f}")
    
    print("="*60)
    
    # Prepare config for saving
    save_config = {
        "dag_path": str(dag_path),
        "num_classes": num_classes,
        "num_experts": num_experts,
        "disagreement_level": disagreement_level,
        "expert_types": expert_types,
        "p_del": p_del,
        "p_add": p_add,
        "p_rev": p_rev,
        "seed": seed,
        "statistics": {
            "u2c": stats_u2c,
            "c2y": stats_c2y,
        },
    }
    
    # Save
    save_expert_graphs(
        expert_u2c_graphs=expert_u2c,
        expert_c2y_graphs=expert_c2y,
        output_dir=output_dir,
        config=save_config,
        concept_names=concept_names,
        task_names=task_names,
    )
    
    # Also save ground truth for reference
    gt_dir = Path(output_dir) / "ground_truth"
    gt_dir.mkdir(exist_ok=True)
    torch.save(u2c_star, gt_dir / "u2c_star.pt")
    torch.save(c2y_star, gt_dir / "c2y_star.pt")
    
    print(f"\nDone! Expert graphs saved to {output_dir}")


if __name__ == "__main__":
    main()
