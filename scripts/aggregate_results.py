"""
Aggregate results from all mCREAM experiments into a summary table.

This script:
1. Finds all result files (CSV) in experiments directory
2. Extracts key metrics
3. Creates a summary table for comparison

Usage:
    python scripts/aggregate_results.py --dataset cfmnist
    python scripts/aggregate_results.py --all --output results/full_summary.csv
"""

import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import json


def find_result_files(experiments_dir: Path, dataset: str = None) -> List[Path]:
    """Find all result CSV files."""
    
    if dataset:
        # Map short name to full dataset name
        dataset_map = {
            "cfmnist": "Complete_Concept_FMNIST",
            "cub": "CUB",
            "celeba": "CelebA",
        }
        search_dir = experiments_dir / dataset_map.get(dataset, dataset)
    else:
        search_dir = experiments_dir
    
    if not search_dir.exists():
        print(f"WARNING: Directory not found: {search_dir}")
        return []
    
    # Find all CSV files in metric directories
    return list(search_dir.rglob("**/last_metrics/*.csv"))


def parse_result_path(result_path: Path) -> Dict[str, str]:
    """Extract experiment info from result file path."""
    
    parts = result_path.parts
    
    info = {
        "dataset": None,
        "mode": None,
        "model": None,
        "method": None,
        "config_name": result_path.stem,
    }
    
    # Try to parse path like:
    # experiments/Complete_Concept_FMNIST/train_cbm/mCREAM/edge/edge_M5_medium/last_metrics/results.csv
    for i, part in enumerate(parts):
        if part in ["Complete_Concept_FMNIST", "CUB", "CelebA"]:
            info["dataset"] = part
        elif part == "train_cbm":
            info["mode"] = part
        elif part in ["CBM", "CREAM", "mCREAM"]:
            info["model"] = part
        elif part in ["edge", "graph", "combined", "baselines"]:
            info["method"] = part
    
    return info


def load_and_standardize_result(result_path: Path) -> Dict[str, Any]:
    """Load a result file and standardize column names."""
    
    try:
        df = pd.read_csv(result_path)
        
        if df.empty:
            return None
        
        # Convert to dict (take first row if multiple seeds averaged)
        result = df.iloc[0].to_dict() if len(df) == 1 else df.mean().to_dict()
        
        # Add path info
        path_info = parse_result_path(result_path)
        result.update(path_info)
        result["result_path"] = str(result_path)
        
        return result
    except Exception as e:
        print(f"WARNING: Could not load {result_path}: {e}")
        return None


def aggregate_results(experiments_dir: Path, dataset: str = None) -> pd.DataFrame:
    """Aggregate all results into a single DataFrame."""
    
    result_files = find_result_files(experiments_dir, dataset)
    print(f"Found {len(result_files)} result files")
    
    results = []
    for rf in result_files:
        result = load_and_standardize_result(rf)
        if result:
            results.append(result)
    
    if not results:
        print("No results found!")
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # Reorder columns for readability
    priority_cols = [
        "dataset", "model", "method", "config_name",
        "test_task_accuracy", "test_concept_accuracy",
        "u2c_f1", "c2y_f1",
        "training_time_min",
    ]
    
    other_cols = [c for c in df.columns if c not in priority_cols]
    ordered_cols = [c for c in priority_cols if c in df.columns] + other_cols
    df = df[ordered_cols]
    
    return df


def print_summary_table(df: pd.DataFrame):
    """Print a formatted summary table."""
    
    if df.empty:
        print("No results to display")
        return
    
    # Select key columns
    display_cols = [
        "model", "method", "config_name",
        "test_task_accuracy", "test_concept_accuracy",
        "u2c_f1", "c2y_f1",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)
    print(df[display_cols].to_string(index=False))
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Aggregate mCREAM experiment results")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cfmnist", "cub", "celeba"],
        help="Dataset to aggregate (optional)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Aggregate all datasets"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/summary.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--experiments-dir",
        type=str,
        default="./experiments",
        help="Experiments directory"
    )
    args = parser.parse_args()
    
    experiments_dir = Path(args.experiments_dir)
    
    if args.all:
        df = aggregate_results(experiments_dir)
    elif args.dataset:
        df = aggregate_results(experiments_dir, args.dataset)
    else:
        print("Please specify --dataset or --all")
        parser.print_help()
        return
    
    if df.empty:
        return
    
    # Print summary
    print_summary_table(df)
    
    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
