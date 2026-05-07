"""
Run all mCREAM experiments for a dataset.

This script:
1. Runs all baseline methods (union, intersection, majority)
2. Runs all learnable methods (edge, graph, combined)
3. Organizes results by method and configuration

Usage:
    python scripts/run_mcream_experiments.py --dataset cfmnist
    python scripts/run_mcream_experiments.py --dataset cfmnist --method edge
    python scripts/run_mcream_experiments.py --config path/to/config.yaml
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List
import time


def find_configs(dataset: str, method: str = None) -> List[Path]:
    """Find all config files for a dataset/method combination."""
    
    config_base = Path("all_configs/mcream_configs") / dataset
    
    if not config_base.exists():
        print(f"ERROR: Config directory not found: {config_base}")
        return []
    
    configs = []
    
    if method:
        # Specific method
        if method in ["union", "intersection", "majority"]:
            method_dir = config_base / "baselines"
            pattern = f"{method}_*.yaml"
        else:
            method_dir = config_base / method
            pattern = "*.yaml"
        
        if method_dir.exists():
            configs = list(method_dir.glob(pattern))
    else:
        # All methods
        for subdir in config_base.iterdir():
            if subdir.is_dir():
                configs.extend(list(subdir.glob("*.yaml")))
    
    return sorted(configs)


def run_experiment(config_path: Path, dry_run: bool = False) -> bool:
    """Run a single experiment."""
    
    cmd = ["python", "mcream_main.py", "--config", str(config_path)]
    
    print(f"\n{'='*60}")
    print(f"Running: {config_path.name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    if dry_run:
        print("[DRY RUN] Would execute the above command")
        return True
    
    try:
        start = time.time()
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start
        print(f"Completed in {elapsed/60:.2f} minutes")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Experiment failed with code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run mCREAM experiments")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cfmnist", "cub", "celeba"],
        help="Dataset to run experiments for"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["union", "intersection", "majority", "edge", "graph", "combined"],
        help="Specific method to run (optional)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to specific config file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip experiments that already have results"
    )
    args = parser.parse_args()
    
    if args.config:
        # Run single config
        configs = [Path(args.config)]
    elif args.dataset:
        configs = find_configs(args.dataset, args.method)
    else:
        print("Please specify --dataset or --config")
        parser.print_help()
        return
    
    if not configs:
        print("No config files found!")
        return
    
    print(f"Found {len(configs)} experiment(s) to run:")
    for c in configs:
        print(f"  - {c}")
    
    # Run experiments
    success = 0
    failed = 0
    skipped = 0
    
    for config in configs:
        if args.skip_existing:
            # Check if results already exist
            # TODO: implement result checking
            pass
        
        if run_experiment(config, args.dry_run):
            success += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"  Success: {success}")
    print(f"  Failed:  {failed}")
    print(f"  Skipped: {skipped}")
    print(f"  Total:   {len(configs)}")


if __name__ == "__main__":
    main()
