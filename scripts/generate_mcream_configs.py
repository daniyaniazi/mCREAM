"""
Generate all mCREAM configuration files for systematic experiments.

This script creates configs for:
- 3 datasets: cfmnist, cub, celeba
- 4 aggregation methods: union, intersection, majority, edge, graph, combined
- 4 expert counts: M = 1, 2, 5, 10
- 3 noise levels: low, medium, high

Usage:
    python scripts/generate_mcream_configs.py
"""

import yaml
from pathlib import Path
from typing import Dict, Any


# =============================================================================
# Dataset-specific settings
# =============================================================================

DATASETS = {
    "cfmnist": {
        "dataset_name": "Complete_Concept_FMNIST",
        "model_name": "Standard_FashionMNIST",
        "num_classes": 10,
        "num_concepts": 11,
        "num_exogenous": 128,  # Will be recalculated
        "num_side_channel": 40,
        "dag_file": "./data/FashionMNIST/Complete_Concept_FMNIST_DAG.csv",
        "softmax_mask": "./data/FashionMNIST/mutually_exclusive_relationships_COMPLETE.json",
        "input_model_path": "./pretrained_models/FMNIST/version_0/checkpoints/epoch=49-step=10750.ckpt",
        "concept_representation": "group_soft",
        "previous_model_output_size": 128,
    },
    "cub": {
        "dataset_name": "CUB",
        "model_name": "Standard_resnet18",
        "num_classes": 200,
        "num_concepts": 112,
        "num_exogenous": 512,
        "num_side_channel": 200,
        "dag_file": "./data/CUB/CUB_DAG_only_Gc.csv",
        "softmax_mask": "./data/CUB/CUB_mutually_exclusive_concepts.json",
        "input_model_path": "./pretrained_models/CUB/version_0/checkpoints/best.ckpt",
        "concept_representation": "group_soft",
        "previous_model_output_size": 512,
    },
    "celeba": {
        "dataset_name": "CelebA",
        "model_name": "Standard_resnet18",
        "num_classes": 2,
        "num_concepts": 39,
        "num_exogenous": 512,
        "num_side_channel": 200,
        "dag_file": "./data/CelebA/final_DAG_unfair.csv",
        "softmax_mask": None,
        "input_model_path": "./pretrained_models/CelebA/version_0/checkpoints/best.ckpt",
        "concept_representation": "soft",
        "previous_model_output_size": 512,
    },
}

# Disagreement levels (noise parameters)
DISAGREEMENT_LEVELS = {
    "low": {"p_del": 0.05, "p_add": 0.05, "p_rev": 0.02},
    "medium": {"p_del": 0.15, "p_add": 0.10, "p_rev": 0.05},
    "high": {"p_del": 0.30, "p_add": 0.20, "p_rev": 0.10},
}

# Aggregation methods
METHODS = {
    "baselines": ["union", "intersection", "majority"],
    "learnable": ["edge", "graph", "combined"],
}

# Expert counts to test
EXPERT_COUNTS = [1, 2, 5, 10]

# Default regularization weights
DEFAULT_REGULARIZATION = {
    "prior_weight": 0.1,
    "sparsity_weight": 0.01,
    "acyclicity_weight": 0.0,
}


def create_config(
    dataset_key: str,
    aggregation_type: str,
    num_experts: int,
    disagreement_level: str,
    seed: int = 42,
) -> Dict[str, Any]:
    """Create a single mCREAM config dictionary."""
    
    ds = DATASETS[dataset_key]
    
    config = {
        "mode": "train_cbm",
        "seed": seed,
        "dataset_name": ds["dataset_name"],
        
        # Dataset parameters
        "dataset_params": {
            "batch_size": 256,
            "workers": 2,
            "return_labels": True,
            "return_images": True,
        },
        
        # Model name
        "model_name": f"mCREAM_{ds['dataset_name']}",
        
        # mCREAM-specific settings
        "multi_expert": {
            "enabled": True,
            "num_experts": num_experts,
            "disagreement_level": disagreement_level,
            "aggregation_type": aggregation_type,
            "graph_regularization": DEFAULT_REGULARIZATION.copy(),
        },
        
        # Model hyperparameters (CREAM-compatible)
        "hyperparameters_model2": {
            "num_classes": ds["num_classes"],
            "num_concepts": ds["num_concepts"],
            "num_exogenous": ds["num_exogenous"],
            "num_side_channel": ds["num_side_channel"],
            "concept_representation": ds["concept_representation"],
            "num_hidden_layers_in_maskedmlp": 0,
            "previous_model_output_size": ds["previous_model_output_size"],
            "side_dropout": True,
            "dropout_prob": 0.9,
        },
        
        "hyperparameters": {
            "learning_rate": 0.001,
            "lambda_weight": 1,
            "frozen_model1": True,
        },
        
        # Training parameters
        "trainer_param": {
            "max_epochs": 50,
            "gradient_clip_val": None,
            "gradient_clip_algorithm": None,
        },
        
        # Paths
        "paths": {
            "default_root_dir": "./experiments/",
            "metric_dir": "./last_metrics/",
            "DAG_file": ds["dag_file"],
            "expert_graphs_dir": f"./data/{ds['dataset_name'].replace('Complete_Concept_', '')}/expert_graphs/M{num_experts}/{disagreement_level}/",
            "input_model_path": ds["input_model_path"],
        },
    }
    
    # Add softmax_mask if available
    if ds["softmax_mask"]:
        config["paths"]["softmax_mask"] = ds["softmax_mask"]
    
    return config


def generate_all_configs(output_dir: Path):
    """Generate all configuration files."""
    
    output_dir = Path(output_dir)
    
    for dataset_key in DATASETS.keys():
        dataset_dir = output_dir / dataset_key
        
        # Generate baseline configs (union, intersection, majority)
        baseline_dir = dataset_dir / "baselines"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        
        for method in METHODS["baselines"]:
            for noise in DISAGREEMENT_LEVELS.keys():
                # Only generate M=5 for baselines (most common)
                config = create_config(dataset_key, method, 5, noise)
                filename = f"{method}_M5_{noise}.yaml"
                
                with open(baseline_dir / filename, "w") as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                print(f"Created: {baseline_dir / filename}")
        
        # Generate learnable method configs (edge, graph, combined)
        for method in METHODS["learnable"]:
            method_dir = dataset_dir / method
            method_dir.mkdir(parents=True, exist_ok=True)
            
            for m in EXPERT_COUNTS:
                for noise in DISAGREEMENT_LEVELS.keys():
                    # Skip M=1 with high noise (not meaningful)
                    if m == 1 and noise != "medium":
                        continue
                    
                    # For M=1 and M=2, only do medium noise
                    if m <= 2 and noise != "medium":
                        continue
                    
                    config = create_config(dataset_key, method, m, noise)
                    filename = f"{method}_M{m}_{noise}.yaml"
                    
                    with open(method_dir / filename, "w") as f:
                        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                    print(f"Created: {method_dir / filename}")


def main():
    # Output directory
    output_dir = Path("all_configs/mcream_configs")
    
    print("=" * 60)
    print("Generating mCREAM Configuration Files")
    print("=" * 60)
    
    generate_all_configs(output_dir)
    
    print("\n" + "=" * 60)
    print("Done! Config files created in:")
    print(f"  {output_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
