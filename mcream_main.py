"""
Training script for mCREAM (Multi-Expert CREAM).

Usage:
    python mcream_main.py --config all_configs/mcream_configs/mCREAM_cfmnist_edge_medium.yaml

This script:
1. Loads expert graphs (or generates them if not found)
2. Creates mCREAM model with specified aggregation method
3. Trains and evaluates the model
4. Logs learned graph statistics
"""

import argparse
import pytorch_lightning as pl
import torch
from pathlib import Path
import yaml
import time
from typing import Tuple, List, Optional

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import get_component_with_dicts, load_config, dict_to_csv
from src.mcream_model import mCREAM_UtoC_Y
from src.expert_graphs.generation import (
    load_expert_graphs,
    generate_expert_graphs_from_dag,
    save_expert_graphs,
    load_and_split_dag,
    compute_edge_statistics,
    DISAGREEMENT_LEVELS,
)
from src.expert_graphs.aggregation import create_aggregation_module


def load_or_generate_expert_graphs(
    config: dict,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Load expert graphs from disk, or generate if not found.
    
    Returns:
        expert_u2c_graphs, expert_c2y_graphs, u2c_star, c2y_star
    """
    expert_dir = Path(config["paths"].get("expert_graphs_dir", ""))
    
    if expert_dir.exists() and (expert_dir / "config.yaml").exists():
        print(f"Loading expert graphs from {expert_dir}")
        expert_u2c, expert_c2y, gen_config = load_expert_graphs(expert_dir)
        
        # Load ground truth
        gt_dir = expert_dir / "ground_truth"
        if gt_dir.exists():
            u2c_star = torch.load(gt_dir / "u2c_star.pt", weights_only=True)
            c2y_star = torch.load(gt_dir / "c2y_star.pt", weights_only=True)
        else:
            num_classes = config["hyperparameters_model2"]["num_classes"]
            u2c_star, c2y_star = load_and_split_dag(config["paths"]["DAG_file"], num_classes)
        
        return expert_u2c, expert_c2y, u2c_star, c2y_star
    
    else:
        print(f"Expert graphs not found at {expert_dir}, generating...")
        
        multi_expert_config = config.get("multi_expert", {})
        num_experts = multi_expert_config.get("num_experts", 5)
        disagreement_level = multi_expert_config.get("disagreement_level", "medium")
        
        params = DISAGREEMENT_LEVELS[disagreement_level]
        
        expert_u2c, expert_c2y, u2c_star, c2y_star = generate_expert_graphs_from_dag(
            dag_path=config["paths"]["DAG_file"],
            num_classes=config["hyperparameters_model2"]["num_classes"],
            num_experts=num_experts,
            p_del=params["p_del"],
            p_add=params["p_add"],
            p_rev=params["p_rev"],
            base_seed=config.get("seed", 42),
        )
        
        # Save for future use
        if expert_dir:
            save_config = {
                "dag_path": str(config["paths"]["DAG_file"]),
                "num_classes": config["hyperparameters_model2"]["num_classes"],
                "num_experts": num_experts,
                "disagreement_level": disagreement_level,
                **params,
                "seed": config.get("seed", 42),
            }
            save_expert_graphs(expert_u2c, expert_c2y, expert_dir, save_config)
        
        return expert_u2c, expert_c2y, u2c_star, c2y_star


def create_mcream_model(
    config: dict,
    expert_u2c_graphs: List[torch.Tensor],
    expert_c2y_graphs: List[torch.Tensor],
    mutually_exclusive_concepts: Optional[List] = None,
) -> mCREAM_UtoC_Y:
    """Create mCREAM model from config."""
    
    multi_expert = config.get("multi_expert", {})
    hparams = config["hyperparameters_model2"]
    
    model = mCREAM_UtoC_Y(
        # Expert graphs
        expert_u2c_graphs=expert_u2c_graphs,
        expert_c2y_graphs=expert_c2y_graphs,
        aggregation_type=multi_expert.get("aggregation_type", "edge"),
        
        # Graph regularization
        prior_weight=multi_expert.get("graph_regularization", {}).get("prior_weight", 0.1),
        sparsity_weight=multi_expert.get("graph_regularization", {}).get("sparsity_weight", 0.01),
        acyclicity_weight=multi_expert.get("graph_regularization", {}).get("acyclicity_weight", 0.0),
        
        # CREAM parameters
        num_exogenous=hparams["num_exogenous"],
        num_concepts=hparams["num_concepts"],
        num_side_channel=hparams["num_side_channel"],
        num_classes=hparams["num_classes"],
        learning_rate=config["hyperparameters"]["learning_rate"],
        lambda_weight=config["hyperparameters"]["lambda_weight"],
        previous_model_output_size=hparams.get("previous_model_output_size"),
        concept_representation=hparams.get("concept_representation", "soft"),
        side_dropout=hparams.get("side_dropout", True),
        dropout_prob=hparams.get("dropout_prob", 0.9),
        num_hidden_layers_in_maskedmlp=hparams.get("num_hidden_layers_in_maskedmlp", 0),
        mutually_exclusive_concepts=mutually_exclusive_concepts,
    )
    
    return model


def evaluate_learned_graphs(
    model: mCREAM_UtoC_Y,
    u2c_star: torch.Tensor,
    c2y_star: torch.Tensor,
) -> dict:
    """
    Evaluate how well the model learned the true graph structure.
    
    Returns:
        Dictionary of graph recovery metrics
    """
    results = {}
    
    # Get learned graphs
    A_u2c_learned, A_c2y_learned = model.get_learned_graphs()
    
    # Binarize at 0.5 threshold
    A_u2c_binary = (A_u2c_learned > 0.5).float()
    A_c2y_binary = (A_c2y_learned > 0.5).float()
    
    # Compare to ground truth
    def compute_metrics(learned: torch.Tensor, gt: torch.Tensor, name: str):
        gt_bool = gt.bool()
        learned_bool = learned.bool()
        
        tp = (gt_bool & learned_bool).sum().item()
        fp = (~gt_bool & learned_bool).sum().item()
        fn = (gt_bool & ~learned_bool).sum().item()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        results[f"{name}_precision"] = precision
        results[f"{name}_recall"] = recall
        results[f"{name}_f1"] = f1
        results[f"{name}_num_edges_learned"] = learned_bool.sum().item()
        results[f"{name}_num_edges_gt"] = gt_bool.sum().item()
    
    compute_metrics(A_u2c_binary, u2c_star, "u2c")
    compute_metrics(A_c2y_binary, c2y_star, "c2y")
    
    # Get expert weights if available
    w_u2c, w_c2y = model.get_expert_weights()
    if w_u2c is not None:
        results["expert_weights_u2c"] = w_u2c.tolist()
    if w_c2y is not None:
        results["expert_weights_c2y"] = w_c2y.tolist()
    
    return results


def main(config_path: str):
    """Main training function."""
    
    # Load config
    config = load_config(config_path)
    config_path = Path(config_path)
    
    print(f"\n{'='*60}")
    print(f"mCREAM Training")
    print(f"Config: {config_path}")
    print(f"{'='*60}\n")
    
    # Set seed
    seed = config.get("seed", 42)
    pl.seed_everything(seed, workers=True)
    
    # Load dataset
    dataset_name = config["dataset_name"]
    print(f"Loading dataset: {dataset_name}")
    
    dataset_class = get_component_with_dicts("dataset", dataset_name)
    if "FMNIST" in dataset_name:
        dataset = dataset_class(
            **config["dataset_params"],
            seed=seed,
            full_concepts=(dataset_name == "Complete_Concept_FMNIST"),
        )
    else:
        dataset = dataset_class(**config["dataset_params"])
    
    # Load mutually exclusive concepts
    mutually_exclusive = None
    if "softmax_mask" in config["paths"]:
        import json
        with open(config["paths"]["softmax_mask"], "r") as f:
            mutually_exclusive = json.load(f)
        print(f"Loaded {len(mutually_exclusive)} mutex concept groups")
    
    # Load or generate expert graphs
    expert_u2c, expert_c2y, u2c_star, c2y_star = load_or_generate_expert_graphs(config)
    print(f"Loaded {len(expert_u2c)} expert graphs")
    print(f"  u2c shape: {expert_u2c[0].shape}")
    print(f"  c2y shape: {expert_c2y[0].shape}")
    
    # Create model
    print(f"\nCreating mCREAM model...")
    print(f"  Aggregation: {config.get('multi_expert', {}).get('aggregation_type', 'edge')}")
    
    model = create_mcream_model(
        config, expert_u2c, expert_c2y, mutually_exclusive
    )
    
    # Setup trainer
    max_epochs = config["trainer_param"]["max_epochs"]
    default_root_dir = (
        Path(config["paths"]["default_root_dir"])
        / dataset_name
        / config["mode"]
        / "mCREAM"
        / config_path.parent.name
    )
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        default_root_dir=default_root_dir,
        deterministic=True,
        enable_progress_bar=True,
    )
    
    # Train
    print(f"\nTraining for {max_epochs} epochs...")
    start_time = time.time()
    
    trainer.fit(model, datamodule=dataset)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time/60:.2f} minutes")
    
    # Test
    print("\nTesting...")
    test_start = time.time()
    trainer.test(model, datamodule=dataset)
    test_time = time.time() - test_start
    
    # Get metrics
    val_train_metrics = {
        key: value.item() for key, value in trainer.callback_metrics.items()
    }
    
    # Evaluate learned graphs
    print("\nEvaluating learned graphs...")
    graph_metrics = evaluate_learned_graphs(model, u2c_star, c2y_star)
    
    print(f"  u2c graph recovery:")
    print(f"    Precision: {graph_metrics['u2c_precision']:.3f}")
    print(f"    Recall: {graph_metrics['u2c_recall']:.3f}")
    print(f"    F1: {graph_metrics['u2c_f1']:.3f}")
    
    print(f"  c2y graph recovery:")
    print(f"    Precision: {graph_metrics['c2y_precision']:.3f}")
    print(f"    Recall: {graph_metrics['c2y_recall']:.3f}")
    print(f"    F1: {graph_metrics['c2y_f1']:.3f}")
    
    # Compile results
    results = {
        "seed": seed,
        "config_name": config_path.stem,
        "aggregation_type": config.get("multi_expert", {}).get("aggregation_type", "edge"),
        "num_experts": len(expert_u2c),
        "training_time_min": training_time / 60.0,
        "test_time_min": test_time / 60.0,
        **val_train_metrics,
        **graph_metrics,
    }
    
    # Save results
    metrics_dir = default_root_dir / Path(config["paths"]["metric_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    dict_to_csv(results, metrics_dir, config_path)
    
    print(f"\nResults saved to {metrics_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mCREAM Training Script")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file"
    )
    args = parser.parse_args()
    
    torch.set_float32_matmul_precision("high")
    main(args.config)
