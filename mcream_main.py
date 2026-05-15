"""
Training script for mCREAM (Multi-Expert CREAM).

Usage:
    python mcream_main.py --config all_configs/mcream_configs/cfmnist/baselines/union_M5_medium.yaml

This script:
1. Loads backbone model (x→u)
2. Loads expert graphs (or generates them if not found)
3. Creates mCREAM model with specified aggregation method
4. Trains and evaluates the model
5. Logs learned graph statistics
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
from src.mcream_model import mCREAM_UtoC_Y, mCREAM_Full
from src.expert_graphs.generation import (
    load_expert_graphs,
    generate_expert_graphs_from_dag,
    generate_structured_experts,
    save_expert_graphs,
    load_and_split_dag,
    compute_edge_statistics,
    DISAGREEMENT_LEVELS,
    EXPERT_BIAS_TYPES,
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
        expert_types = multi_expert_config.get("expert_types", None)
        
        # Use structured experts if expert_types is specified
        if expert_types:
            print(f"  Using structured expert types: {expert_types}")
            expert_u2c, expert_c2y, u2c_star, c2y_star = generate_structured_experts(
                dag_path=config["paths"]["DAG_file"],
                num_classes=config["hyperparameters_model2"]["num_classes"],
                expert_types=expert_types,
                base_seed=config.get("seed", 42),
            )
            num_experts = len(expert_types)
            save_config = {
                "dag_path": str(config["paths"]["DAG_file"]),
                "num_classes": config["hyperparameters_model2"]["num_classes"],
                "num_experts": num_experts,
                "expert_types": expert_types,
                "seed": config.get("seed", 42),
            }
        else:
            # Use uniform disagreement level
            params = DISAGREEMENT_LEVELS[disagreement_level]
            print(f"  Using disagreement level '{disagreement_level}': {params}")
            
            expert_u2c, expert_c2y, u2c_star, c2y_star = generate_expert_graphs_from_dag(
                dag_path=config["paths"]["DAG_file"],
                num_classes=config["hyperparameters_model2"]["num_classes"],
                num_experts=num_experts,
                p_del=params["p_del"],
                p_add=params["p_add"],
                p_rev=params["p_rev"],
                base_seed=config.get("seed", 42),
            )
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


def load_backbone(config: dict, dataset_name: str):
    """
    Load pretrained backbone model (x → u).
    
    This is the same backbone loading logic as in simple_main.py.
    """
    # Use backbone_model if specified, otherwise fall back to model_name
    model_name = config.get("backbone_model") or config.get("model_name")
    
    if model_name is None:
        raise ValueError("Config must specify 'backbone_model' or 'model_name'")
    
    # Strip any mCREAM prefix if present (e.g., "mCREAM_Complete_Concept_FMNIST" -> error)
    if model_name.startswith("mCREAM"):
        raise ValueError(
            f"backbone_model should be the backbone class (e.g., 'Standard_FashionMNIST'), "
            f"not '{model_name}'. Please update your config."
        )
    
    model_class = get_component_with_dicts("model", model_name)
    
    if "input_model_path" in config["paths"]:
        checkpoint_path = Path(config["paths"]["input_model_path"])
        backbone = model_class.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            dataset=dataset_name,
            frozen=config["hyperparameters"].get("frozen_model1", True),
        )
        print(f"Loaded backbone from: {checkpoint_path}")
    else:
        # Train from scratch (not recommended)
        hyperparams = {
            "num_classes": config["hyperparameters_model2"]["num_classes"],
            "learning_rate": config["hyperparameters"]["learning_rate"],
        }
        backbone = model_class(**hyperparams)
        print("Created new backbone (not pretrained)")
    
    return backbone


def create_mcream_model(
    config: dict,
    expert_u2c_graphs: List[torch.Tensor],
    expert_c2y_graphs: List[torch.Tensor],
    mutually_exclusive_concepts: Optional[List] = None,
) -> mCREAM_UtoC_Y:
    """Create mCREAM_UtoC_Y model (u→c,y part only) from config."""
    
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


def create_full_mcream_model(
    config: dict,
    backbone: pl.LightningModule,
    expert_u2c_graphs: List[torch.Tensor],
    expert_c2y_graphs: List[torch.Tensor],
    mutually_exclusive_concepts: Optional[List] = None,
) -> mCREAM_Full:
    """Create full mCREAM model (backbone + u→c,y) from config."""
    
    # Create the u→c,y part
    mcream_ucy = create_mcream_model(
        config, expert_u2c_graphs, expert_c2y_graphs, mutually_exclusive_concepts
    )
    
    # Wrap with backbone
    full_model = mCREAM_Full(
        backbone=backbone,
        mcream_model=mcream_ucy,
        frozen_backbone=config["hyperparameters"].get("frozen_model1", True),
        learning_rate=config["hyperparameters"]["learning_rate"],
    )
    
    return full_model


def evaluate_learned_graphs(
    model,  # mCREAM_Full or mCREAM_UtoC_Y
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


def run_single_seed(config: dict, config_path: Path, seed: int):
    """Run a single mCREAM experiment with a given seed. Returns results dict."""
    
    print(f"\n{'='*60}")
    print(f"mCREAM Training  |  seed={seed}")
    print(f"Config: {config_path}")
    print(f"{'='*60}\n")
    
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
    
    # Load backbone model (x → u)
    print(f"\nLoading backbone model...")
    backbone = load_backbone(config, dataset_name)
    
    # Create full mCREAM model (backbone + u→c,y)
    print(f"\nCreating mCREAM model...")
    print(f"  Aggregation: {config.get('multi_expert', {}).get('aggregation_type', 'edge')}")
    
    model = create_full_mcream_model(
        config, backbone, expert_u2c, expert_c2y, mutually_exclusive
    )
    
    # Setup trainer
    max_epochs = config["trainer_param"]["max_epochs"]
    
    # Use experiment_name if provided, otherwise fall back to config filename
    experiment_name = config.get("experiment_name", config_path.stem)
    
    default_root_dir = (
        Path(config["paths"]["default_root_dir"])
        / dataset_name
        / config["mode"]
        / "mCREAM"
        / experiment_name
        / f"seed_{seed}"
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
    
    # Get metrics (includes val_* and test_* from trainer)
    val_train_metrics = {
        key: value.item() for key, value in trainer.callback_metrics.items()
    }
    
    # =========================================================================
    # Save intermediate values (like CREAM does for comparison)
    # =========================================================================
    pl_checkpoint_path = trainer.logger.log_dir
    
    from src.saving_intermediate_utils import save_intermediate_values
    
    print("\nSaving intermediate values for analysis...")
    
    train_latent = save_intermediate_values(
        dataset=dataset,
        dataset_name=dataset_name,
        model=model,
        training_set=True,
        DAG_path=config["paths"]["DAG_file"],
        seed=seed,
        save_directory=pl_checkpoint_path,
    )
    
    test_latent = save_intermediate_values(
        dataset=dataset,
        dataset_name=dataset_name,
        model=model,
        training_set=False,
        DAG_path=config["paths"]["DAG_file"],
        seed=seed,
        save_directory=pl_checkpoint_path,
    )
    
    print(f"  Saved to: {pl_checkpoint_path}")
    
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
    
    # Compile results (CREAM-compatible format)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    results = {
        # === CREAM-compatible fields ===
        "max_epochs": max_epochs,
        "seed": seed,
        "experiment_name": config_path.stem,
        "train_val_time": training_time / 60.0,
        "test_time": test_time / 60.0,
        "num_trainable_parameters": num_params,
        
        # === Core metrics (from trainer — includes val_* and test_*) ===
        **val_train_metrics,
        
        # === mCREAM-specific fields ===
        "aggregation_type": config.get("multi_expert", {}).get("aggregation_type", "edge"),
        "num_experts": len(expert_u2c),
        "disagreement_level": config.get("multi_expert", {}).get("disagreement_level", "medium"),
        
        # === Graph recovery metrics ===
        **graph_metrics,
    }
    
    # =========================================================================
    # PFI (Permutation Feature Importance) — same as CREAM
    # =========================================================================
    num_concepts = config["hyperparameters_model2"]["num_concepts"]
    num_side = config["hyperparameters_model2"]["num_side_channel"]
    
    if num_side > 0:
        from src.PFI_accuracy import PFI_accuracies
        
        print("\nComputing PFI importances...")
        concept_dropped_score, side_dropped_score = PFI_accuracies(
            model.u_to_CY.last_layer, test_latent[0], num_concepts, repeat=100
        )
        PFI_concept_importance = results["test_task_accuracy"] - concept_dropped_score
        PFI_side_importance = results["test_task_accuracy"] - side_dropped_score
        print(f"  PFI concept importance: {PFI_concept_importance}")
        print(f"  PFI side importance: {PFI_side_importance}")
        results["PFI_concept_importance"] = PFI_concept_importance
        results["PFI_side_importance"] = PFI_side_importance
        
        # =================================================================
        # SAGE / CCI (Concept Completeness Index) — same as CREAM
        # =================================================================
        print("\nComputing SAGE / CCI...")
        try:
            import sage
            from torch import nn
            from src.sage_importance_functions import (
                prepare_shap_data,
                group_importance_metric,
            )
            from src.diff_permutation_estimator import (
                PermutationEstimator as my_PermutationEstimator,
            )
            from src.utils import timeout
            
            workers = config.get("dataset_params", {}).get("num_workers", 4)
            batch_size = config.get("dataset_params", {}).get("batch_size", 128)
            
            @timeout(3600)
            def run_sage(train_latent, test_latent, config, results):
                sage_df_train = train_latent[1]
                sage_df_test = test_latent[1]
                
                train_x, _, train_group_names, train_groups = prepare_shap_data(sage_df_train)
                train_x = train_x.to_numpy()
                
                test_x, test_y, test_group_names, test_groups = prepare_shap_data(sage_df_test)
                test_x = test_x.to_numpy()
                test_y = test_y.to_numpy()
                
                assert train_group_names == test_group_names and train_groups == test_groups
                
                num_classes = config["hyperparameters_model2"]["num_classes"]
                if num_classes == 1:
                    explained_model = nn.Sequential(model.u_to_CY.last_layer, nn.Sigmoid())
                else:
                    explained_model = nn.Sequential(model.u_to_CY.last_layer, nn.Softmax(dim=1))
                
                twenty_pct = int(len(sage_df_train) * 0.2)
                imputer = sage.GroupedMarginalImputer(
                    explained_model, train_x[:twenty_pct], test_groups
                )
                estimator = my_PermutationEstimator(
                    imputer, "cross entropy", random_state=seed, n_jobs=workers
                )
                sage_values = estimator(test_x, test_y, batch_size=batch_size, thresh=0.05, bar=False)
                
                explanation_values = dict(zip(test_group_names, sage_values.values))
                cci = group_importance_metric(explanation_values)
                print(f"  CCI: {cci}")
                
                results["CCI"] = cci
                results["debugging_sage_metrics_concepts"] = explanation_values["concepts"]
                results["debugging_sage_metrics_side_channel"] = explanation_values["side_channel"]
                return results
            
            results = run_sage(train_latent, test_latent, config, results)
        except Exception as e:
            print(f"  SAGE/CCI failed: {e}")
            results["CCI"] = None
            results["debugging_sage_metrics_concepts"] = None
            results["debugging_sage_metrics_side_channel"] = None
    
    # Save results to version-specific directory
    pl_checkpoint_path = Path(trainer.logger.log_dir)
    
    print(f"\nSaving results to version folder: {pl_checkpoint_path}")
    dict_to_csv(results, pl_checkpoint_path, config_path)
    
    # Also save to central metrics directory
    metrics_dir = Path(config["paths"]["default_root_dir"]) / "metrics" / dataset_name / "mCREAM"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    dict_to_csv(results, metrics_dir, config_path)
    
    print(f"Results also saved to: {metrics_dir}")
    print(f"{'='*60}\n")
    
    return results


def main(config_path: str):
    """Main training function with multi-seed support."""
    import numpy as np
    import pandas as pd
    
    config = load_config(config_path)
    config_path = Path(config_path)
    
    # Multi-seed support: config can specify seeds as list or single int
    seeds = config.get("seeds", None)
    if seeds is None:
        seeds = [config.get("seed", 42)]
    
    print(f"\n{'#'*60}")
    print(f"mCREAM Multi-Seed Run  |  seeds={seeds}")
    print(f"{'#'*60}\n")
    
    all_results = []
    for i, seed in enumerate(seeds):
        print(f"\n>>> Seed {i+1}/{len(seeds)}: {seed}")
        # Override the seed in config for this run
        run_config = {**config, "seed": seed}
        result = run_single_seed(run_config, config_path, seed)
        all_results.append(result)
    
    # If multiple seeds, compute and save summary (mean ± std)
    if len(seeds) > 1:
        print(f"\n{'#'*60}")
        print(f"MULTI-SEED SUMMARY ({len(seeds)} seeds)")
        print(f"{'#'*60}")
        
        df = pd.DataFrame(all_results)
        
        # Identify numeric columns for aggregation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude seed and max_epochs from aggregation
        agg_cols = [c for c in numeric_cols if c not in ("seed", "max_epochs", "num_experts", "num_trainable_parameters")]
        
        summary = {"experiment_name": config_path.stem, "num_seeds": len(seeds), "seeds": str(seeds)}
        for col in agg_cols:
            vals = df[col].dropna()
            if len(vals) > 0:
                summary[f"{col}_mean"] = vals.mean()
                summary[f"{col}_std"] = vals.std()
                print(f"  {col}: {vals.mean():.4f} ± {vals.std():.4f}")
        
        # Save per-seed CSV and summary CSV
        dataset_name = config["dataset_name"]
        metrics_dir = Path(config["paths"]["default_root_dir"]) / "metrics" / dataset_name / "mCREAM"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        per_seed_path = metrics_dir / f"{config_path.stem}_per_seed.csv"
        df.to_csv(per_seed_path, index=False)
        print(f"\nPer-seed results: {per_seed_path}")
        
        summary_df = pd.DataFrame([summary])
        summary_path = metrics_dir / f"{config_path.stem}_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary results:  {summary_path}")
        
        print(f"\n{'#'*60}\n")


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
