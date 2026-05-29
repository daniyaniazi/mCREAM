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
from src.mcream_model import mCREAM_UtoC_Y, mCREAM_Full, SoftMaskedLinear
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
        prior_weight=multi_expert.get("graph_regularization", {}).get("prior_weight", 0.01),
        sparsity_weight=multi_expert.get("graph_regularization", {}).get("sparsity_weight", 0.001),
        acyclicity_weight=multi_expert.get("graph_regularization", {}).get("acyclicity_weight", 0.0),
        
        # Graph learning schedule
        graph_lr=multi_expert.get("graph_lr", 0.01),
        graph_warmup_epochs=multi_expert.get("graph_warmup_epochs", 5),
        separate_graph_opt=multi_expert.get("separate_graph_opt", True),
        
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
    
    # Print per-expert corruption statistics
    print(f"\n  Per-expert corruption (vs ground truth):")
    expert_corruption_stats = []
    for m in range(len(expert_u2c)):
        diff_u2c = (expert_u2c[m].bool() != u2c_star.bool()).sum().item()
        diff_c2y = (expert_c2y[m].bool() != c2y_star.bool()).sum().item()
        total_cells_u2c = u2c_star.numel()
        total_cells_c2y = c2y_star.numel()
        pct_u2c = diff_u2c / total_cells_u2c * 100
        pct_c2y = diff_c2y / total_cells_c2y * 100
        print(f"    expert_{m}: u2c {diff_u2c}/{total_cells_u2c} ({pct_u2c:.1f}%) changed, c2y {diff_c2y}/{total_cells_c2y} ({pct_c2y:.1f}%) changed")
        expert_corruption_stats.append({
            f"expert_{m}_u2c_corruption_pct": pct_u2c,
            f"expert_{m}_c2y_corruption_pct": pct_c2y,
        })
    
    # Move ground truth to GPU for later comparisons with learned graphs
    if torch.cuda.is_available():
        u2c_star = u2c_star.cuda()
        c2y_star = c2y_star.cuda()
    
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
    
    # Track GPU memory
    peak_gpu_memory_mb = 0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # Train
    print(f"\nTraining for {max_epochs} epochs...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    trainer.fit(model, datamodule=dataset)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    training_time = time.perf_counter() - start_time
    print(f"Training completed in {training_time/60:.2f} minutes")
    
    # Test
    print("\nTesting...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    test_start = time.perf_counter()
    trainer.test(model, datamodule=dataset)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    test_time = time.perf_counter() - test_start
    
    # Capture peak GPU memory
    if torch.cuda.is_available():
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak GPU memory: {peak_gpu_memory_mb:.1f} MB")
    
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
    
    # =========================================================================
    # Benchmark (CREAM's run_benchmark — CPU/GPU timing & memory)
    # =========================================================================
    print("\nRunning benchmark (CPU/GPU timing & memory)...")
    from src.utils import run_benchmark
    try:
        # CREAM's run_benchmark expects model(batch) → (y, c) and uses
        # model.concept_loss_function / model.task_loss_function for backprop.
        # mCREAM_Full returns (y, c, c_logits) and has different loss API.
        # We wrap the model to make it compatible.
        class BenchmarkWrapper(torch.nn.Module):
            def __init__(self, mcream_model):
                super().__init__()
                self.mcream_model = mcream_model
                # Expose loss functions that run_benchmark expects
                self.concept_loss_function = torch.nn.BCEWithLogitsLoss()
                self.task_loss_function = torch.nn.CrossEntropyLoss()
            def forward(self, x):
                y, c, _ = self.mcream_model(x)
                return y, c
        
        benchmark_model = BenchmarkWrapper(model)
        benchmark_results = run_benchmark(benchmark_model, dataset.test_dataloader())
        print(f"  GPU time (no backprop): {benchmark_results.get('GPU_time_backprop_False', 'N/A'):.4f}s")
        print(f"  GPU memory (no backprop): {benchmark_results.get('GPU_memory_backprop_False', 'N/A'):.1f} MB")
    except Exception as e:
        print(f"  Benchmark failed: {e}")
        benchmark_results = {}
    
    # =========================================================================
    # Dropout test accuracy (CREAM's trainer.predict → dropout_test_acc)
    # Test with side channel fully dropped (p=1.0) to check concept sufficiency
    # =========================================================================
    dropout_test_acc = None
    num_side = config["hyperparameters_model2"]["num_side_channel"]
    if num_side > 0 and hasattr(model, 'u_to_CY') and model.u_to_CY.side_channel is not None:
        print("\nTesting with side channel fully dropped (dropout p=1.0)...")
        try:
            model.eval()
            # Find the StochasticDepth layer and set p=1 (drop everything)
            from torchvision.ops import StochasticDepth
            original_p = None
            stoch_layer = None
            for module in model.u_to_CY.side_channel.modules():
                if isinstance(module, StochasticDepth):
                    original_p = module.p
                    module.p = 1.0  # Drop all side channel
                    module.train()  # StochasticDepth only drops in train mode
                    stoch_layer = module
                    break
            
            if stoch_layer is not None:
                all_preds = []
                all_labels = []
                dataset.setup(stage="test")
                test_loader = dataset.test_dataloader()
                
                with torch.no_grad():
                    for batch in test_loader:
                        x, true_concepts, y_true = batch
                        if torch.cuda.is_available():
                            x = x.cuda()
                            model = model.cuda()
                        
                        y_pred, c_pred, _ = model(x)
                        
                        if config["hyperparameters_model2"]["num_classes"] == 1:
                            task_preds = (torch.sigmoid(y_pred) > 0.5).int().view(-1)
                        else:
                            task_preds = y_pred.argmax(dim=1)
                        
                        all_preds.append(task_preds.cpu())
                        all_labels.append(y_true.view(-1).cpu())
                
                all_preds = torch.cat(all_preds)
                all_labels = torch.cat(all_labels)
                dropout_test_acc = (all_preds == all_labels).float().mean().item()
                print(f"  Dropout test accuracy (side fully dropped): {dropout_test_acc:.4f}")
                
                # Restore original dropout
                stoch_layer.p = original_p
                stoch_layer.eval()
            else:
                print("  No StochasticDepth layer found in side channel")
        except Exception as e:
            print(f"  Dropout test failed: {e}")
    
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
    
    # =========================================================================
    # Save aggregated graph as CSV (same format as ground-truth DAG)
    # =========================================================================
    import pandas as pd
    import numpy as np
    
    A_u2c_learned, A_c2y_learned = model.get_learned_graphs()
    
    # Read node names from ground-truth DAG
    gt_dag = pd.read_csv(config["paths"]["DAG_file"], index_col=0)
    node_names = list(gt_dag.index)
    K = config["hyperparameters_model2"]["num_concepts"]
    T = config["hyperparameters_model2"]["num_classes"]
    
    # Reconstruct full (K+T) × (K+T) adjacency matrix
    full_adj = np.zeros((K + T, K + T), dtype=float)
    
    # u2c block [K×K] — soft values
    u2c_np = A_u2c_learned.detach().cpu().numpy()
    full_adj[:K, :K] = u2c_np[:K, :K]
    
    # c2y block [T×(K+T)] — soft values
    c2y_np = A_c2y_learned.detach().cpu().numpy()
    full_adj[K:, :] = c2y_np[:T, :K + T]
    
    # Save soft (continuous) version
    soft_dag_df = pd.DataFrame(full_adj, index=node_names, columns=node_names)
    graph_save_dir = Path(pl_checkpoint_path) / "learned_graphs"
    graph_save_dir.mkdir(parents=True, exist_ok=True)
    
    soft_path = graph_save_dir / "aggregated_dag_soft.csv"
    soft_dag_df.round(4).to_csv(soft_path)
    print(f"\n  Saved soft aggregated DAG to: {soft_path}")
    
    # Save binary (thresholded at 0.5) version
    binary_adj = (full_adj > 0.5).astype(bool)
    binary_dag_df = pd.DataFrame(binary_adj, index=node_names, columns=node_names)
    binary_path = graph_save_dir / "aggregated_dag_binary.csv"
    binary_dag_df.to_csv(binary_path)
    print(f"  Saved binary aggregated DAG to: {binary_path}")
    
    # Also save ground truth for easy side-by-side comparison
    gt_path = graph_save_dir / "ground_truth_dag.csv"
    gt_dag.to_csv(gt_path)
    print(f"  Saved ground truth DAG to: {gt_path}")
    
    # Save raw u2c and c2y tensors
    torch.save(A_u2c_learned.detach().cpu(), graph_save_dir / "learned_u2c.pt")
    torch.save(A_c2y_learned.detach().cpu(), graph_save_dir / "learned_c2y.pt")
    
    # =========================================================================
    # Interventions (CREAM Figure 6 equivalent)
    # =========================================================================
    print("\nRunning intervention experiment...")
    
    # Get activation percentiles for soft intervention values
    from src.saving_intermediate_utils import save_activation_percentiles
    
    concept_rep = config["hyperparameters_model2"].get("concept_representation", "soft")
    intervention_percentile_df = None
    if concept_rep in ("soft", "group_soft", "logits"):
        intervention_percentile_df = save_activation_percentiles(
            dataset=dataset,
            dataset_name=dataset_name,
            model=model,
            DAG_path=config["paths"]["DAG_file"],
        )
    
    # Determine max interventions (number of concepts, like CREAM)
    A_c2y_for_direct = A_c2y_learned.detach()
    concept_cols = A_c2y_for_direct[:, :K]
    direct_mask = concept_cols.sum(dim=0) > 0.5
    num_direct = direct_mask.sum().item()
    max_interventions = K  # iterate over ALL concepts (like CREAM)
    print(f"  Direct concepts: {num_direct}, max interventions: {max_interventions}")
    
    # Check if propagating interventions are supported for this config
    can_propagate = (
        hasattr(model, 'u_to_CY') and 
        model.u_to_CY.input_per_concept == 1 and
        isinstance(model.u_to_CY.u2c_model, SoftMaskedLinear)
    ) if hasattr(model, 'u_to_CY') else (
        model.input_per_concept == 1 and
        isinstance(model.u2c_model, SoftMaskedLinear)
    )
    
    # Run interventions for both simple and propagating (if supported)
    intervention_modes = ["simple"]
    if can_propagate:
        intervention_modes.append("propagating")
        print("  Propagating interventions: ENABLED (input_per_concept=1, depth=0)")
    else:
        print("  Propagating interventions: DISABLED (input_per_concept>1 or hidden layers)")
    
    all_intervention_results = []
    
    for interv_mode in intervention_modes:
        print(f"\n  Running {interv_mode} interventions...")
        intervention_results = []
        intervention_trainer = pl.Trainer(
            max_epochs=max_epochs,
            default_root_dir=default_root_dir,
            enable_progress_bar=False,
            deterministic=True,
            logger=False,
        )
        
        for n_interv in range(max_interventions + 1):
            model.eval()
            all_task_correct = []
            all_concept_correct = []
            
            dataset.setup(stage="test")
            test_loader = dataset.test_dataloader()
            
            with torch.no_grad():
                for batch in test_loader:
                    x, true_concepts, y_true = batch
                    if torch.cuda.is_available():
                        x = x.cuda()
                        true_concepts = true_concepts.cuda()
                        y_true = y_true.cuda()
                        model = model.cuda()
                    
                    # Convert hard interventions to soft if needed
                    interv_concepts = true_concepts.clone()
                    if intervention_percentile_df is not None and concept_rep in ("soft", "group_soft", "logits"):
                        percentiles_5th = torch.tensor(
                            intervention_percentile_df["5th_percentile"].values,
                            device=x.device, dtype=x.dtype
                        )
                        percentiles_95th = torch.tensor(
                            intervention_percentile_df["95th_percentile"].values,
                            device=x.device, dtype=x.dtype
                        )
                        interv_concepts = true_concepts.float() * percentiles_95th + (1 - true_concepts.float()) * percentiles_5th
                    
                    # Choose intervention method
                    if interv_mode == "propagating":
                        y_pred, c_pred, _ = model.forward_with_propagating_interventions(
                            x, interv_concepts, num_interventions=n_interv
                        )
                    else:
                        y_pred, c_pred, _ = model.forward_with_interventions(
                            x, interv_concepts, num_interventions=n_interv
                        )
                    
                    # Task accuracy
                    if config["hyperparameters_model2"]["num_classes"] == 1:
                        task_preds = (torch.sigmoid(y_pred) > 0.5).int().view(-1)
                        all_task_correct.append((task_preds == y_true.view(-1)).float())
                    else:
                        task_preds = y_pred.argmax(dim=1)
                        all_task_correct.append((task_preds == y_true.view(-1)).float())
                    
                    # Concept accuracy
                    all_concept_correct.append(((c_pred > 0.5) == true_concepts).float().mean(dim=1))
            
            task_acc = torch.cat(all_task_correct).mean().item()
            concept_acc = torch.cat(all_concept_correct).mean().item()
            
            intervention_results.append({
                "mode": interv_mode,
                "num_interventions": n_interv,
                "test_task_accuracy": task_acc,
                "test_concept_accuracy": concept_acc,
            })
            print(f"    [{interv_mode}] interventions={n_interv}: task_acc={task_acc:.4f}, concept_acc={concept_acc:.4f}")
        
        all_intervention_results.extend(intervention_results)
    
    # Save all intervention results as CSV
    interv_df = pd.DataFrame(all_intervention_results)
    interv_path = Path(pl_checkpoint_path) / "intervention_results.csv"
    interv_df.to_csv(interv_path, index=False)
    print(f"  Saved intervention results to: {interv_path}")
    
    # =========================================================================
    # Exogenous Correlation Matrix (CREAM Figure 7/15 equivalent)
    # =========================================================================
    print("\nComputing exogenous correlation matrix...")
    
    all_exogenous = []
    model.eval()
    dataset.setup(stage="test")
    with torch.no_grad():
        for batch in dataset.test_dataloader():
            x = batch[0]
            if torch.cuda.is_available():
                x = x.cuda()
                model = model.cuda()
            # Get exogenous variables (after backbone + splitter)
            u_features = model.x_to_u.concept_extractor(x)
            u_split = model.u_to_CY.u2u_model(u_features)
            all_exogenous.append(u_split.cpu())
    
    all_exogenous = torch.cat(all_exogenous, dim=0)  # [N, num_exogenous]
    corr_matrix = torch.corrcoef(all_exogenous.T).numpy()  # [num_exo, num_exo]
    
    # Save correlation matrix
    corr_save_dir = Path(pl_checkpoint_path) / "correlation_analysis"
    corr_save_dir.mkdir(parents=True, exist_ok=True)
    
    corr_df = pd.DataFrame(corr_matrix)
    corr_df.to_csv(corr_save_dir / "exogenous_correlation_matrix.csv")
    
    # Save absolute correlation as heatmap-ready CSV with concept labels
    abs_corr = np.abs(corr_matrix)
    abs_corr_df = pd.DataFrame(abs_corr)
    abs_corr_df.to_csv(corr_save_dir / "exogenous_abs_correlation_matrix.csv")
    print(f"  Saved correlation matrices to: {corr_save_dir}")
    
    # =========================================================================
    # Concept Leakage Check (CREAM Table 4 equivalent)
    # Train C_true→Y baseline in-situ (same dataset split, same seed)
    # Λ = max(ACC_f - ACC_optimal, 0)
    # =========================================================================
    print("\nTraining C_true→Y baseline for leakage check...")
    K = config["hyperparameters_model2"]["num_concepts"]
    T = config["hyperparameters_model2"]["num_classes"]
    
    # Simple linear model: ground truth concepts → task label
    c2y_baseline = torch.nn.Linear(K, T)
    if torch.cuda.is_available():
        c2y_baseline = c2y_baseline.cuda()
    
    c2y_optimizer = torch.optim.Adam(c2y_baseline.parameters(), lr=0.001)
    
    # Train on training set
    dataset.setup(stage="fit")
    train_loader = dataset.train_dataloader()
    c2y_baseline.train()
    for epoch in range(50):  # 50 epochs is plenty for a linear model
        for batch in train_loader:
            _, true_concepts, y_true = batch
            if torch.cuda.is_available():
                true_concepts = true_concepts.cuda().float()
                y_true = y_true.cuda()
            
            y_pred = c2y_baseline(true_concepts)
            if T == 1:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    y_pred.view(-1), y_true.float().view(-1)
                )
            else:
                loss = torch.nn.functional.cross_entropy(y_pred, y_true)
            
            c2y_optimizer.zero_grad()
            loss.backward()
            c2y_optimizer.step()
    
    # Evaluate on test set
    c2y_baseline.eval()
    dataset.setup(stage="test")
    test_loader = dataset.test_dataloader()
    all_correct = []
    with torch.no_grad():
        for batch in test_loader:
            _, true_concepts, y_true = batch
            if torch.cuda.is_available():
                true_concepts = true_concepts.cuda().float()
                y_true = y_true.cuda()
            
            y_pred = c2y_baseline(true_concepts)
            if T == 1:
                preds = (torch.sigmoid(y_pred) > 0.5).int().view(-1)
            else:
                preds = y_pred.argmax(dim=1)
            all_correct.append((preds == y_true.view(-1)).float())
    
    c2y_baseline_acc = torch.cat(all_correct).mean().item()
    print(f"  C_true→Y baseline accuracy (ACC_optimal): {c2y_baseline_acc:.4f}")
    
    # Compute leakage: Λ = max(ACC_f - ACC_optimal, 0)
    # ACC_f = model's test task accuracy (from no-side runs, this is the full model without shortcuts)
    acc_f = val_train_metrics.get("test_task_accuracy", 0.0)
    leakage = max(acc_f - c2y_baseline_acc, 0.0)
    print(f"  Model test accuracy (ACC_f): {acc_f:.4f}")
    print(f"  Concept leakage (Λ): {leakage:.4f}")
    
    # Compile results (CREAM-compatible format)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count soft-masked params (mCREAM equivalent of CREAM's count_maskedmlp_params)
    # In CREAM, disconnected weights (mask=0) are subtracted from param count.
    # In mCREAM, we count weights where the soft mask < 0.5 (effectively disconnected).
    disconnected_weights = 0
    A_u2c_for_mask = model.u_to_CY.graph_agg_u2c().detach()
    A_c2y_for_mask = model.u_to_CY.graph_agg_c2y().detach()
    mask_u2c = model.u_to_CY._build_u2c_mask(A_u2c_for_mask)
    mask_c2y = model.u_to_CY._build_c2y_mask(A_c2y_for_mask)
    disconnected_weights += (mask_u2c < 0.5).sum().item()  # u2c layer
    disconnected_weights += (mask_c2y < 0.5).sum().item()  # c2y layer
    effective_params = num_params - disconnected_weights
    
    results = {
        # === CREAM-compatible fields ===
        "max_epochs": max_epochs,
        "seed": seed,
        "experiment_name": config_path.stem,
        "train_val_time": training_time / 60.0,
        "test_time": test_time / 60.0,
        "num_trainable_parameters": num_params,
        "num_effective_parameters": effective_params,
        "disconnected_weights": disconnected_weights,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "test_dropout_task_accuracy": dropout_test_acc,
        "c2y_baseline_accuracy": c2y_baseline_acc,
        "concept_leakage": leakage,
        
        # === Benchmark results (CREAM's run_benchmark) ===
        **benchmark_results,
        
        # === Core metrics (from trainer — includes val_* and test_*) ===
        **val_train_metrics,
        
        # === mCREAM-specific fields ===
        "aggregation_type": config.get("multi_expert", {}).get("aggregation_type", "edge"),
        "num_experts": len(expert_u2c),
        "disagreement_level": config.get("multi_expert", {}).get("disagreement_level", "medium"),
        
        # === Graph recovery metrics ===
        **graph_metrics,
        
        # === Per-expert corruption stats ===
        **{k: v for d in expert_corruption_stats for k, v in d.items()},
        
        # === Learned graph corruption vs GT ===
        "learned_u2c_corruption_pct": (A_u2c_learned.detach() > 0.5).bool().ne(u2c_star.bool()).sum().item() / u2c_star.numel() * 100,
        "learned_c2y_corruption_pct": (A_c2y_learned.detach() > 0.5).bool().ne(c2y_star.bool()).sum().item() / c2y_star.numel() * 100,
        
        # === Intervention results (from simple mode) ===
        "num_direct_concepts": num_direct,
        "intervention_acc_0": next((r["test_task_accuracy"] for r in all_intervention_results if r["mode"] == "simple" and r["num_interventions"] == 0), None),
        "intervention_acc_max": next((r["test_task_accuracy"] for r in all_intervention_results if r["mode"] == "simple" and r["num_interventions"] == max_interventions), None),
        "propagating_intervention_acc_max": next((r["test_task_accuracy"] for r in all_intervention_results if r["mode"] == "propagating" and r["num_interventions"] == max_interventions), None),
    }
    
    # =========================================================================
    # PFI (Permutation Feature Importance) — same as CREAM
    # =========================================================================
    num_concepts = config["hyperparameters_model2"]["num_concepts"]
    num_side = config["hyperparameters_model2"]["num_side_channel"]
    
    if num_side > 0:
        from src.PFI_accuracy import PFI_accuracies
        
        # Wrap SoftMaskedLinear so it can be called as model(x) without mask arg.
        # PFI and SAGE expect a standard nn.Module. We bake in the learned c2y mask.
        A_c2y_for_wrap = model.u_to_CY.graph_agg_c2y().detach()
        c2y_mask_frozen = model.u_to_CY._build_c2y_mask(A_c2y_for_wrap)
        
        class LastLayerWithMask(torch.nn.Module):
            def __init__(self, soft_masked_linear, mask):
                super().__init__()
                self.layer = soft_masked_linear
                self.mask = mask
            def forward(self, x):
                return self.layer(x, self.mask)
        
        last_layer_wrapped = LastLayerWithMask(model.u_to_CY.last_layer, c2y_mask_frozen)
        
        print("\nComputing PFI importances...")
        concept_dropped_score, side_dropped_score = PFI_accuracies(
            last_layer_wrapped, test_latent[0], num_concepts, repeat=100
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
            
            workers = config.get("dataset_params", {}).get("num_workers", 4)
            batch_size = config.get("dataset_params", {}).get("batch_size", 128)
            
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
                explained_model = nn.Sequential(last_layer_wrapped, nn.Sigmoid())
            else:
                explained_model = nn.Sequential(last_layer_wrapped, nn.Softmax(dim=1))
            
            twenty_pct = int(len(sage_df_train) * 0.2)
            imputer = sage.GroupedMarginalImputer(
                explained_model, train_x[:twenty_pct], test_groups
            )
            estimator = my_PermutationEstimator(
                imputer, "cross entropy", random_state=seed, n_jobs=workers
            )
            # max_time=3600: if not converged within 1 hour, return partial results
            sage_values = estimator(
                test_x, test_y, batch_size=batch_size, thresh=0.05,
                bar=False, max_time=3600
            )
            
            explanation_values = dict(zip(test_group_names, sage_values.values))
            cci = group_importance_metric(explanation_values)
            print(f"  CCI: {cci}")
            
            results["CCI"] = cci
            results["debugging_sage_metrics_concepts"] = explanation_values["concepts"]
            results["debugging_sage_metrics_side_channel"] = explanation_values["side_channel"]
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
