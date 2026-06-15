"""
Training script for mCREAM Graph Module Ensemble.

Architecture: M Concept-Concept blocks (one per expert graph),
shared backbone / side-channel / concept-task head.
Aggregation at CONCEPT level (not prediction level).

Usage:
    python mcream_graph_ensemble_main.py --config all_configs/mcream_graph_ensemble_configs/cfmnist/deletion/average_deletion_medium.yaml

Difference from mcream_ensemble_main.py:
    mcream_ensemble_main:         each expert is a full CREAM, aggregate y
    mcream_graph_ensemble_main:   only u2c block is duplicated, aggregate c
"""

import argparse
import pytorch_lightning as pl
import torch
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import get_component_with_dicts, load_config, dict_to_csv
from src.mcream_graph_ensemble import mCREAM_GraphEnsemble
from src.expert_graphs.generation import (
    load_expert_graphs,
    generate_single_action_experts,
    generate_structured_experts,
    generate_expert_graphs_from_dag,
    save_expert_graphs,
    load_and_split_dag,
    DISAGREEMENT_LEVELS,
    SINGLE_ACTION_NOISE_LEVELS,
)


# =============================================================================
# Expert graph loading — identical to mcream_ensemble_main.py
# =============================================================================

def load_or_generate_expert_graphs(config):
    expert_dir = Path(config["paths"].get("expert_graphs_dir", ""))

    if expert_dir.exists() and (expert_dir / "config.yaml").exists():
        print(f"Loading expert graphs from {expert_dir}")
        expert_u2c, expert_c2y, gen_config = load_expert_graphs(expert_dir)
        M = config.get("multi_expert", {}).get("num_experts", len(expert_u2c))
        expert_u2c = expert_u2c[:M]
        expert_c2y = expert_c2y[:M]
        print(f"  Using {M} of {gen_config.get('num_experts','?')} available expert graphs")

        gt_dir = expert_dir / "ground_truth"
        if gt_dir.exists():
            u2c_star = torch.load(gt_dir / "u2c_star.pt", weights_only=True)
            c2y_star = torch.load(gt_dir / "c2y_star.pt", weights_only=True)
        else:
            num_classes = config["hyperparameters_model2"]["num_classes"]
            u2c_star, c2y_star = load_and_split_dag(config["paths"]["DAG_file"], num_classes)

        return expert_u2c, expert_c2y, u2c_star, c2y_star

    print(f"Expert graphs not found at {expert_dir}, generating...")
    me  = config.get("multi_expert", {})
    M   = me.get("num_experts", 5)
    num_classes = config["hyperparameters_model2"]["num_classes"]
    dag_path    = config["paths"]["DAG_file"]
    seed        = config.get("seed", 42)
    noise_type  = me.get("noise_type", "single_action")

    if noise_type == "single_edge_perturbation":
        # ALL M experts use the SAME single-edge perturbed graph (dag_path).
        # Same as standalone CREAM single-edge perturbation, but with M experts.
        # Direct comparison: CREAM vs GraphEnsemble on identical graph error.
        #
        # dag_path = the perturbed DAG (e.g. del_edge_Tops_Clothes.csv)
        # gt_dag_file = original GT DAG for reference/corruption stats
        gt_dag = config["paths"].get("gt_dag_file",
                 "data/FashionMNIST/Complete_Concept_FMNIST_DAG.csv")
        u2c_perturbed, c2y_perturbed = load_and_split_dag(dag_path, num_classes)
        u2c_star, c2y_star           = load_and_split_dag(gt_dag,   num_classes)

        # All M experts see the same perturbed graph
        expert_u2c = [u2c_perturbed.clone() for _ in range(M)]
        expert_c2y = [c2y_perturbed.clone() for _ in range(M)]

        save_cfg = {"dag_path": str(dag_path), "noise_type": "single_edge_perturbation",
                    "num_experts": M, "seed": seed}
        save_expert_graphs(expert_u2c, expert_c2y, expert_dir, save_cfg)
        return expert_u2c, expert_c2y, u2c_star, c2y_star

    elif noise_type == "edge_count_multi_seed":
        # Each expert gets a different seed's graph — all with the same edge count.
        # expert_dag_files: list of M DAG CSV paths (one per seed/expert).
        expert_dag_files = me.get("expert_dag_files", [])
        if not expert_dag_files:
            raise ValueError("edge_count_multi_seed requires expert_dag_files list in config")
        expert_u2c, expert_c2y = [], []
        for dag_f in expert_dag_files[:M]:
            u2c_m, c2y_m = load_and_split_dag(dag_f, num_classes)
            expert_u2c.append(u2c_m)
            expert_c2y.append(c2y_m)
        u2c_star, c2y_star = load_and_split_dag(dag_path, num_classes)
        save_cfg = {"dag_path": str(dag_path), "noise_type": "edge_count_multi_seed",
                    "num_experts": len(expert_u2c), "seed": seed}
        save_expert_graphs(expert_u2c, expert_c2y, expert_dir, save_cfg)
        return expert_u2c, expert_c2y, u2c_star, c2y_star

    elif noise_type == "single_action":
        action = me.get("action", "deletion")
        level  = me.get("noise_level", "medium")
        expert_u2c, expert_c2y, u2c_star, c2y_star = generate_single_action_experts(
            dag_path=dag_path, num_classes=num_classes,
            num_experts=M, action=action, noise_level=level, base_seed=seed,
        )
        save_cfg = {"dag_path": str(dag_path), "num_classes": num_classes,
                    "num_experts": M, "noise_type": "single_action",
                    "action": action, "noise_level": level,
                    **SINGLE_ACTION_NOISE_LEVELS[action][level], "seed": seed}
    else:
        disagreement_level = me.get("disagreement_level", "medium")
        params = DISAGREEMENT_LEVELS[disagreement_level]
        expert_u2c, expert_c2y, u2c_star, c2y_star = generate_expert_graphs_from_dag(
            dag_path=dag_path, num_classes=num_classes,
            num_experts=M, base_seed=seed, **params,
        )
        save_cfg = {"dag_path": str(dag_path), "num_classes": num_classes,
                    "num_experts": M, "noise_type": "mixed",
                    "disagreement_level": disagreement_level, **params, "seed": seed}

    save_expert_graphs(expert_u2c, expert_c2y, expert_dir, save_cfg)
    return expert_u2c, expert_c2y, u2c_star, c2y_star


def build_full_expert_graphs(expert_u2c, expert_c2y, K, T):
    full = []
    for u2c, c2y in zip(expert_u2c, expert_c2y):
        g = torch.zeros(K + T, K + T, dtype=torch.bool)
        g[:K, :K] = u2c.bool()
        g[K:, :]  = c2y.bool()
        full.append(g)
    return full


def load_backbone(config, dataset_name):
    model_name = config.get("backbone_model") or config.get("model_name")
    model_class = get_component_with_dicts("model", model_name)
    checkpoint_path = Path(config["paths"]["input_model_path"])
    backbone = model_class.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        dataset=dataset_name,
        frozen=config["hyperparameters"].get("frozen_model1", True),
    )
    print(f"Loaded backbone from: {checkpoint_path}")
    return backbone


# =============================================================================
# Run single seed
# =============================================================================

def run_single_seed(config, config_path, seed):
    print(f"\n{'='*60}")
    print(f"mCREAM Graph Ensemble  |  seed={seed}")
    print(f"Config: {config_path}")
    print(f"{'='*60}\n")

    pl.seed_everything(seed, workers=True)

    dataset_name  = config["dataset_name"]
    dataset_class = get_component_with_dicts("dataset", dataset_name)
    if "FMNIST" in dataset_name:
        dataset = dataset_class(
            **config["dataset_params"], seed=seed,
            full_concepts=(dataset_name == "Complete_Concept_FMNIST"),
        )
    else:
        dataset = dataset_class(**config["dataset_params"])

    mutually_exclusive = None
    if "softmax_mask" in config["paths"]:
        with open(config["paths"]["softmax_mask"], "r") as f:
            mutually_exclusive = json.load(f)

    expert_u2c, expert_c2y, u2c_star, c2y_star = load_or_generate_expert_graphs(config)
    M = len(expert_u2c)
    K = config["hyperparameters_model2"]["num_concepts"]
    T = config["hyperparameters_model2"]["num_classes"]
    print(f"Loaded {M} expert graphs")

    # Build full (K+T)x(K+T) graphs for experts
    expert_full = build_full_expert_graphs(expert_u2c, expert_c2y, K, T)

    # Reference graph for shared last_layer — use GT DAG
    import pandas as _pd
    gt_df  = _pd.read_csv(config["paths"]["DAG_file"], index_col=0)
    ref_graph = torch.tensor((gt_df.values != 0), dtype=torch.bool)

    backbone = load_backbone(config, dataset_name)
    hparams  = config["hyperparameters_model2"]
    me       = config.get("multi_expert", {})

    model = mCREAM_GraphEnsemble(
        backbone=backbone,
        expert_graphs=expert_full,
        ref_graph=ref_graph,
        num_exogenous=hparams["num_exogenous"],
        num_concepts=K,
        num_side_channel=hparams.get("num_side_channel", 0),
        num_classes=T,
        learning_rate=config["hyperparameters"]["learning_rate"],
        lambda_weight=config["hyperparameters"]["lambda_weight"],
        previous_model_output_size=hparams.get("previous_model_output_size"),
        concept_representation=hparams.get("concept_representation", "group_soft"),
        side_dropout=hparams.get("side_dropout", True),
        dropout_prob=hparams.get("dropout_prob", 0.9),
        num_hidden_layers_in_maskedmlp=hparams.get("num_hidden_layers_in_maskedmlp", 0),
        mutually_exclusive_concepts=mutually_exclusive,
        frozen_backbone=config["hyperparameters"].get("frozen_model1", True),
    )

    experiment_name = config.get("experiment_name", config_path.stem)
    default_root_dir = (
        Path(config["paths"]["default_root_dir"])
        / dataset_name / config["mode"] / "mCREAM_GraphEnsemble"
        / experiment_name / f"seed_{seed}"
    )

    trainer = pl.Trainer(
        max_epochs=config["trainer_param"]["max_epochs"],
        default_root_dir=default_root_dir,
        deterministic=True,
        enable_progress_bar=True,
    )

    peak_gpu = 0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print(f"Training {config['trainer_param']['max_epochs']} epochs...")
    start = time.perf_counter()
    trainer.fit(model, datamodule=dataset)
    training_time = time.perf_counter() - start
    print(f"Done in {training_time/60:.2f} min")

    print("Testing...")
    trainer.test(model, datamodule=dataset)

    if torch.cuda.is_available():
        peak_gpu = torch.cuda.max_memory_allocated() / (1024**2)

    val_train_metrics = {k: v.item() for k, v in trainer.callback_metrics.items()}
    pl_checkpoint_path = trainer.logger.log_dir

    # Save intermediate values for SAGE/PFI
    from src.saving_intermediate_utils import save_intermediate_values
    train_latent = save_intermediate_values(
        dataset=dataset, dataset_name=dataset_name, model=model,
        training_set=True, DAG_path=config["paths"]["DAG_file"],
        seed=seed, save_directory=pl_checkpoint_path,
    )
    test_latent = save_intermediate_values(
        dataset=dataset, dataset_name=dataset_name, model=model,
        training_set=False, DAG_path=config["paths"]["DAG_file"],
        seed=seed, save_directory=pl_checkpoint_path,
    )

    # Intervention curve
    print("Running interventions...")
    intervention_results = []
    for n_interv in range(K + 1):
        model.eval()
        all_task, all_concept = [], []
        dataset.setup(stage="test")
        with torch.no_grad():
            for batch in dataset.test_dataloader():
                x, true_concepts, y_true = batch
                if torch.cuda.is_available():
                    x, true_concepts, y_true = x.cuda(), true_concepts.cuda(), y_true.cuda()
                    model = model.cuda()
                # Use concept_extractor directly — same as mCREAM_GraphEnsemble.forward
                features = model.x_to_u.concept_extractor(x)   # [B, 128]
                y_pred, c_pred = model.u_to_CY.forward_with_interventions(
                    features, true_concepts, n_interv
                )
                if T == 1:
                    task_preds = (torch.sigmoid(y_pred) > 0.5).int().view(-1)
                else:
                    task_preds = y_pred.argmax(dim=1)
                all_task.append((task_preds == y_true.view(-1)).float())
                all_concept.append(((c_pred > 0.5) == true_concepts).float().mean(dim=1))
        intervention_results.append({
            "num_interventions": n_interv,
            "test_task_accuracy": torch.cat(all_task).mean().item(),
            "test_concept_accuracy": torch.cat(all_concept).mean().item(),
        })
        print(f"    n={n_interv}: acc={intervention_results[-1]['test_task_accuracy']:.4f}")

    pd.DataFrame(intervention_results).to_csv(
        Path(pl_checkpoint_path) / "intervention_results.csv", index=False)

    # Per-expert corruption stats
    expert_corruption_stats = []
    for m, (u2c, c2y) in enumerate(zip(expert_u2c, expert_c2y)):
        pct_u2c = (u2c.bool() != u2c_star.bool()).sum().item() / u2c_star.numel() * 100
        pct_c2y = (c2y.bool() != c2y_star.bool()).sum().item() / c2y_star.numel() * 100
        expert_corruption_stats.append({
            f"expert_{m}_u2c_corruption_pct": pct_u2c,
            f"expert_{m}_c2y_corruption_pct": pct_c2y,
        })

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    results = {
        "max_epochs":           config["trainer_param"]["max_epochs"],
        "seed":                 seed,
        "experiment_name":      config_path.stem,
        "train_val_time":       training_time / 60.0,
        "num_trainable_params": num_params,
        "peak_gpu_memory_mb":   peak_gpu,
        "model_type":           "mCREAM_GraphEnsemble",
        "num_experts":          M,
        "noise_type":           me.get("noise_type", "single_action"),
        "action":               me.get("action", ""),
        "noise_level":          me.get("noise_level", ""),
        **val_train_metrics,
        **{k: v for d in expert_corruption_stats for k, v in d.items()},
        "intervention_acc_0":   intervention_results[0]["test_task_accuracy"],
        "intervention_acc_max": intervention_results[-1]["test_task_accuracy"],
        # Learned λ_m weights per expert (Kavya: multi-task learning style)
        **{f"lambda_{m}": model.u_to_CY.expert_weights[m].item()
           for m in range(M)},
    }

    pl_checkpoint_path = Path(trainer.logger.log_dir)
    dict_to_csv(results, pl_checkpoint_path, config_path)

    metrics_dir = (
        Path(config["paths"]["default_root_dir"])
        / "metrics" / dataset_name / "mCREAM_GraphEnsemble"
    )
    metrics_dir.mkdir(parents=True, exist_ok=True)
    dict_to_csv(results, metrics_dir, config_path)
    print(f"Results saved to: {metrics_dir}")
    return results


# =============================================================================
# Main
# =============================================================================

def main(config_path):
    config      = load_config(config_path)
    config_path = Path(config_path)

    seeds = config.get("seeds", [config.get("seed", 42)])
    if isinstance(seeds, int):
        seeds = [seeds]

    all_results = []
    for i, seed in enumerate(seeds):
        print(f"\n>>> Seed {i+1}/{len(seeds)}: {seed}")
        result = run_single_seed({**config, "seed": seed}, config_path, seed)
        all_results.append(result)

    if len(seeds) > 1:
        df = pd.DataFrame(all_results)
        numeric = [c for c in df.select_dtypes(include=[np.number]).columns
                   if c not in ("seed", "max_epochs")]
        print(f"\nMULTI-SEED SUMMARY ({len(seeds)} seeds)")
        for col in numeric:
            vals = df[col].dropna()
            if len(vals) > 0:
                print(f"  {col}: {vals.mean():.4f} ± {vals.std():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    torch.set_float32_matmul_precision("high")
    main(args.config)
