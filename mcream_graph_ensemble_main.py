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

    elif noise_type == "mixed_noise_levels":
        # Each expert gets a graph from a DIFFERENT noise level directory.
        # e.g. expert_0=low, expert_1=medium, expert_2=high, expert_3=low, expert_4=high
        # → experts have meaningfully different BCE values → λ can learn
        # Config specifies expert_graph_paths: list of (dir, expert_idx) pairs
        expert_graph_paths = me.get("expert_graph_paths", [])
        if not expert_graph_paths:
            raise ValueError("mixed_noise_levels requires expert_graph_paths list")
        expert_u2c, expert_c2y = [], []
        for entry in expert_graph_paths[:M]:
            graph_dir   = Path(entry["dir"])
            expert_idx  = entry.get("expert_idx", 0)
            u2c_f = graph_dir / "u2c" / f"expert_{expert_idx}.pt"
            c2y_f = graph_dir / "c2y" / f"expert_{expert_idx}.pt"
            expert_u2c.append(torch.load(u2c_f, weights_only=True).float())
            expert_c2y.append(torch.load(c2y_f, weights_only=True).float())
        u2c_star, c2y_star = load_and_split_dag(dag_path, num_classes)
        save_cfg = {"noise_type": "mixed_noise_levels", "num_experts": len(expert_u2c)}
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

    elif noise_type == "consensus":
        # Exp4: consensus-based experts — shared base noise + private per-expert flip
        # graphs pre-generated by generate_consensus_expert_graphs.py
        # expert_graphs_dir = data/.../expert_graphs/consensus/{level}/
        level = me.get("noise_level", "medium")
        # graphs already saved by generate_consensus_expert_graphs.py — load them
        gt_dir = expert_dir / "ground_truth"
        if gt_dir.exists():
            for m in range(M):
                expert_u2c.append(torch.load(expert_dir/"u2c"/f"expert_{m}.pt", weights_only=True).float())
                expert_c2y.append(torch.load(expert_dir/"c2y"/f"expert_{m}.pt", weights_only=True).float())
            u2c_star = torch.load(gt_dir/"u2c_star.pt", weights_only=True)
            c2y_star = torch.load(gt_dir/"c2y_star.pt", weights_only=True)
            return expert_u2c, expert_c2y, u2c_star, c2y_star
        else:
            raise FileNotFoundError(
                f"Consensus graphs not found at {expert_dir}. "
                f"Run: python generate_consensus_expert_graphs.py --dataset cfmnist"
            )

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
        use_alpha=hparams.get("use_alpha", False),
        alpha_l1_weight=hparams.get("alpha_l1_weight", 0.001),
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

    # ── Intervention curve — mirrors CREAM's simple_main.py exactly ─────────
    # CREAM sets model.intervention_percentile_df then calls trainer.test()
    # for each num_interventions. We do the same so the percentile scaling
    # and group_interventions logic are handled by the inherited CREAM code.
    from src.saving_intermediate_utils import save_activation_percentiles
    from json import load as json_load

    concept_rep = config["hyperparameters_model2"].get("concept_representation", "soft")
    if concept_rep in ("soft", "group_soft", "logits"):
        try:
            intervention_percentile_df = save_activation_percentiles(
                dataset=dataset, dataset_name=dataset_name, model=model,
                DAG_path=config["paths"]["DAG_file"],
            )
            print(f"  Percentile df: {type(intervention_percentile_df)}, "
                  f"len={len(intervention_percentile_df) if intervention_percentile_df is not None else 'None'}, "
                  f"cols={list(intervention_percentile_df.columns) if intervention_percentile_df is not None and len(intervention_percentile_df)>0 else 'empty'}")
            # Only store if valid — has correct columns and K rows
            if (intervention_percentile_df is not None
                    and len(intervention_percentile_df) == K
                    and "5th_percentile" in intervention_percentile_df.columns):
                model.intervention_percentile_df = intervention_percentile_df
                print(f"  Percentile scaling ENABLED ({K} concepts, "
                      f"p5 range: {intervention_percentile_df['5th_percentile'].min():.3f} to {intervention_percentile_df['5th_percentile'].max():.3f})")
            else:
                print("  WARNING: Percentile df invalid — computing manually from forward pass")
                # Compute percentiles directly from model activations (no hooks needed)
                model.eval()
                all_c = []
                dataset.setup(stage="fit")
                with torch.no_grad():
                    for batch in dataset.train_dataloader():
                        x_b, _, _ = batch
                        if torch.cuda.is_available():
                            x_b = x_b.cuda(); model_tmp = model.cuda()
                        else:
                            model_tmp = model
                        _, c_b = model_tmp(x_b)
                        all_c.append(c_b.cpu())
                all_c_np = torch.cat(all_c, dim=0).numpy()  # [N, K]
                perc_rows = []
                for k in range(K):
                    perc_rows.append({
                        "dimension": f"c_dim_{k}",
                        "5th_percentile":  float(np.percentile(all_c_np[:, k], 5)),
                        "95th_percentile": float(np.percentile(all_c_np[:, k], 95)),
                    })
                model.intervention_percentile_df = pd.DataFrame(perc_rows)
                print(f"  Manual percentile scaling ENABLED")
        except Exception as e:
            print(f"  Percentile computation failed: {e} — using hard 0/1 targets")

    # Enable intervention debug logging (writes to separate file)
    model.u_to_CY._debug_interventions = True
    model.u_to_CY._debug_log_path = str(
        Path(pl_checkpoint_path) / "intervention_debug_graph_ensemble.txt"
    )
    import os; os.makedirs(os.path.dirname(model.u_to_CY._debug_log_path), exist_ok=True)

    print("Running interventions...")
    intervention_results = []
    interv_trainer = pl.Trainer(
        max_epochs=config["trainer_param"]["max_epochs"],
        default_root_dir=default_root_dir,
        enable_progress_bar=False,
        deterministic=True,
        logger=False,
    )

    for group_interventions in (True, False):
        if group_interventions is True and "softmax_mask" in config["paths"]:
            with open(config["paths"]["softmax_mask"], "r") as f:
                softmax_mask = json_load(f)
            total_interventions = len(softmax_mask)
        elif group_interventions is True:
            continue  # no softmax_mask → skip group interventions
        else:
            total_interventions = K

        model.u_to_CY.group_interventions = group_interventions

        for n_interv in range(total_interventions + 1):
            model.num_interventions = n_interv
            model.interventions     = True
            interv_trainer.test(model, dataloaders=dataset, verbose=False)
            test_metrics = {k: v.item() for k, v in interv_trainer.callback_metrics.items()}
            intervention_results.append({
                "group_interventions": group_interventions,
                "num_interventions":   n_interv,
                "test_task_accuracy":  test_metrics.get("test_task_accuracy", float("nan")),
                "test_concept_accuracy": test_metrics.get("test_concept_accuracy", float("nan")),
            })
            print(f"    group={group_interventions} n={n_interv}: "
                  f"acc={intervention_results[-1]['test_task_accuracy']:.4f}")

    model.interventions = False

    pd.DataFrame(intervention_results).to_csv(
        Path(pl_checkpoint_path) / "intervention_results.csv", index=False)

    # ── PFI + SAGE/CCI ────────────────────────────────────────────────────────
    results_pfi = {}
    num_side = config["hyperparameters_model2"].get("num_side_channel", 0)
    if num_side > 0:
        from src.PFI_accuracy import PFI_accuracies
        from src.sage_importance_functions import prepare_shap_data, group_importance_metric
        from src.diff_permutation_estimator import PermutationEstimator as my_PermutationEstimator
        import sage as sage_lib
        import torch.nn as tnn
        from src.utils import timeout

        class LastLayerWrap(tnn.Module):
            def __init__(self, layer): super().__init__(); self.layer = layer
            def forward(self, x): return self.layer(x)

        pfi_layer = LastLayerWrap(model.u_to_CY.last_layer)

        print("\nComputing PFI...")
        try:
            concept_dropped, side_dropped = PFI_accuracies(
                pfi_layer, test_latent[0], K, repeat=100
            )
            test_acc = val_train_metrics.get("test_task_accuracy", 0.0)
            results_pfi["PFI_concept_importance"] = float(test_acc - concept_dropped)
            results_pfi["PFI_side_importance"]    = float(test_acc - side_dropped)
            print(f"  PFI concept={results_pfi['PFI_concept_importance']:.4f}  side={results_pfi['PFI_side_importance']:.4f}")
        except Exception as e:
            print(f"  PFI failed: {e}")

        print("\nComputing SAGE/CCI (1hr timeout)...")
        try:
            train_x, _, grp_names, grp_idx = prepare_shap_data(train_latent[1])
            test_x, test_y, _, _           = prepare_shap_data(test_latent[1])
            train_x, test_x, test_y = train_x.to_numpy(), test_x.to_numpy(), test_y.to_numpy()

            T_cls = config["hyperparameters_model2"]["num_classes"]
            explained = tnn.Sequential(
                pfi_layer,
                tnn.Softmax(dim=1) if T_cls > 1 else tnn.Sigmoid(),
            )
            twenty = int(len(train_latent[1]) * 0.2)
            imputer = sage_lib.GroupedMarginalImputer(explained, train_x[:twenty], grp_idx)
            estimator = my_PermutationEstimator(
                imputer, "cross entropy", random_state=seed,
                n_jobs=config.get("dataset_params", {}).get("num_workers", 4)
            )

            @timeout(3600)
            def run_sage():
                return estimator(
                    test_x, test_y,
                    batch_size=config.get("dataset_params", {}).get("batch_size", 128),
                    thresh=0.05, bar=False,
                )

            sv = run_sage()
            expl = dict(zip(grp_names, sv.values))
            cci  = group_importance_metric(expl)
            results_pfi["CCI"]                              = float(cci)
            results_pfi["debugging_sage_metrics_concepts"]  = expl.get("concepts")
            results_pfi["debugging_sage_metrics_side"]      = expl.get("side_channel")
            print(f"  CCI={cci:.4f}")
        except Exception as e:
            print(f"  SAGE/CCI failed: {e}")
            results_pfi["CCI"] = None

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
        # Learned λ_m weights per expert
        **{f"lambda_{m}": model.u_to_CY.expert_weights[m].item() for m in range(M)},
        # PFI + CCI
        **results_pfi,
    }

    pl_checkpoint_path = Path(trainer.logger.log_dir)
    dict_to_csv(results, pl_checkpoint_path, config_path)

    # ── Save alpha matrix if use_alpha=True ──────────────────────────────────
    # alpha_prob [K, K]: learned edge importance probabilities in (0,1)
    # Saved as CSV with concept names as row/col headers for interpretability
    if getattr(model.u_to_CY, 'use_alpha', False):
        alpha_np = model.u_to_CY.alpha_prob.detach().cpu().numpy()  # [K, K]

        # Try to get concept names from DAG file
        dag_path = config["paths"].get("DAG_file", None)
        if dag_path and Path(dag_path).exists():
            dag_df     = pd.read_csv(dag_path, index_col=0)
            num_classes = config["hyperparameters_model2"]["num_classes"]
            concept_names = list(dag_df.columns)[:-num_classes]
        else:
            concept_names = [f"c{k}" for k in range(alpha_np.shape[0])]

        alpha_df = pd.DataFrame(
            alpha_np,
            index=concept_names,    # rows = source concept (i)
            columns=concept_names,  # cols = target concept (j)
        )
        # alpha_df[i,j] = importance of edge i→j

        alpha_path = pl_checkpoint_path / f"alpha_prob_seed{seed}.csv"
        alpha_df.to_csv(alpha_path, float_format="%.6f")
        print(f"Alpha matrix saved: {alpha_path}")

        # Also add per-edge summary stats to main results dict
        results["alpha_mean"]    = float(alpha_np.mean())
        results["alpha_max"]     = float(alpha_np.max())
        results["alpha_sparsity"] = float((alpha_np < 0.1).mean())  # fraction near-zero

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
