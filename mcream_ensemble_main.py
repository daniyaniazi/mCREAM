"""
Training script for mCREAM Ensemble.

M separate CREAM models (one per expert graph) trained jointly.
Predictions aggregated at the y level — not at the graph level.

Usage:
    python mcream_ensemble_main.py --config all_configs/mcream_configs/cfmnist/ensemble/deletion_M5_medium_weighted.yaml

Ensemble types:
    weighted  — y = sum_m pi_m * y_m  (pi learned end-to-end)
    average   — y = (1/M) sum y_m
    majority  — same logits as average; majority logic applies at argmax

Noise types (multi_expert.noise_type in config):
    single_action  — all M experts share ONE action (deletion/addition/reversal)
                     at ONE noise level (low/medium/high). Cleanest science.
    mixed          — old p_del+p_add+p_rev simultaneously
    structured     — named bias types (conservative/liberal/balanced)
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
from src.mcream_ensemble import mCREAM_Ensemble
from src.expert_graphs.generation import (
    load_expert_graphs,
    generate_expert_graphs_from_dag,
    generate_single_action_experts,
    generate_structured_experts,
    save_expert_graphs,
    load_and_split_dag,
    DISAGREEMENT_LEVELS,
    SINGLE_ACTION_NOISE_LEVELS,
)


# =============================================================================
# Expert graph loading / generation
# =============================================================================

def load_or_generate_expert_graphs(
    config: dict,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Load expert graphs from disk, or generate if not found.

    Dispatches on multi_expert.noise_type:
        'single_action' — one action type + one noise level, same for all M experts
        'mixed'         — simultaneous p_del + p_add + p_rev (old behaviour)
        'structured'    — named bias types (conservative/liberal/balanced)
    """
    expert_dir = Path(config["paths"].get("expert_graphs_dir", ""))

    if expert_dir.exists() and (expert_dir / "config.yaml").exists():
        print(f"Loading expert graphs from {expert_dir}")
        expert_u2c, expert_c2y, gen_config = load_expert_graphs(expert_dir)

        # Slice to num_experts from config — generate 10 once, run M=2/5/10 from same folder.
        M = config.get("multi_expert", {}).get("num_experts", len(expert_u2c))
        expert_u2c = expert_u2c[:M]
        expert_c2y = expert_c2y[:M]
        print(f"  Using {M} of {gen_config.get('num_experts', '?')} available expert graphs")

        gt_dir = expert_dir / "ground_truth"
        if gt_dir.exists():
            u2c_star = torch.load(gt_dir / "u2c_star.pt", weights_only=True)
            c2y_star = torch.load(gt_dir / "c2y_star.pt", weights_only=True)
        else:
            num_classes = config["hyperparameters_model2"]["num_classes"]
            u2c_star, c2y_star = load_and_split_dag(config["paths"]["DAG_file"], num_classes)

        return expert_u2c, expert_c2y, u2c_star, c2y_star

    print(f"Expert graphs not found at {expert_dir}, generating...")

    me = config.get("multi_expert", {})
    num_experts = me.get("num_experts", 5)
    num_classes = config["hyperparameters_model2"]["num_classes"]
    dag_path = config["paths"]["DAG_file"]
    seed = config.get("seed", 42)
    noise_type = me.get("noise_type", "single_action")

    if noise_type == "single_action":
        action = me.get("action", "deletion")          # deletion | addition | reversal
        noise_level = me.get("noise_level", "medium")  # low | medium | high
        print(f"  Single-action: action={action}, noise_level={noise_level}")
        expert_u2c, expert_c2y, u2c_star, c2y_star = generate_single_action_experts(
            dag_path=dag_path, num_classes=num_classes,
            num_experts=num_experts, action=action,
            noise_level=noise_level, base_seed=seed,
        )
        save_cfg = {
            "dag_path": str(dag_path), "num_classes": num_classes,
            "num_experts": num_experts, "noise_type": "single_action",
            "action": action, "noise_level": noise_level,
            **SINGLE_ACTION_NOISE_LEVELS[action][noise_level],
            "seed": seed,
        }

    elif noise_type == "structured":
        expert_types = me.get("expert_types", ["conservative", "liberal", "balanced"])
        print(f"  Structured bias types: {expert_types}")
        expert_u2c, expert_c2y, u2c_star, c2y_star = generate_structured_experts(
            dag_path=dag_path, num_classes=num_classes,
            expert_types=expert_types, base_seed=seed,
        )
        num_experts = len(expert_types)
        save_cfg = {
            "dag_path": str(dag_path), "num_classes": num_classes,
            "num_experts": num_experts, "noise_type": "structured",
            "expert_types": expert_types, "seed": seed,
        }

    else:  # mixed
        disagreement_level = me.get("disagreement_level", "medium")
        params = DISAGREEMENT_LEVELS[disagreement_level]
        print(f"  Mixed noise, level={disagreement_level}: {params}")
        expert_u2c, expert_c2y, u2c_star, c2y_star = generate_expert_graphs_from_dag(
            dag_path=dag_path, num_classes=num_classes,
            num_experts=num_experts, base_seed=seed, **params,
        )
        save_cfg = {
            "dag_path": str(dag_path), "num_classes": num_classes,
            "num_experts": num_experts, "noise_type": "mixed",
            "disagreement_level": disagreement_level, **params, "seed": seed,
        }

    save_expert_graphs(expert_u2c, expert_c2y, expert_dir, save_cfg)
    return expert_u2c, expert_c2y, u2c_star, c2y_star


# =============================================================================
# Model construction
# =============================================================================

def load_backbone(config: dict, dataset_name: str):
    model_name = config.get("backbone_model") or config.get("model_name")
    if model_name is None:
        raise ValueError("Config must specify 'backbone_model'")

    model_class = get_component_with_dicts("model", model_name)
    checkpoint_path = Path(config["paths"]["input_model_path"])
    backbone = model_class.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        dataset=dataset_name,
        frozen=config["hyperparameters"].get("frozen_model1", True),
    )
    print(f"Loaded backbone from: {checkpoint_path}")
    return backbone


def build_full_expert_graphs(
    expert_u2c: List[torch.Tensor],
    expert_c2y: List[torch.Tensor],
    K: int,
    T: int,
) -> List[torch.Tensor]:
    """
    Reconstruct full (K+T)x(K+T) causal graphs from split (u2c, c2y) tensors.
    This is the format CREAM's UtoY_model expects.
    """
    full_graphs = []
    for u2c, c2y in zip(expert_u2c, expert_c2y):
        g = torch.zeros(K + T, K + T, dtype=torch.bool)
        g[:K, :K] = u2c.bool()   # concept→concept block
        g[K:, :] = c2y.bool()    # task rows
        full_graphs.append(g)
    return full_graphs


def create_ensemble_model(
    config: dict,
    backbone: pl.LightningModule,
    expert_u2c: List[torch.Tensor],
    expert_c2y: List[torch.Tensor],
    mutually_exclusive_concepts: Optional[List],
) -> mCREAM_Ensemble:
    hparams = config["hyperparameters_model2"]
    K = hparams["num_concepts"]
    T = hparams["num_classes"]
    me = config.get("multi_expert", {})

    expert_mode  = me.get("expert_mode", "hard")      # 'hard' | 'soft_edge'
    ensemble_type = me.get("ensemble_type", "weighted") # 'average' | 'weighted'

    shared = dict(
        backbone=backbone,
        expert_mode=expert_mode,
        ensemble_type=ensemble_type,
        num_exogenous=hparams["num_exogenous"],
        num_concepts=K,
        num_side_channel=hparams.get("num_side_channel", 0),
        num_classes=T,
        learning_rate=config["hyperparameters"]["learning_rate"],
        lambda_weight=config["hyperparameters"]["lambda_weight"],
        previous_model_output_size=hparams.get("previous_model_output_size"),
        concept_representation=hparams.get("concept_representation", "soft"),
        side_dropout=hparams.get("side_dropout", True),
        dropout_prob=hparams.get("dropout_prob", 0.9),
        num_hidden_layers_in_maskedmlp=hparams.get("num_hidden_layers_in_maskedmlp", 0),
        mutually_exclusive_concepts=mutually_exclusive_concepts,
        frozen_backbone=config["hyperparameters"].get("frozen_model1", True),
    )

    if expert_mode == "hard":
        # Hard mode: pass full (K+T)×(K+T) graphs — binary mask baked into each CREAM
        full_graphs = build_full_expert_graphs(expert_u2c, expert_c2y, K, T)
        return mCREAM_Ensemble(expert_full_graphs=full_graphs, **shared)

    elif expert_mode == "soft_edge":
        # Soft-edge mode: pass u2c and c2y separately — each expert learns its own α
        return mCREAM_Ensemble(
            expert_u2c_graphs=expert_u2c,
            expert_c2y_graphs=expert_c2y,
            **shared,
        )

    else:
        raise ValueError(f"Unknown expert_mode '{expert_mode}'. Choose from: hard, soft_edge")


# =============================================================================
# Per-expert corruption logging
# =============================================================================

def log_expert_corruption(
    expert_u2c: List[torch.Tensor],
    expert_c2y: List[torch.Tensor],
    u2c_star: torch.Tensor,
    c2y_star: torch.Tensor,
) -> List[dict]:
    """
    For each expert compute:
      spurious edges  = edges in expert but NOT in ground truth  (false positives, addition noise)
      missing edges   = edges in ground truth but NOT in expert  (false negatives, deletion noise)
      total changed   = spurious + missing
    Reported separately for u2c (concept→concept) and c2y (concept→task) graphs.
    """
    stats = []
    for m, (u2c, c2y) in enumerate(zip(expert_u2c, expert_c2y)):
        def edge_stats(expert: torch.Tensor, gt: torch.Tensor):
            e = expert.bool()
            g = gt.bool()
            spurious = (~g &  e).sum().item()   # added edges (FP)
            missing  = ( g & ~e).sum().item()   # deleted edges (FN)
            total    = g.numel()
            gt_edges = g.sum().item()
            return spurious, missing, total, gt_edges

        sp_u2c, ms_u2c, tot_u2c, gt_u2c = edge_stats(u2c, u2c_star)
        sp_c2y, ms_c2y, tot_c2y, gt_c2y = edge_stats(c2y, c2y_star)

        print(f"    expert_{m}: "
              f"u2c spurious={sp_u2c} missing={ms_u2c}/{gt_u2c} edges | "
              f"c2y spurious={sp_c2y} missing={ms_c2y}/{gt_c2y} edges")

        stats.append({
            f"expert_{m}_u2c_spurious_edges":    sp_u2c,
            f"expert_{m}_u2c_missing_edges":     ms_u2c,
            f"expert_{m}_u2c_gt_edges":          gt_u2c,
            f"expert_{m}_u2c_corruption_pct":    (sp_u2c + ms_u2c) / tot_u2c * 100,
            f"expert_{m}_c2y_spurious_edges":    sp_c2y,
            f"expert_{m}_c2y_missing_edges":     ms_c2y,
            f"expert_{m}_c2y_gt_edges":          gt_c2y,
            f"expert_{m}_c2y_corruption_pct":    (sp_c2y + ms_c2y) / tot_c2y * 100,
        })
    return stats


# =============================================================================
# Single seed run
# =============================================================================

def run_single_seed(config: dict, config_path: Path, seed: int) -> dict:
    print(f"\n{'='*60}")
    print(f"mCREAM Ensemble  |  seed={seed}")
    print(f"Config: {config_path}")
    print(f"{'='*60}\n")

    pl.seed_everything(seed, workers=True)

    dataset_name = config["dataset_name"]
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
        print(f"Loaded {len(mutually_exclusive)} mutex concept groups")

    expert_u2c, expert_c2y, u2c_star, c2y_star = load_or_generate_expert_graphs(config)
    M = len(expert_u2c)
    print(f"Loaded {M} expert graphs")

    print("\n  Per-expert corruption (vs ground truth):")
    expert_corruption_stats = log_expert_corruption(expert_u2c, expert_c2y, u2c_star, c2y_star)

    backbone = load_backbone(config, dataset_name)

    me = config.get("multi_expert", {})
    print(f"\nBuilding mCREAM_Ensemble:")
    print(f"  M            = {M}")
    print(f"  expert_mode  = {me.get('expert_mode', 'hard')}")
    print(f"  ensemble_type= {me.get('ensemble_type', 'weighted')}")
    model = create_ensemble_model(config, backbone, expert_u2c, expert_c2y, mutually_exclusive)

    max_epochs = config["trainer_param"]["max_epochs"]
    experiment_name = config.get("experiment_name", config_path.stem)

    default_root_dir = (
        Path(config["paths"]["default_root_dir"])
        / dataset_name / config["mode"] / "mCREAM_Ensemble"
        / experiment_name / f"seed_{seed}"
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        default_root_dir=default_root_dir,
        deterministic=True,
        enable_progress_bar=True,
    )

    peak_gpu_memory_mb = 0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print(f"\nTraining for {max_epochs} epochs...")
    start_time = time.perf_counter()
    trainer.fit(model, datamodule=dataset)
    training_time = time.perf_counter() - start_time
    print(f"Training completed in {training_time/60:.2f} minutes")

    print("\nTesting...")
    test_start = time.perf_counter()
    trainer.test(model, datamodule=dataset)
    test_time = time.perf_counter() - test_start

    if torch.cuda.is_available():
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak GPU memory: {peak_gpu_memory_mb:.1f} MB")

    val_train_metrics = {
        key: value.item() for key, value in trainer.callback_metrics.items()
    }

    # =========================================================================
    # Benchmark
    # =========================================================================
    from src.utils import run_benchmark

    T = config["hyperparameters_model2"]["num_classes"]
    K = config["hyperparameters_model2"]["num_concepts"]

    class BenchmarkWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
            self.concept_loss_function = torch.nn.BCELoss()
            # CrossEntropyLoss expects integer targets [B], not one-hot [B, T].
            # We use BCEWithLogitsLoss with a float pseudo-target as a proxy
            # so the backward pass is valid regardless of num_classes.
            self.task_loss_function = torch.nn.BCEWithLogitsLoss()
        def forward(self, x):
            y, c = self.m(x)
            # Benchmark expects (y, c) with y shape [B, T] or [B, 1].
            # Flatten y to [B, T] so loss shapes are consistent.
            return y, c

    try:
        benchmark_results = run_benchmark(BenchmarkWrapper(model), dataset.test_dataloader())
    except Exception as e:
        print(f"  Benchmark failed: {e}")
        benchmark_results = {}

    # =========================================================================
    # Intermediate values (latent activations for SAGE/PFI)
    # =========================================================================
    pl_checkpoint_path = trainer.logger.log_dir
    from src.saving_intermediate_utils import save_intermediate_values

    print("\nSaving intermediate values...")
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

    # =========================================================================
    # Expert weights (weighted ensemble only)
    # =========================================================================
    expert_weights = model.get_expert_weights()
    if expert_weights is not None:
        print(f"\nLearned expert weights (π): {expert_weights.tolist()}")

    # =========================================================================
    # Edge reliabilities (soft_edge mode only)
    # =========================================================================
    edge_reliabilities = model.get_edge_reliabilities()
    if edge_reliabilities is not None:
        print("\nLearned edge reliabilities (alpha) per expert:")
        for m, (r_u2c, r_c2y) in enumerate(edge_reliabilities):
            if r_u2c is not None:
                print(f"  expert_{m}: u2c mean={r_u2c.mean():.3f}  min={r_u2c.min():.3f}  max={r_u2c.max():.3f}")

    # =========================================================================
    # Per-expert predictions on test set
    # Saves y_m for every expert + ensemble for every test sample.
    # Allows post-hoc analysis: which expert was right, where ensemble helped.
    # =========================================================================
    print("\nSaving per-expert predictions on test set...")
    device = next(model.parameters()).device
    dataset.setup(stage="test")
    expert_preds_df = model.collect_expert_predictions(
        dataset.test_dataloader(), device=device
    )

    preds_save_dir = Path(pl_checkpoint_path) / "expert_predictions"
    preds_save_dir.mkdir(parents=True, exist_ok=True)
    preds_path = preds_save_dir / "per_expert_predictions.csv"
    expert_preds_df.to_csv(preds_path, index=False)
    print(f"  Saved {len(expert_preds_df)} rows x {len(expert_preds_df.columns)} cols to: {preds_path}")

    # Quick per-expert accuracy summary
    print("  Per-expert accuracy:")
    for m in range(model.num_experts):
        acc_m = expert_preds_df[f"y_{m}_correct"].mean()
        print(f"    expert_{m}: {acc_m:.4f}")
    ensemble_acc = expert_preds_df["ensemble_correct"].mean()
    print(f"    ensemble:  {ensemble_acc:.4f}")

    # =========================================================================
    # Intervention curve
    # =========================================================================
    print("\nRunning intervention experiment...")
    from src.saving_intermediate_utils import save_activation_percentiles

    K = config["hyperparameters_model2"]["num_concepts"]
    concept_rep = config["hyperparameters_model2"].get("concept_representation", "soft")

    # Now that _compute_loss calls self(x) = forward(), the proxy hooks fire
    # during trainer.test(), so save_activation_percentiles works correctly.
    intervention_percentile_df = None
    if concept_rep in ("soft", "group_soft", "logits"):
        try:
            intervention_percentile_df = save_activation_percentiles(
                dataset=dataset, dataset_name=dataset_name, model=model,
                DAG_path=config["paths"]["DAG_file"],
            )
            if intervention_percentile_df is None or len(intervention_percentile_df) != K:
                print(f"  Percentile df has wrong size — using hard 0/1 targets.")
                intervention_percentile_df = None
        except Exception as e:
            print(f"  save_activation_percentiles failed: {e} — using hard 0/1 targets.")

    # per_expert_task_correct[m] accumulates correct booleans for expert m
    M = model.num_experts
    T = config["hyperparameters_model2"]["num_classes"]

    # Pre-compute per-expert edge stats so every intervention row carries them.
    # This links the "how broken is this graph" question directly to "how
    # does the intervention curve behave for this expert".
    def _edge_stats(expert_graph: torch.Tensor, gt: torch.Tensor):
        """Returns (spurious, missing, gt_edges) for concept columns only."""
        K_ = config["hyperparameters_model2"]["num_concepts"]
        # For c2y graphs [T×(K+T)], only the first K columns are concept→task
        if expert_graph.shape[0] != expert_graph.shape[1]:
            e = expert_graph[:, :K_].bool()
            g = gt[:, :K_].bool()
        else:
            e = expert_graph.bool()
            g = gt.bool()
        return (~g & e).sum().item(), (g & ~e).sum().item(), g.sum().item()

    expert_edge_info = []
    for m, (u2c, c2y) in enumerate(zip(expert_u2c, expert_c2y)):
        sp_u2c, ms_u2c, gt_u2c = _edge_stats(u2c, u2c_star)
        sp_c2y, ms_c2y, gt_c2y = _edge_stats(c2y, c2y_star)
        expert_edge_info.append({
            "expert": m,
            "u2c_spurious": sp_u2c, "u2c_missing": ms_u2c, "u2c_gt_edges": gt_u2c,
            "c2y_spurious": sp_c2y, "c2y_missing": ms_c2y, "c2y_gt_edges": gt_c2y,
            "total_spurious": sp_u2c + sp_c2y,
            "total_missing":  ms_u2c + ms_c2y,
        })

    intervention_results = []
    per_expert_intervention_results = []   # one row per (n_interv, expert)

    for n_interv in range(K + 1):
        model.eval()
        all_task_correct = []
        all_concept_correct = []
        # Per-expert accumulators: list of M lists
        expert_task_correct = [[] for _ in range(M)]

        dataset.setup(stage="test")
        with torch.no_grad():
            for batch in dataset.test_dataloader():
                x, true_concepts, y_true = batch
                if torch.cuda.is_available():
                    x, true_concepts, y_true = x.cuda(), true_concepts.cuda(), y_true.cuda()
                    model = model.cuda()

                interv_concepts = true_concepts.float()
                if intervention_percentile_df is not None:
                    p5 = torch.tensor(
                        intervention_percentile_df["5th_percentile"].values,
                        device=x.device, dtype=x.dtype
                    )
                    p95 = torch.tensor(
                        intervention_percentile_df["95th_percentile"].values,
                        device=x.device, dtype=x.dtype
                    )
                    interv_concepts = true_concepts.float() * p95 + (1 - true_concepts.float()) * p5

                # ── Ensemble intervention (existing) ─────────────────────────
                y_pred, c_pred = model.forward_with_interventions(x, interv_concepts, n_interv)

                if T == 1:
                    task_preds = (torch.sigmoid(y_pred) > 0.5).int().view(-1)
                else:
                    task_preds = y_pred.argmax(dim=1)

                all_task_correct.append((task_preds == y_true.view(-1)).float())
                all_concept_correct.append(((c_pred > 0.5) == true_concepts).float().mean(dim=1))

                # ── Per-expert intervention ───────────────────────────────────
                # Run each expert independently with the same intervention count
                u = model.backbone.concept_extractor(x)
                for m, expert in enumerate(model.experts):
                    if model.expert_mode == "hard":
                        y_m, c_m = expert.forward_with_interventions(
                            u, interv_concepts, n_interv
                        )
                    else:  # soft_edge
                        y_m, c_m, _ = expert.forward_with_interventions(
                            u, interv_concepts, n_interv
                        )
                    if T == 1:
                        preds_m = (torch.sigmoid(y_m) > 0.5).int().view(-1)
                    else:
                        preds_m = y_m.argmax(dim=1)
                    expert_task_correct[m].append((preds_m == y_true.view(-1)).float())

        # ── Aggregate ensemble results ────────────────────────────────────────
        task_acc    = torch.cat(all_task_correct).mean().item()
        concept_acc = torch.cat(all_concept_correct).mean().item()
        intervention_results.append({
            "num_interventions": n_interv,
            "test_task_accuracy": task_acc,
            "test_concept_accuracy": concept_acc,
        })

        # ── Aggregate per-expert results ──────────────────────────────────────
        for m in range(M):
            expert_acc = torch.cat(expert_task_correct[m]).mean().item()
            row = {
                "num_interventions": n_interv,
                "expert": m,
                "test_task_accuracy": expert_acc,
            }
            # Attach edge corruption info so notebook can plot both together
            row.update(expert_edge_info[m])
            per_expert_intervention_results.append(row)

        print(f"    interventions={n_interv}: ensemble={task_acc:.4f}  "
              f"experts=[{', '.join(f'{torch.cat(expert_task_correct[m]).mean().item():.3f}' for m in range(M))}]")

    interv_path = Path(pl_checkpoint_path) / "intervention_results.csv"
    pd.DataFrame(intervention_results).to_csv(interv_path, index=False)

    per_expert_interv_path = Path(pl_checkpoint_path) / "per_expert_intervention_results.csv"
    pd.DataFrame(per_expert_intervention_results).to_csv(per_expert_interv_path, index=False)
    print(f"  Saved per-expert intervention results to: {per_expert_interv_path}")

    # =========================================================================
    # PFI / SAGE / CCI  (same pipeline as CREAM, using first expert's last layer)
    # =========================================================================
    num_side = config["hyperparameters_model2"].get("num_side_channel", 0)
    results_pfi = {}

    if num_side > 0:
        from src.PFI_accuracy import PFI_accuracies

        first_expert = model.experts[0]

        class LastLayerForPFI(torch.nn.Module):
            def __init__(self, layer):
                super().__init__()
                self.layer = layer
            def forward(self, x):
                return self.layer(x)

        pfi_layer = LastLayerForPFI(first_expert.last_layer)

        print("\nComputing PFI importances...")
        try:
            concept_dropped_score, side_dropped_score = PFI_accuracies(
                pfi_layer, test_latent[0], K, repeat=100
            )
            test_acc = val_train_metrics.get("test_task_accuracy", 0.0)
            results_pfi["PFI_concept_importance"] = test_acc - concept_dropped_score
            results_pfi["PFI_side_importance"] = test_acc - side_dropped_score
            print(f"  PFI concept importance: {results_pfi['PFI_concept_importance']:.4f}")
            print(f"  PFI side importance: {results_pfi['PFI_side_importance']:.4f}")
        except Exception as e:
            print(f"  PFI failed: {e}")

        print("\nComputing SAGE / CCI...")
        try:
            import sage
            import torch.nn as tnn
            from src.sage_importance_functions import prepare_shap_data, group_importance_metric
            from src.diff_permutation_estimator import PermutationEstimator as my_PermutationEstimator

            train_x, _, train_group_names, train_groups = prepare_shap_data(train_latent[1])
            test_x, test_y, test_group_names, test_groups = prepare_shap_data(test_latent[1])
            train_x, test_x, test_y = train_x.to_numpy(), test_x.to_numpy(), test_y.to_numpy()

            T = config["hyperparameters_model2"]["num_classes"]
            explained_model = tnn.Sequential(
                pfi_layer,
                tnn.Softmax(dim=1) if T > 1 else tnn.Sigmoid(),
            )
            twenty_pct = int(len(train_latent[1]) * 0.2)
            imputer = sage.GroupedMarginalImputer(explained_model, train_x[:twenty_pct], test_groups)
            estimator = my_PermutationEstimator(
                imputer, "cross entropy", random_state=seed,
                n_jobs=config.get("dataset_params", {}).get("num_workers", 4)
            )
            sage_values = estimator(
                test_x, test_y,
                batch_size=config.get("dataset_params", {}).get("batch_size", 128),
                thresh=0.05, bar=False, max_time=3600,
            )
            explanation_values = dict(zip(test_group_names, sage_values.values))
            cci = group_importance_metric(explanation_values)
            print(f"  CCI: {cci:.4f}")
            results_pfi["CCI"] = cci
        except Exception as e:
            print(f"  SAGE/CCI failed: {e}")
            results_pfi["CCI"] = None

    # =========================================================================
    # Compile and save results
    # =========================================================================
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    results = {
        "max_epochs": max_epochs,
        "seed": seed,
        "experiment_name": config_path.stem,
        "train_val_time": training_time / 60.0,
        "test_time": test_time / 60.0,
        "num_trainable_parameters": num_params,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        **benchmark_results,
        **val_train_metrics,
        # mCREAM Ensemble fields
        "num_experts": M,
        "expert_mode": me.get("expert_mode", "hard"),
        "ensemble_type": me.get("ensemble_type", "weighted"),
        "noise_type": me.get("noise_type", "single_action"),
        "action": me.get("action", ""),
        "noise_level": me.get("noise_level", me.get("disagreement_level", "")),
        "expert_weights": expert_weights.tolist() if expert_weights is not None else None,
        "expert_predictions_path": str(preds_path),
        # Per-expert test accuracy (from predictions CSV)
        **{f"expert_{m}_test_accuracy": expert_preds_df[f"y_{m}_correct"].mean()
           for m in range(model.num_experts)},
        # Per-expert mean edge reliability (soft_edge mode only)
        **({
            f"expert_{m}_u2c_alpha_mean": r_u2c.mean().item() if r_u2c is not None else None
            for m, (r_u2c, _) in enumerate(edge_reliabilities)
        } if edge_reliabilities is not None else {}),
        # Per-expert corruption
        **{k: v for d in expert_corruption_stats for k, v in d.items()},
        # Intervention summary
        "intervention_acc_0": next(
            (r["test_task_accuracy"] for r in intervention_results if r["num_interventions"] == 0), None
        ),
        "intervention_acc_max": next(
            (r["test_task_accuracy"] for r in intervention_results if r["num_interventions"] == K), None
        ),
        **results_pfi,
    }

    pl_checkpoint_path = Path(trainer.logger.log_dir)
    dict_to_csv(results, pl_checkpoint_path, config_path)

    metrics_dir = (
        Path(config["paths"]["default_root_dir"])
        / "metrics" / dataset_name / "mCREAM_Ensemble"
    )
    metrics_dir.mkdir(parents=True, exist_ok=True)
    dict_to_csv(results, metrics_dir, config_path)
    print(f"\nResults saved to: {metrics_dir}")
    print(f"{'='*60}\n")

    return results


# =============================================================================
# Main (multi-seed)
# =============================================================================

def main(config_path: str):
    config = load_config(config_path)
    config_path = Path(config_path)

    seeds = config.get("seeds", None)
    if seeds is None:
        seeds = [config.get("seed", 42)]

    print(f"\n{'#'*60}")
    print(f"mCREAM Ensemble Multi-Seed Run  |  seeds={seeds}")
    print(f"{'#'*60}\n")

    all_results = []
    for i, seed in enumerate(seeds):
        print(f"\n>>> Seed {i+1}/{len(seeds)}: {seed}")
        result = run_single_seed({**config, "seed": seed}, config_path, seed)
        all_results.append(result)

    if len(seeds) > 1:
        print(f"\n{'#'*60}")
        print(f"MULTI-SEED SUMMARY ({len(seeds)} seeds)")
        print(f"{'#'*60}")

        df = pd.DataFrame(all_results)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        agg_cols = [c for c in numeric_cols
                    if c not in ("seed", "max_epochs", "num_experts", "num_trainable_parameters")]

        summary = {"experiment_name": config_path.stem, "num_seeds": len(seeds), "seeds": str(seeds)}
        for col in agg_cols:
            vals = df[col].dropna()
            if len(vals) > 0:
                summary[f"{col}_mean"] = vals.mean()
                summary[f"{col}_std"] = vals.std()
                print(f"  {col}: {vals.mean():.4f} ± {vals.std():.4f}")

        dataset_name = config["dataset_name"]
        metrics_dir = (
            Path(config["paths"]["default_root_dir"])
            / "metrics" / dataset_name / "mCREAM_Ensemble"
        )
        metrics_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(metrics_dir / f"{config_path.stem}_per_seed.csv", index=False)
        pd.DataFrame([summary]).to_csv(metrics_dir / f"{config_path.stem}_summary.csv", index=False)
        print(f"\nPer-seed:  {metrics_dir}/{config_path.stem}_per_seed.csv")
        print(f"Summary:   {metrics_dir}/{config_path.stem}_summary.csv")
        print(f"\n{'#'*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mCREAM Ensemble Training")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    torch.set_float32_matmul_precision("high")
    main(args.config)
