"""
Generate CREAM configs from learned alpha matrices (Exp4 → alpha graph recovery).

For each consensus experiment (low/medium/high):
  1. Load all alpha_prob_seed*.csv files for that experiment
  2. Average alpha across seeds → mean_alpha [K, K]
  3. Threshold at `threshold` (default 0.5) → binary mask [K, K]
  4. Merge with GT DAG: replace u2c block with alpha-recovered binary mask
  5. Save as CSV → CREAM can load it as a normal DAG
  6. Write a CREAM config pointing to that DAG

Usage:
  python generate_alpha_recovered_graphs.py --dataset cfmnist --threshold 0.5
  python generate_alpha_recovered_graphs.py --dataset celeba  --threshold 0.4
  python generate_alpha_recovered_graphs.py --dataset cub     --threshold 0.5
  python generate_alpha_recovered_graphs.py --dataset all     --threshold 0.5
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


# ── Dataset configs ───────────────────────────────────────────────────────────
DATASETS = {
    "cfmnist": {
        "experiments_dir": "experiments/Complete_Concept_FMNIST/train_cbm/mCREAM_GraphEnsemble",
        "gt_dag":          "data/FashionMNIST/Complete_Concept_FMNIST_DAG.csv",
        "num_classes":     10,
        "num_concepts":    11,
        "output_dag_dir":  "data/FashionMNIST/alpha_recovered_dags",
        "config_dir":      "all_configs/cream_alpha_cfmnist",
        "dataset_name":    "Complete_Concept_FMNIST",
        "model_name":      "Standard_FashionMNIST",
        "ckpt":            "./pretrained_models/FMNIST/version_0/checkpoints/epoch=49-step=10750.ckpt",
        "softmax_mask":    "./data/FashionMNIST/mutually_exclusive_relationships_COMPLETE.json",
        "concept_rep": "group_soft",
        "workers": 2,
        "hparams": {
            "num_classes": 10, "num_exogenous": 128, "num_side_channel": 40,
            "num_concepts": 11, "dropout_prob": 0.8, "max_epochs": 50,
            "learning_rate": 0.001, "batch_size": 256,
        },
    },
    "celeba": {
        "experiments_dir": "experiments/CelebA/train_cbm/mCREAM_GraphEnsemble",
        "gt_dag":          "data/CelebA/final_DAG_unfair.csv",
        "num_classes":     1,
        "num_concepts":    7,
        "output_dag_dir":  "data/CelebA/alpha_recovered_dags",
        "config_dir":      "all_configs/cream_alpha_celeba",
        "dataset_name":    "CelebA",
        "model_name":      "Standard_CelebA",
        "ckpt":            "./pretrained_models/CelebA/version_11/checkpoints/epoch=89-step=6840.ckpt",
        "softmax_mask":    None,
        "concept_rep": "soft",   # CelebA: no mutex groups → soft not group_soft
        "workers": 2,
        "hparams": {
            "num_classes": 1, "num_exogenous": 75, "num_side_channel": 5,
            "num_concepts": 7, "dropout_prob": 0.1, "max_epochs": 20,
            "learning_rate": 0.001, "batch_size": 256,
        },
    },
    "cub": {
        "experiments_dir": "experiments/CUB/train_cbm/mCREAM_GraphEnsemble",
        "gt_dag":          "data/CUB/CUB_DAG_only_Gc.csv",
        "num_classes":     200,
        "num_concepts":    112,
        "output_dag_dir":  "data/CUB/alpha_recovered_dags",
        "config_dir":      "all_configs/cream_alpha_cub",
        "dataset_name":    "CUB",
        "model_name":      "Standard_CUB",
        "ckpt":            "./pretrained_models/CUB/version_1/checkpoints/epoch=49-step=3750.ckpt",
        "softmax_mask":    "./data/CUB/CUB_mutually_exclusive_concepts.json",
        "concept_rep": "group_soft",
        "workers": 0,   # CUB: workers=0 to avoid too many open files (113 intervention calls)
        "hparams": {
            "num_classes": 200, "num_exogenous": 648, "num_side_channel": 200,
            "num_concepts": 112, "dropout_prob": 0.8, "max_epochs": 300,
            "learning_rate": 0.0001, "batch_size": 64,
        },
    },
}

LEVELS = ["low", "medium", "high"]

CONFIG_TEMPLATE = """\
# CREAM on alpha-recovered graph — {dataset} {level} noise
# Alpha threshold: {threshold}  |  Edges recovered: {n_edges} / {n_gt_edges} GT edges
mode: train_cbm
seed: 42
experiment_name: cream_alpha_{level}
dataset_name: {dataset_name}

dataset_params:
  batch_size: {batch_size}
  workers: {workers}
  return_labels: true
  return_images: true{celeba_extra}

model_name: {model_name}

hyperparameters_model2:
  num_classes: {num_classes}
  num_exogenous: {num_exogenous}
  num_side_channel: {num_side_channel}
  num_concepts: {num_concepts}
  masking_algorithm: zuko
  num_hidden_layers_in_maskedmlp: 0
  previous_model_output_size: {prev_size}
  last_layer_mask: true
  concept_representation: {concept_rep}
  side_dropout: true
  dropout_prob: {dropout_prob}

hyperparameters:
  learning_rate: {learning_rate}
  lambda_weight: 1
  frozen_model1: true

trainer_param:
  max_epochs: {max_epochs}

paths:
  default_root_dir: ./experiments/
  metric_dir: ./last_metrics/
  DAG_file: {dag_file}
  input_model_path: {ckpt}{softmax_line}
"""


def load_alpha_csvs(exp_dir: Path, level: str) -> np.ndarray | None:
    """Load all alpha_prob_seed*.csv for one experiment level, return mean [K,K]."""
    exp_name = f"graph_ensemble_consensus_{level}"
    # Also check dynamic variant: graph_ensemble_consensus_dynamic_{level}
    dynamic_name  = f"graph_ensemble_consensus_dynamic_{level}"
    consensus_dir = exp_dir / exp_name
    if not consensus_dir.exists() and (exp_dir / dynamic_name).exists():
        consensus_dir = exp_dir / dynamic_name
        exp_name = dynamic_name
    if not consensus_dir.exists():
        print(f"  [SKIP] Not found: {consensus_dir}")
        return None, None

    alphas = []
    concept_names = None
    for seed_dir in sorted(consensus_dir.glob("seed_*/lightning_logs/version_*")):
        for alpha_f in sorted(seed_dir.glob("alpha_prob_seed*.csv")):
            df = pd.read_csv(alpha_f, index_col=0)
            alphas.append(df.values)
            if concept_names is None:
                concept_names = list(df.index)

    if not alphas:
        print(f"  [SKIP] No alpha CSVs found in {consensus_dir}")
        return None, None

    mean_alpha = np.stack(alphas, axis=0).mean(axis=0)
    print(f"  Loaded {len(alphas)} alpha seeds for {level}, shape={mean_alpha.shape}")
    return mean_alpha, concept_names


def alpha_to_dag_csv(mean_alpha: np.ndarray, threshold: float,
                     gt_dag_path: str, num_classes: int, num_concepts: int) -> pd.DataFrame:
    """Replace u2c block of GT DAG with thresholded alpha, keep c2y from GT."""
    gt_df   = pd.read_csv(gt_dag_path, index_col=0)
    gt_vals = gt_df.values.astype(bool)

    # Binarise alpha
    alpha_binary = (mean_alpha >= threshold).astype(bool)  # [K, K]

    # Replace u2c block (top-left [K,K]) with alpha-recovered mask
    recovered = gt_vals.copy()
    recovered[:num_concepts, :num_concepts] = alpha_binary

    return pd.DataFrame(recovered, index=gt_df.index, columns=gt_df.columns)


def process_dataset(ds_name: str, threshold: float):
    cfg = DATASETS[ds_name]
    exp_dir    = Path(cfg["experiments_dir"])
    gt_dag     = cfg["gt_dag"]
    K          = cfg["num_concepts"]
    T          = cfg["num_classes"]
    dag_outdir = Path(cfg["output_dag_dir"])
    config_dir = Path(cfg["config_dir"])
    dag_outdir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    # GT u2c edge count for reference
    gt_df   = pd.read_csv(gt_dag, index_col=0)
    gt_u2c  = gt_df.values.astype(bool)[:K, :K]
    n_gt    = int(gt_u2c.sum())

    print(f"\n{'='*55}")
    print(f"Dataset: {ds_name}  |  GT u2c edges: {n_gt}  |  threshold: {threshold}")
    print(f"{'='*55}")

    for level in LEVELS:
        mean_alpha, concept_names = load_alpha_csvs(exp_dir, level)
        if mean_alpha is None:
            continue

        # Stats
        n_above = int((mean_alpha >= threshold).sum())
        n_gt_present_above = int(((mean_alpha >= threshold) & gt_u2c).sum())
        precision = n_gt_present_above / max(n_above, 1)
        recall    = n_gt_present_above / max(n_gt, 1)
        print(f"  {level}: α≥{threshold} → {n_above} edges  "
              f"(GT recall={recall:.1%}, precision={precision:.1%})")

        # Save recovered DAG CSV
        dag_df   = alpha_to_dag_csv(mean_alpha, threshold, gt_dag, T, K)
        dag_path = dag_outdir / f"alpha_dag_{level}_t{threshold:.2f}.csv"
        dag_df.to_csv(dag_path)
        print(f"  Saved DAG: {dag_path}")

        # Save thresholded alpha visualisation CSV (with concept names)
        if concept_names:
            alpha_bin_df = pd.DataFrame(
                (mean_alpha >= threshold).astype(int),
                index=concept_names, columns=concept_names
            )
            alpha_bin_df.to_csv(dag_outdir / f"alpha_binary_{level}_t{threshold:.2f}.csv")

        # Write CREAM config
        h = cfg["hparams"]
        celeba_extra = "\n  class_name: unfair" if ds_name == "celeba" else ""
        softmax_line = (f"\n  softmax_mask: {cfg['softmax_mask']}"
                        if cfg["softmax_mask"] else "")
        prev_size = {"cfmnist": 128, "celeba": 512, "cub": 512}[ds_name]

        config_str = CONFIG_TEMPLATE.format(
            dataset=ds_name, level=level, threshold=threshold,
            n_edges=n_above, n_gt_edges=n_gt,
            dataset_name=cfg["dataset_name"],
            model_name=cfg["model_name"],
            batch_size=h["batch_size"],
            workers=cfg.get("workers", 2),
            celeba_extra=celeba_extra,
            concept_rep=cfg.get("concept_rep", "group_soft"),
            num_classes=h["num_classes"],
            num_exogenous=h["num_exogenous"],
            num_side_channel=h["num_side_channel"],
            num_concepts=h["num_concepts"],
            prev_size=prev_size,
            dropout_prob=h["dropout_prob"],
            learning_rate=h["learning_rate"],
            max_epochs=h["max_epochs"],
            dag_file=f"./{dag_path}",
            ckpt=cfg["ckpt"],
            softmax_line=softmax_line,
        )

        config_path = config_dir / f"cream_alpha_{level}.yaml"
        config_path.write_text(config_str)
        print(f"  Saved config: {config_path}")

    print(f"Done: {ds_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   choices=["cfmnist","celeba","cub","all"], default="cfmnist")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Alpha binarisation threshold (default 0.5)")
    parser.add_argument("--source",    choices=["static","dynamic","auto"], default="auto",
                        help="Load alpha from static (consensus_0.9) or dynamic experiment. "
                             "auto = use dynamic if exists, else static (default)")
    args = parser.parse_args()

    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        process_dataset(ds, args.threshold)


if __name__ == "__main__":
    main()
