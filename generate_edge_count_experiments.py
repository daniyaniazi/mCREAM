"""
Scenario 1: Edge Count vs Performance Analysis.

For each target edge count, generate N_SEEDS graphs with exactly that
many edges (randomly chosen which to add/remove), train CREAM on each.
Plot accuracy boxplot per edge count.

Usage:
    python generate_edge_count_experiments.py --dataset cfmnist
    python generate_edge_count_experiments.py --dataset celeba
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

DATASETS = {
    'cfmnist': {
        'dag':          'data/FashionMNIST/Complete_Concept_FMNIST_DAG.csv',
        'num_classes':  10,
        'num_concepts': 11,
        'output_dir':   'data/FashionMNIST/edge_count_experiments',
        'config_dir':   'all_configs/mcream_configs/cfmnist/edge_count_experiments',
        'dataset_name': 'Complete_Concept_FMNIST',
        'backbone':     'Standard_FashionMNIST',
        'num_exogenous': 128,
        'num_side':     40,
        'concept_rep':  'group_soft',
        'dropout':      0.9,
        'max_epochs':   50,
        'ckpt':         './pretrained_models/FMNIST/version_0/checkpoints/epoch=49-step=10750.ckpt',
        'softmax':      './data/FashionMNIST/mutually_exclusive_relationships_COMPLETE.json',
        # Vary u2c edges (17 in GT) — she said 14,15,16,17,18,19,20
        'target_counts': [14, 15, 16, 17, 18, 19, 20],
        'graph_type':   'u2c',   # which graph block to vary
    },
    'celeba': {
        'dag':          'data/CelebA/final_DAG_unfair.csv',
        'num_classes':  1,
        'num_concepts': 7,
        'output_dir':   'data/CelebA/edge_count_experiments',
        'config_dir':   'all_configs/mcream_configs/celeba/edge_count_experiments',
        'dataset_name': 'CelebA',
        'backbone':     'Standard_CelebA',
        'num_exogenous': 75,
        'num_side':     5,
        'concept_rep':  'soft',
        'dropout':      0.1,
        'max_epochs':   20,
        'ckpt':         './pretrained_models/CelebA/version_11/checkpoints/epoch=89-step=6840.ckpt',
        'softmax':      None,
        'target_counts': [5, 6, 7, 8, 9, 10],
        'graph_type':   'u2c',
    },
}

CONFIG_TEMPLATE = """\
# Edge count experiment: {graph_type} target={target_count} edges, seed={seed}
# Scenario 1: fixed edge count, random which edges → boxplot per count
mode: train_cbm
seed: 42
experiment_name: {exp_name}
dataset_name: {dataset_name}

dataset_params:
  batch_size: 256
  workers: 2
  return_labels: true
  return_images: true{celeba_extra}

model_name: {backbone}

hyperparameters_model2:
  num_classes: {num_classes}
  num_exogenous: {num_exogenous}
  num_side_channel: {num_side}
  num_concepts: {num_concepts}
  masking_algorithm: zuko
  num_hidden_layers_in_maskedmlp: 0
  previous_model_output_size: {prev_size}
  last_layer_mask: true
  concept_representation: {concept_rep}
  side_dropout: true
  dropout_prob: {dropout}

hyperparameters:
  learning_rate: 0.001
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


def generate_for_dataset(dataset_key: str, n_seeds: int = 5):
    cfg = DATASETS[dataset_key]
    dag_path = Path(cfg['dag'])
    K = cfg['num_concepts']
    T = cfg['num_classes']

    gt_df = pd.read_csv(dag_path, index_col=0)
    node_names   = list(gt_df.index)
    concept_names = node_names[:K]
    task_names    = node_names[K:]
    gt_vals = (gt_df.values != 0).astype(int)

    # Which block to vary
    if cfg['graph_type'] == 'u2c':
        # u2c: rows 0..K-1, cols 0..K-1 (square, no diagonal)
        block_rows = slice(0, K)
        block_cols = slice(0, K)
        # Existing and absent positions (exclude diagonal — no self-loops)
        existing = [(r, c) for r in range(K) for c in range(K)
                    if r != c and gt_vals[r, c] == 1]
        absent   = [(r, c) for r in range(K) for c in range(K)
                    if r != c and gt_vals[r, c] == 0]
        gt_count = len(existing)
    else:
        # c2y: rows K..K+T, cols 0..K
        existing = [(K+t, c) for t in range(T) for c in range(K)
                    if gt_vals[K+t, c] == 1]
        absent   = [(K+t, c) for t in range(T) for c in range(K)
                    if gt_vals[K+t, c] == 0]
        gt_count = len(existing)

    output_dir = Path(cfg['output_dir'])
    config_dir = Path(cfg['config_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    prev_size    = 128 if dataset_key == 'cfmnist' else 512
    celeba_extra = '\n  class_name: unfair' if dataset_key == 'celeba' else ''
    softmax_line = f'\n  softmax_mask: {cfg["softmax"]}' if cfg['softmax'] else ''

    print(f'\n{dataset_key}: GT {cfg["graph_type"]} edges = {gt_count}')
    print(f'  Target counts: {cfg["target_counts"]}')
    print(f'  Seeds per count: {n_seeds}')
    print(f'  Total experiments: {len(cfg["target_counts"]) * n_seeds}')

    configs_written = 0

    for target_count in cfg['target_counts']:
        diff = target_count - gt_count

        for seed in range(n_seeds):
            rng = np.random.default_rng(seed * 1000 + target_count)
            perturbed = gt_vals.copy()

            if diff == 0:
                # GT graph — no changes
                description = f'GT ({gt_count} edges)'
            elif diff < 0:
                # Deletion: remove exactly |diff| random existing edges
                n_remove = abs(diff)
                if n_remove > len(existing):
                    print(f'  WARNING: cannot remove {n_remove} from {len(existing)} edges, skipping')
                    continue
                to_remove = rng.choice(len(existing), size=n_remove, replace=False)
                for idx in to_remove:
                    r, c = existing[idx]
                    perturbed[r, c] = 0
                description = f'delete {n_remove} edges (count={target_count})'
            else:
                # Addition: add exactly diff random absent edges
                n_add = diff
                if n_add > len(absent):
                    print(f'  WARNING: cannot add {n_add} to {len(absent)} absent, skipping')
                    continue
                to_add = rng.choice(len(absent), size=n_add, replace=False)
                for idx in to_add:
                    r, c = absent[idx]
                    perturbed[r, c] = 1
                description = f'add {n_add} edges (count={target_count})'

            exp_name = (f'edge_count_{cfg["graph_type"]}'
                        f'_{target_count}edges_seed{seed}')
            dag_save = output_dir / f'{exp_name}.csv'

            pd.DataFrame(perturbed,
                         index=node_names,
                         columns=node_names).to_csv(dag_save)

            config_content = CONFIG_TEMPLATE.format(
                graph_type=cfg['graph_type'],
                target_count=target_count,
                seed=seed,
                description=description,
                exp_name=exp_name,
                dataset_name=cfg['dataset_name'],
                celeba_extra=celeba_extra,
                backbone=cfg['backbone'],
                num_classes=T,
                num_exogenous=cfg['num_exogenous'],
                num_side=cfg['num_side'],
                num_concepts=K,
                prev_size=prev_size,
                concept_rep=cfg['concept_rep'],
                dropout=cfg['dropout'],
                max_epochs=cfg['max_epochs'],
                dag_file=f'./{dag_save}',
                ckpt=cfg['ckpt'],
                softmax_line=softmax_line,
            )
            (config_dir / f'{exp_name}.yaml').write_text(
                config_content, encoding='utf-8')
            configs_written += 1

    print(f'{dataset_key}: wrote {configs_written} configs to {config_dir}')
    return configs_written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cfmnist', 'celeba', 'all'],
                        default='all')
    parser.add_argument('--n_seeds', type=int, default=5,
                        help='Random seeds per edge count (default: 5 → boxplot)')
    parser.add_argument('--delta', type=int, default=5,
                        help='Range around GT: GT-delta to GT+delta (default: 5)')
    args = parser.parse_args()

    datasets = list(DATASETS.keys()) if args.dataset == 'all' else [args.dataset]

    # Override target_counts using delta — computed from actual GT edge count
    for ds_key in datasets:
        cfg = DATASETS[ds_key]
        gt_df = pd.read_csv(cfg['dag'], index_col=0)
        K = cfg['num_concepts']
        gt_vals = (gt_df.values != 0).astype(int)
        # u2c edges (no diagonal)
        gt_count = sum(1 for r in range(K) for c in range(K)
                       if r != c and gt_vals[r, c] == 1)
        cfg['target_counts'] = list(range(
            max(0, gt_count - args.delta),
            gt_count + args.delta + 1
        ))
        print(f'{ds_key}: GT u2c edges = {gt_count}  range {cfg["target_counts"]}')

    total = 0
    for ds in datasets:
        total += generate_for_dataset(ds, n_seeds=args.n_seeds)

    print(f'\nTotal configs: {total}  ({len(DATASETS[datasets[0]]["target_counts"])} counts × {args.n_seeds} seeds)')
    print('Submit: for f in all_configs/mcream_configs/cfmnist/edge_count_experiments/*.yaml; do')
    print('          python simple_main.py --config "$f"; done')


if __name__ == '__main__':
    main()
