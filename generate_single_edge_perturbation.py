"""
Generate single-edge perturbation experiments for edge importance analysis.

For each existing c2y edge: create a DAG with that ONE edge removed.
For N random non-existing c2y edges: create a DAG with that ONE edge added.

suggestion 2: single-edge perturbation → CCI vs accuracy scatter.

Usage:
    python generate_single_edge_perturbation.py --dataset cfmnist
    python generate_single_edge_perturbation.py --dataset celeba
"""

import argparse
import pandas as pd
import numpy as np
import os
from pathlib import Path

DATASETS = {
    'cfmnist': {
        'dag':         'data/FashionMNIST/Complete_Concept_FMNIST_DAG.csv',
        'num_classes': 10,
        'num_concepts': 11,
        'output_dir':  'data/FashionMNIST/single_edge_perturbation',
        'config_dir':  'all_configs/mcream_configs/cfmnist/single_edge_perturbation',
        'dataset_name': 'Complete_Concept_FMNIST',
        'backbone':    'Standard_FashionMNIST',
        'num_exogenous': 128,
        'num_side':    40,
        'concept_rep': 'group_soft',
        'dropout':     0.9,
        'max_epochs':  50,
        'ckpt':        './pretrained_models/FMNIST/version_0/checkpoints/epoch=49-step=10750.ckpt',
        'softmax':     './data/FashionMNIST/mutually_exclusive_relationships_COMPLETE.json',
    },
    'celeba': {
        'dag':         'data/CelebA/final_DAG_unfair.csv',
        'num_classes': 1,
        'num_concepts': 7,
        'output_dir':  'data/CelebA/single_edge_perturbation',
        'config_dir':  'all_configs/mcream_configs/celeba/single_edge_perturbation',
        'dataset_name': 'CelebA',
        'backbone':    'Standard_CelebA',
        'num_exogenous': 75,
        'num_side':    5,
        'concept_rep': 'soft',
        'dropout':     0.1,
        'max_epochs':  20,
        'ckpt':        './pretrained_models/CelebA/version_11/checkpoints/epoch=89-step=6840.ckpt',
        'softmax':     None,
    },
}

CONFIG_TEMPLATE = """\
# Single-edge perturbation: {description}
# Supervisor suggestion 2: one edge changed → CCI vs accuracy scatter
mode: train_cbm
seed: 42
seeds: [42, 7, 1, 134, 89]
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


def generate_for_dataset(dataset_key: str, n_additions: int = 10, seed: int = 42):
    cfg = DATASETS[dataset_key]
    dag_path = Path(cfg['dag'])
    K = cfg['num_concepts']
    T = cfg['num_classes']

    # Load ground truth DAG
    gt_df = pd.read_csv(dag_path, index_col=0)
    node_names = list(gt_df.index)
    concept_names = node_names[:K]
    task_names    = node_names[K:]
    gt_vals = (gt_df.values != 0).astype(int)

    # c2y block: rows K..K+T, cols 0..K (concept columns only)
    c2y_gt = gt_vals[K:, :K]   # [T, K]

    output_dir = Path(cfg['output_dir'])
    config_dir = Path(cfg['config_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    prev_size = 128 if dataset_key == 'cfmnist' else 512
    celeba_extra = '\n  class_name: unfair' if dataset_key == 'celeba' else ''
    softmax_line = f'\n  softmax_mask: {cfg["softmax"]}' if cfg['softmax'] else ''

    configs_written = 0

    # ── Deletion: remove one existing c2y edge at a time ─────────────────────
    existing_edges = [(t, c) for t in range(T) for c in range(K) if c2y_gt[t, c] == 1]
    print(f'\n{dataset_key}: {len(existing_edges)} existing c2y edges → deletion experiments')

    for t, c in existing_edges:
        task_name    = task_names[t]
        concept_name = concept_names[c]
        exp_name     = f'del_edge_{concept_name}_{task_name}'.replace(' ', '_').replace('/', '_')

        # Build perturbed DAG
        perturbed = gt_vals.copy()
        perturbed[K + t, c] = 0   # remove this c2y edge

        dag_save_path = output_dir / f'{exp_name}.csv'
        pd.DataFrame(perturbed, index=node_names, columns=node_names).to_csv(dag_save_path)

        # Write config
        config_content = CONFIG_TEMPLATE.format(
            description=f'delete c2y edge {concept_name}→{task_name}',
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
            dag_file=f'./{dag_save_path}',
            ckpt=cfg['ckpt'],
            softmax_line=softmax_line,
        )
        (config_dir / f'{exp_name}.yaml').write_text(config_content, encoding='utf-8')
        configs_written += 1

    # ── Addition: add one non-existing c2y edge at a time ────────────────────
    absent_edges = [(t, c) for t in range(T) for c in range(K) if c2y_gt[t, c] == 0]
    rng = np.random.default_rng(seed)
    selected_additions = rng.choice(len(absent_edges),
                                    size=min(n_additions, len(absent_edges)),
                                    replace=False)
    print(f'{dataset_key}: {len(absent_edges)} absent c2y edges, sampling {len(selected_additions)} for addition experiments')

    for idx in selected_additions:
        t, c = absent_edges[idx]
        task_name    = task_names[t]
        concept_name = concept_names[c]
        exp_name     = f'add_edge_{concept_name}_{task_name}'.replace(' ', '_').replace('/', '_')

        perturbed = gt_vals.copy()
        perturbed[K + t, c] = 1   # add this spurious c2y edge

        dag_save_path = output_dir / f'{exp_name}.csv'
        pd.DataFrame(perturbed, index=node_names, columns=node_names).to_csv(dag_save_path)

        config_content = CONFIG_TEMPLATE.format(
            description=f'add spurious c2y edge {concept_name}→{task_name}',
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
            dag_file=f'./{dag_save_path}',
            ckpt=cfg['ckpt'],
            softmax_line=softmax_line,
        )
        (config_dir / f'{exp_name}.yaml').write_text(config_content, encoding='utf-8')
        configs_written += 1

    print(f'{dataset_key}: wrote {configs_written} configs to {config_dir}')
    return configs_written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cfmnist', 'celeba', 'all'], default='all')
    parser.add_argument('--n_additions', type=int, default=10,
                        help='Number of random addition experiments (default: 10)')
    args = parser.parse_args()

    datasets = list(DATASETS.keys()) if args.dataset == 'all' else [args.dataset]
    total = 0
    for ds in datasets:
        total += generate_for_dataset(ds, n_additions=args.n_additions)

    print(f'\nTotal configs written: {total}')
    print('Run with: python simple_main.py --config <config_path>')
    print('Or generate server scripts with: python gen_single_edge_scripts.py')


if __name__ == '__main__':
    main()
