"""
Generate mCREAM Graph Ensemble configs for single-edge perturbation.

Each expert gets a DIFFERENT single-edge perturbed graph.
One config = one group of 5 experts, each missing/adding a different edge.

Groups generated:
  del_group_0: experts 0-4 each missing a different deletion edge
  del_group_1: experts 0-4 each missing a different 5 deletion edges (next batch)
  ...
  add_group_0: experts 0-4 each adding a different spurious edge

Scientific question:
  If each expert is missing a DIFFERENT edge, does c_avg recover GT performance?
  Hypothesis: yes — collectively experts cover all edges.

Usage:
    python generate_graph_ensemble_single_edge_configs.py --dataset cfmnist
"""

import argparse
import glob
import os
from pathlib import Path

DATASETS = {
    'cfmnist': {
        'dag':        './data/FashionMNIST/Complete_Concept_FMNIST_DAG.csv',
        'perturb_dir':'data/FashionMNIST/single_edge_perturbation',
        'config_dir': 'all_configs/mcream_graph_ensemble_configs/cfmnist/single_edge',
        'dataset_name': 'Complete_Concept_FMNIST',
        'backbone':   'Standard_FashionMNIST',
        'num_classes': 10, 'num_concepts': 11,
        'num_exogenous': 128, 'num_side': 40,
        'concept_rep': 'group_soft', 'dropout': 0.9, 'max_epochs': 50,
        'ckpt': './pretrained_models/FMNIST/version_0/checkpoints/epoch=49-step=10750.ckpt',
        'softmax': './data/FashionMNIST/mutually_exclusive_relationships_COMPLETE.json',
        'prev_size': 128,
    },
}

CONFIG_TEMPLATE = """\
# mCREAM Graph Ensemble - single-edge perturbation group: {group_name}
# Each expert has a DIFFERENT single-edge perturbed graph.
# Expert m is missing/adding edge m — collectively all M edges are covered.
# Hypothesis: c_avg recovers GT performance when experts cover different edges.
mode: train_cbm
seed: 42
experiment_name: gensingle_{group_name}
dataset_name: {dataset_name}

dataset_params:
  batch_size: 256
  workers: 2
  return_labels: true
  return_images: true

backbone_model: {backbone}

multi_expert:
  num_experts: {M}
  noise_type: edge_count_multi_seed
  expert_dag_files:
{dag_lines}

hyperparameters_model2:
  num_classes: {num_classes}
  num_concepts: {num_concepts}
  num_exogenous: {num_exogenous}
  num_side_channel: {num_side}
  concept_representation: {concept_rep}
  num_hidden_layers_in_maskedmlp: 0
  previous_model_output_size: {prev_size}
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
  DAG_file: {dag}
  expert_graphs_dir: ./data/FashionMNIST/expert_graphs/graph_ensemble_single_edge/{group_name}/
  input_model_path: {ckpt}
  softmax_mask: {softmax}
"""


def generate(dataset_key: str, M: int = 5):
    cfg = DATASETS[dataset_key]
    perturb_dir = Path(cfg['perturb_dir'])
    config_dir  = Path(cfg['config_dir'])
    config_dir.mkdir(parents=True, exist_ok=True)

    if not perturb_dir.exists():
        print(f'Run first: python generate_single_edge_perturbation.py --dataset {dataset_key}')
        return 0

    del_csvs = sorted(perturb_dir.glob('del_edge_*.csv'))
    add_csvs = sorted(perturb_dir.glob('add_edge_*.csv'))
    print(f'{dataset_key}: {len(del_csvs)} deletion, {len(add_csvs)} addition DAGs')

    count = 0

    def write_group(group_name, dag_files):
        dag_lines = '\n'.join(f'    - ./{f}' for f in dag_files)
        content = CONFIG_TEMPLATE.format(
            group_name=group_name, M=len(dag_files),
            dag_lines=dag_lines,
            dataset_name=cfg['dataset_name'], backbone=cfg['backbone'],
            num_classes=cfg['num_classes'], num_concepts=cfg['num_concepts'],
            num_exogenous=cfg['num_exogenous'], num_side=cfg['num_side'],
            concept_rep=cfg['concept_rep'], dropout=cfg['dropout'],
            max_epochs=cfg['max_epochs'], prev_size=cfg['prev_size'],
            dag=cfg['dag'], ckpt=cfg['ckpt'], softmax=cfg['softmax'],
        )
        (config_dir / f'gensingle_{group_name}.yaml').write_text(content, encoding='utf-8')
        return 1

    # Deletion groups: consecutive batches of M experts, each missing a different edge
    for start in range(0, len(del_csvs), M):
        batch = del_csvs[start:start + M]
        if len(batch) < M:
            break   # skip incomplete groups
        group_name = f'del_group_{start // M}'
        count += write_group(group_name, [str(f) for f in batch])
        edge_names = [f.stem.replace('del_edge_','') for f in batch]
        print(f'  {group_name}: experts missing {edge_names}')

    # Addition groups
    for start in range(0, len(add_csvs), M):
        batch = add_csvs[start:start + M]
        if len(batch) < M:
            break
        group_name = f'add_group_{start // M}'
        count += write_group(group_name, [str(f) for f in batch])
        edge_names = [f.stem.replace('add_edge_','') for f in batch]
        print(f'  {group_name}: experts adding {edge_names}')

    print(f'Written {count} configs to {config_dir}')
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cfmnist', 'all'], default='cfmnist')
    parser.add_argument('--M', type=int, default=5)
    args = parser.parse_args()
    datasets = list(DATASETS.keys()) if args.dataset == 'all' else [args.dataset]
    for ds in datasets:
        generate(ds, M=args.M)


if __name__ == '__main__':
    main()
