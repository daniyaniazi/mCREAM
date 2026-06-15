"""
Generate mCREAM Graph Ensemble configs for single-edge perturbation.

Same as standalone CREAM single-edge perturbation (scenario 2),
but using mCREAM_GraphEnsemble with M experts all sharing the same perturbed graph.

Direct comparison:
  standalone CREAM: 1 model, graph with 1 edge wrong
  GraphEnsemble:    M models, all with same 1 edge wrong

Both produce one (accuracy, CCI) point per edge. Compare the two scatter plots.

Usage:
    python generate_graph_ensemble_single_edge_configs.py --dataset cfmnist
"""

import argparse
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
# mCREAM Graph Ensemble - single-edge perturbation: {description}
# All {M} experts use the SAME perturbed graph.
# Same as standalone CREAM single-edge, but with M graph module experts.
mode: train_cbm
seed: 42
experiment_name: {exp_name}
dataset_name: {dataset_name}

dataset_params:
  batch_size: 256
  workers: 2
  return_labels: true
  return_images: true

backbone_model: {backbone}

multi_expert:
  num_experts: {M}
  noise_type: single_edge_perturbation

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
  DAG_file: ./{perturbed_dag}
  gt_dag_file: {gt_dag}
  expert_graphs_dir: ./data/FashionMNIST/expert_graphs/graph_ensemble_single_edge/{exp_name}/
  input_model_path: {ckpt}
  softmax_mask: {softmax}
"""


def generate(dataset_key: str, M: int = 5):
    cfg = DATASETS[dataset_key]
    perturb_dir = Path(cfg['perturb_dir'])
    config_dir  = Path(cfg['config_dir'])
    config_dir.mkdir(parents=True, exist_ok=True)

    if not perturb_dir.exists():
        print(f'Run generate_single_edge_perturbation.py first — {perturb_dir} not found')
        return 0

    all_csvs = sorted(list(perturb_dir.glob('del_edge_*.csv')) +
                      list(perturb_dir.glob('add_edge_*.csv')))
    print(f'{dataset_key}: found {len(all_csvs)} perturbed DAGs')

    count = 0
    for csv_f in all_csvs:
        name = csv_f.stem   # e.g. del_edge_Tops_Clothes
        exp_name = f'gensingle_{name}'
        description = name.replace('_', ' ')

        content = CONFIG_TEMPLATE.format(
            description=description,
            exp_name=exp_name,
            M=M,
            dataset_name=cfg['dataset_name'],
            backbone=cfg['backbone'],
            num_classes=cfg['num_classes'],
            num_concepts=cfg['num_concepts'],
            num_exogenous=cfg['num_exogenous'],
            num_side=cfg['num_side'],
            concept_rep=cfg['concept_rep'],
            dropout=cfg['dropout'],
            max_epochs=cfg['max_epochs'],
            prev_size=cfg['prev_size'],
            perturbed_dag=str(csv_f),
            gt_dag=cfg['dag'],
            ckpt=cfg['ckpt'],
            softmax=cfg['softmax'],
        )
        (config_dir / f'{exp_name}.yaml').write_text(content, encoding='utf-8')
        count += 1

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
