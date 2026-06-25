"""
Generate mCREAM Graph Ensemble configs for CelebA edge count experiments.

GT u2c edges = 10, range = 5..15 (±5).
Each expert gets a DIFFERENT seed's graph at the same edge count.

Usage:
    python gen_graph_ensemble_edge_count_configs_celeba.py
"""

import os, glob, re
from pathlib import Path
from collections import defaultdict

src_dir = 'all_configs/mcream_configs/celeba/edge_count_experiments'
dst_dir = 'all_configs/mcream_graph_ensemble_configs/celeba/edge_count'
os.makedirs(dst_dir, exist_ok=True)

CONFIG_TEMPLATE = """\
# mCREAM Graph Ensemble - CelebA - edge count: {graph_type} {count} edges (GT=10)
# Expert m uses seed-m graph — same count, different random edge selection.
mode: train_cbm
seed: 42
seeds: [42, 7, 1, 134, 89]
experiment_name: gensemble_edge_count_{graph_type}_{count}edges
dataset_name: CelebA

dataset_params:
  batch_size: 256
  workers: 2
  return_labels: true
  return_images: true
  class_name: unfair

backbone_model: Standard_CelebA

multi_expert:
  num_experts: {M}
  noise_type: edge_count_multi_seed
  expert_dag_files:
{dag_lines}

hyperparameters_model2:
  num_classes: 1
  num_concepts: 7
  num_exogenous: 75
  num_side_channel: 5
  concept_representation: soft
  num_hidden_layers_in_maskedmlp: 0
  previous_model_output_size: 512
  side_dropout: true
  dropout_prob: 0.1

hyperparameters:
  learning_rate: 0.001
  lambda_weight: 1
  frozen_model1: true

trainer_param:
  max_epochs: 20

paths:
  default_root_dir: ./experiments/
  DAG_file: ./data/CelebA/final_DAG_unfair.csv
  expert_graphs_dir: ./data/CelebA/expert_graphs/graph_ensemble_edge_count/{graph_type}_{count}edges/
  input_model_path: ./pretrained_models/CelebA/version_11/checkpoints/epoch=89-step=6840.ckpt
"""

# Group CREAM configs by (graph_type, edge_count) → list of DAG files per seed
groups = defaultdict(list)

for yaml in sorted(glob.glob(os.path.join(src_dir, 'edge_count_*.yaml'))):
    m = re.match(r'.*edge_count_(u2c|c2y)_(\d+)edges_seed(\d+)\.yaml', yaml)
    if not m: continue
    gtype, count, seed = m.group(1), int(m.group(2)), int(m.group(3))

    with open(yaml, encoding='utf-8') as f:
        content = f.read()
    dag_m = re.search(r'DAG_file:\s*(\S+)', content)
    if not dag_m: continue
    dag_file = dag_m.group(1).strip()

    groups[(gtype, count)].append((seed, dag_file))

count_written = 0
for (gtype, n_edges), seed_dags in sorted(groups.items()):
    seed_dags_sorted = sorted(seed_dags, key=lambda x: x[0])
    M = len(seed_dags_sorted)
    dag_lines = '\n'.join(
        f'    - ./{d.lstrip("./").replace(chr(92), "/")}'
        for _, d in seed_dags_sorted
    )

    content = CONFIG_TEMPLATE.format(
        graph_type=gtype,
        count=n_edges,
        M=M,
        dag_lines=dag_lines,
    )

    fname = f'gensemble_edge_count_{gtype}_{n_edges}edges.yaml'
    with open(os.path.join(dst_dir, fname), 'w', encoding='utf-8') as f:
        f.write(content)
    count_written += 1

print(f'Written {count_written} configs to {dst_dir}')
print(f'  GT=10 u2c edges, range 5..15')
print(f'  Each config: 5 experts with different seed graphs of same edge count')
