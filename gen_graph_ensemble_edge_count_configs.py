"""
Generate mCREAM Graph Ensemble configs for edge count experiments.

Each expert gets a DIFFERENT seed's graph — all with the same edge count.
The 5 seeds from CREAM edge count become the 5 experts.

One config per (graph_type, edge_count):
  u2c_models[0] ← seed0 graph (12 u2c edges, random selection A)
  u2c_models[1] ← seed1 graph (12 u2c edges, random selection B)
  u2c_models[2] ← seed2 graph (12 u2c edges, random selection C)
  u2c_models[3] ← seed3 graph (12 u2c edges, random selection D)
  u2c_models[4] ← seed4 graph (12 u2c edges, random selection E)

Compare: CREAM boxplot (5 separate models) vs GraphEnsemble (5 experts in one model)
"""

import os, glob, re
from pathlib import Path
from collections import defaultdict

src_dir = 'all_configs/mcream_configs/cfmnist/edge_count_experiments'
dst_dir = 'all_configs/mcream_graph_ensemble_configs/cfmnist/edge_count'
os.makedirs(dst_dir, exist_ok=True)

CONFIG_TEMPLATE = """\
# mCREAM Graph Ensemble - edge count: {graph_type} {count} edges
# Expert m uses seed-m graph — same count, different random edge selection.
# 5 diverse graphs → 1 ensemble (compare with CREAM boxplot at same count).
mode: train_cbm
seed: 42
experiment_name: gensemble_edge_count_{graph_type}_{count}edges
dataset_name: Complete_Concept_FMNIST

dataset_params:
  batch_size: 256
  workers: 2
  return_labels: true
  return_images: true

backbone_model: Standard_FashionMNIST

multi_expert:
  num_experts: {M}
  noise_type: edge_count_multi_seed
  expert_dag_files:
{dag_lines}

hyperparameters_model2:
  num_classes: 10
  num_concepts: 11
  num_exogenous: 128
  num_side_channel: 40
  concept_representation: group_soft
  num_hidden_layers_in_maskedmlp: 0
  previous_model_output_size: 128
  side_dropout: true
  dropout_prob: 0.9

hyperparameters:
  learning_rate: 0.001
  lambda_weight: 1
  frozen_model1: true

trainer_param:
  max_epochs: 50

paths:
  default_root_dir: ./experiments/
  DAG_file: ./data/FashionMNIST/Complete_Concept_FMNIST_DAG.csv
  expert_graphs_dir: ./data/FashionMNIST/expert_graphs/graph_ensemble_edge_count/{graph_type}_{count}edges/
  input_model_path: ./pretrained_models/FMNIST/version_0/checkpoints/epoch=49-step=10750.ckpt
  softmax_mask: ./data/FashionMNIST/mutually_exclusive_relationships_COMPLETE.json
"""

# Group CREAM configs by (graph_type, edge_count) → list of DAG files per seed
groups = defaultdict(list)  # key=(gtype, count) → [(seed, dag_file)]

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

# Write one config per (graph_type, edge_count)
count_written = 0
for (gtype, n_edges), seed_dags in sorted(groups.items()):
    seed_dags_sorted = sorted(seed_dags, key=lambda x: x[0])  # sort by seed
    M = len(seed_dags_sorted)
    dag_lines = '\n'.join(f'    - ./{d.lstrip("./")}' for _, d in seed_dags_sorted)

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
print(f'  Each config: 5 experts, each with different seed graph of same edge count')
print(f'  e.g. gensemble_edge_count_u2c_12edges.yaml → 5 experts with different 12-edge u2c graphs')
