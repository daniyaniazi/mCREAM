#!/usr/bin/env python3
"""Generate all M=10 YAML configs + HTCondor .sub/.sh server scripts
for cfmnist and celeba: baselines (intersection, majority, union), edge, graph.
All 4 disagreement levels: low, medium, high, structured_bias.
"""
import os

DISAGREEMENT_LEVELS = ["low", "medium", "high", "structured_bias"]
BASELINES = ["intersection", "majority", "union"]
LEARNABLE = ["edge", "graph"]

PROJECT_ROOT_SERVER = "/home/dani00003/mCREAM"
CONDA_PYTHON = "/home/dani00003/miniconda3/envs/mcream/bin/python"

# ─── Dataset-specific settings ───────────────────────────────────────────────
DATASETS = {
    "cfmnist": {
        "dataset_name": "Complete_Concept_FMNIST",
        "backbone_model": "Standard_FashionMNIST",
        "dataset_params": """  batch_size: 256
  workers: 2
  return_labels: true
  return_images: true""",
        "num_classes": 10,
        "num_concepts": 11,
        "num_exogenous": 128,
        "num_side_channel": 40,
        "concept_representation": "group_soft",
        "previous_model_output_size": 128,
        "side_dropout": True,
        "dropout_prob": 0.9,
        "max_epochs": 50,
        "gradient_clip_extra": "\n  gradient_clip_val: null\n  gradient_clip_algorithm: null",
        "dag_file": "./data/FashionMNIST/Complete_Concept_FMNIST_DAG.csv",
        "expert_graphs_base": "./data/FashionMNIST/expert_graphs",
        "input_model_path": "./pretrained_models/FMNIST/version_0/checkpoints/epoch=49-step=10750.ckpt",
        "softmax_mask": "  softmax_mask: ./data/FashionMNIST/mutually_exclusive_relationships_COMPLETE.json",
        # Server paths
        "dag_path_server": "data/FashionMNIST/Complete_Concept_FMNIST_DAG.csv",
        "expert_dir_base_server": "data/FashionMNIST/expert_graphs",
        "config_dir": "all_configs/mcream_configs/cfmnist",
        "server_dir": "server_scripts/mcream_experiment/cfmnist",
        "log_prefix": "mcream_cfmnist",
    },
    "celeba": {
        "dataset_name": "CelebA",
        "backbone_model": "Standard_CelebA",
        "dataset_params": """  batch_size: 256
  workers: 2
  class_name: unfair""",
        "num_classes": 1,
        "num_concepts": 7,
        "num_exogenous": 75,
        "num_side_channel": 5,
        "concept_representation": "soft",
        "previous_model_output_size": 512,
        "side_dropout": True,
        "dropout_prob": 0.1,
        "max_epochs": 20,
        "gradient_clip_extra": "\n  gradient_clip_val: null",
        "dag_file": "./data/CelebA/final_DAG_unfair.csv",
        "expert_graphs_base": "./data/CelebA/expert_graphs",
        "input_model_path": "./pretrained_models/CelebA/version_11/checkpoints/epoch=89-step=6840.ckpt",
        "softmax_mask": "",
        # Server paths
        "dag_path_server": "data/CelebA/final_DAG_unfair.csv",
        "expert_dir_base_server": "data/CelebA/expert_graphs",
        "config_dir": "all_configs/mcream_configs/celeba",
        "server_dir": "server_scripts/mcream_experiment/celeba",
        "log_prefix": "mcream_celeba",
    },
}

BASELINE_DESCRIPTIONS = {
    "intersection": "Intersection of all expert graphs (most conservative, AND operation)",
    "majority": "Majority vote — keep edges that >50% of experts include",
    "union": "Union of all expert graphs (most permissive, OR operation)",
}

LEARNABLE_DESCRIPTIONS = {
    "edge": "Learn per-edge reliability α, output = sigmoid(α)",
    "graph": "Learn per-expert weight π, output = Σ π_m * A^(m)",
}

M = 10
LOCAL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

created = []


def write_file(rel_path, content):
    full = os.path.join(LOCAL_ROOT, rel_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    if os.path.exists(full):
        print(f"  SKIP (exists): {rel_path}")
        return
    with open(full, "w", newline="\n", encoding="utf-8") as f:
        f.write(content)
    created.append(rel_path)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_config_yaml(ds_key, exp_type, agg_type, level):
    """Generate YAML config content."""
    d = DATASETS[ds_key]
    exp_name = f"{agg_type}_M{M}_{level}"
    is_baseline = agg_type in BASELINES

    if is_baseline:
        desc = BASELINE_DESCRIPTIONS[agg_type]
        reg = """    prior_weight: 0.0
    sparsity_weight: 0.0
    acyclicity_weight: 0.0"""
    else:
        desc = LEARNABLE_DESCRIPTIONS[agg_type]
        reg = """    prior_weight: 0.1
    sparsity_weight: 0.01
    acyclicity_weight: 0.0"""

    ds_label = "CelebA" if ds_key == "celeba" else ""
    header = f"# mCREAM - {agg_type.capitalize()} {'Baseline' if is_baseline else 'Aggregation'} (M={M}, {level.replace('_', ' ').title()}){' - ' + ds_label if ds_label else ''}"
    header += f"\n# {desc}"

    softmax_line = f"\n{d['softmax_mask']}" if d["softmax_mask"] else ""

    yaml = f"""{header}

mode: train_cbm
seed: 42
seeds: [42, 7, 1, 134, 89]
dataset_name: {d['dataset_name']}

dataset_params:
{d['dataset_params']}

backbone_model: {d['backbone_model']}

multi_expert:
  enabled: true
  num_experts: {M}
  disagreement_level: {level}
  aggregation_type: {agg_type}
  graph_regularization:
{reg}

hyperparameters_model2:
  num_classes: {d['num_classes']}
  num_concepts: {d['num_concepts']}
  num_exogenous: {d['num_exogenous']}
  num_side_channel: {d['num_side_channel']}
  concept_representation: {d['concept_representation']}
  num_hidden_layers_in_maskedmlp: 0
  previous_model_output_size: {d['previous_model_output_size']}
  side_dropout: {'true' if d['side_dropout'] else 'false'}
  dropout_prob: {d['dropout_prob']}

hyperparameters:
  learning_rate: 0.001
  lambda_weight: 1
  frozen_model1: true

trainer_param:
  max_epochs: {d['max_epochs']}{d['gradient_clip_extra']}

experiment_name: {exp_name}

paths:
  default_root_dir: ./experiments/
  metric_dir: ./last_metrics/
  DAG_file: {d['dag_file']}
  expert_graphs_dir: {d['expert_graphs_base']}/M{M}/{level}/
  input_model_path: {d['input_model_path']}{softmax_line}
"""
    return yaml


def make_run_sh(ds_key, exp_type, agg_type, level, config_rel):
    """Generate .sh run script."""
    d = DATASETS[ds_key]
    exp_name = f"{agg_type}_M{M}_{level}"
    ds_label = ds_key.upper() if ds_key == "celeba" else "CFMNIST"
    is_baseline = agg_type in BASELINES

    expert_gen = ""
    if not is_baseline:
        expert_dir_server = f"$PROJECT_ROOT/{d['expert_dir_base_server']}/M{M}/{level}"
        expert_gen = f"""
"$PYTHON_BIN" -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available())"

# Check if expert graphs exist
EXPERT_DIR="{expert_dir_server}"
if [ ! -d "$EXPERT_DIR" ]; then
    echo "Generating expert graphs (M={M}, {level})..."
    "$PYTHON_BIN" -c "
from src.expert_graphs.generation import generate_expert_graphs_from_dag, save_expert_graphs, DISAGREEMENT_LEVELS
params = DISAGREEMENT_LEVELS['{level}']
expert_u2c, expert_c2y, u2c_star, c2y_star = generate_expert_graphs_from_dag(
    dag_path='{d['dag_path_server']}',
    num_classes={d['num_classes']}, num_experts={M},
    p_del=params['p_del'], p_add=params['p_add'], p_rev=params['p_rev'],
    base_seed=42
)
save_expert_graphs(expert_u2c, expert_c2y, '$EXPERT_DIR', {{'disagreement_level': '{level}', 'num_experts': {M}}})
print('Expert graphs saved!')
"
fi
"""

    agg_label = agg_type.capitalize()
    sh = f"""#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT="{PROJECT_ROOT_SERVER}"
CONDA_PYTHON="{CONDA_PYTHON}"

if [ -x "$CONDA_PYTHON" ]; then
    PYTHON_BIN="$CONDA_PYTHON"
else
    echo "ERROR: Conda env not found at $CONDA_PYTHON" >&2
    exit 127
fi

cd "$PROJECT_ROOT"

echo "=============================================="
echo "mCREAM: {agg_label} (M={M}, {level.replace('_', ' ').title()}) - {ds_label}"
echo "=============================================="
echo "HOST=$(hostname)"
"$PYTHON_BIN" -V
nvidia-smi || true
{expert_gen}
echo ""
echo "Running mCREAM with {agg_label} aggregation (M={M}, {level})..."
"$PYTHON_BIN" mcream_main.py --config {config_rel}

echo "Done!"
"""
    return sh


def make_job_sub(ds_key, agg_type, level, sh_server_path):
    """Generate HTCondor .sub file."""
    d = DATASETS[ds_key]
    exp_name = f"{agg_type}_M{M}_{level}"
    log_base = f"{d['log_prefix']}_{exp_name}"

    sub = f"""universe                = docker
docker_image            = pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
executable              = {sh_server_path}

output                  = {PROJECT_ROOT_SERVER}/logs/{log_base}.$(ClusterId).$(ProcId).out
error                   = {PROJECT_ROOT_SERVER}/logs/{log_base}.$(ClusterId).$(ProcId).err
log                     = {PROJECT_ROOT_SERVER}/logs/{log_base}.$(ClusterId).log

request_GPUs            = 1
request_CPUs            = 8
request_memory          = 32G
requirements            = UidDomain == "cs.uni-saarland.de"
+WantGPUHomeMounted     = true
queue 1
"""
    return sub


# ─── Main generation ─────────────────────────────────────────────────────────

for ds_key in ["cfmnist", "celeba"]:
    d = DATASETS[ds_key]

    for level in DISAGREEMENT_LEVELS:
        # --- Baselines ---
        for bl in BASELINES:
            exp_name = f"{bl}_M{M}_{level}"
            config_rel = f"{d['config_dir']}/baselines/{exp_name}.yaml"
            server_script_dir = f"{d['server_dir']}/baselines/{bl}"
            sh_name = f"run_{exp_name}.sh"
            sub_name = f"{exp_name}_job.sub"

            write_file(config_rel, make_config_yaml(ds_key, "baselines", bl, level))

            sh_server_path = f"{PROJECT_ROOT_SERVER}/{server_script_dir}/{sh_name}"
            write_file(f"{server_script_dir}/{sh_name}",
                        make_run_sh(ds_key, "baselines", bl, level, config_rel))
            write_file(f"{server_script_dir}/{sub_name}",
                        make_job_sub(ds_key, bl, level, sh_server_path))

        # --- Edge & Graph ---
        for agg in LEARNABLE:
            exp_name = f"{agg}_M{M}_{level}"
            config_rel = f"{d['config_dir']}/{agg}/{exp_name}.yaml"
            server_script_dir = f"{d['server_dir']}/{agg}"
            sh_name = f"run_{exp_name}.sh"
            sub_name = f"{exp_name}_job.sub"

            write_file(config_rel, make_config_yaml(ds_key, agg, agg, level))

            sh_server_path = f"{PROJECT_ROOT_SERVER}/{server_script_dir}/{sh_name}"
            write_file(f"{server_script_dir}/{sh_name}",
                        make_run_sh(ds_key, agg, agg, level, config_rel))
            write_file(f"{server_script_dir}/{sub_name}",
                        make_job_sub(ds_key, agg, level, sh_server_path))

print(f"\n{'='*60}")
print(f"Created {len(created)} files:")
for f in sorted(created):
    print(f"  {f}")
print(f"{'='*60}")
