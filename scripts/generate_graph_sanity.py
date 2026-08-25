import argparse
import csv
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_SERVER = "/home/dani00003/mCREAM"
CONDA_PYTHON_SERVER = "/home/dani00003/miniconda3/envs/mcream/bin/python"
DOCKER_IMAGE = "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime"


DATASETS = {
    "celeba": {
        "display": "CelebA",
        "server_dir": "celeba",
        "base_config": "all_configs/best_hparams/CREAM/CREAM_best_celeba.yaml",
        "base_dag": "data/CelebA/final_DAG_unfair.csv",
        "dag_prefix": "final_DAG_unfair",
        "experiment_prefix": "CREAM_celeba_graph_sanity",
        "model_name": "Standard_CelebA",
    },
    "cub": {
        "display": "CUB",
        "server_dir": "cub",
        "base_config": "all_configs/best_hparams/CREAM/CREAM_best_cub_soft_config.yaml",
        "base_dag": "data/CUB/CUB_DAG_only_Gc.csv",
        "dag_prefix": "CUB_DAG_only_Gc",
        "experiment_prefix": "CREAM_cub_graph_sanity",
        "model_name": "Standard_CUB",
    },
    "cfmnist": {
        "display": "Complete_Concept_FMNIST",
        "server_dir": "cfmnist",
        "base_config": "all_configs/best_hparams/CREAM/CREAM_best_cfmnist_soft_config.yaml",
        "base_dag": "data/FashionMNIST/Complete_Concept_FMNIST_DAG.csv",
        "dag_prefix": "Complete_Concept_FMNIST_DAG",
        "experiment_prefix": "CREAM_cfmnist_graph_sanity",
        "model_name": "Standard_FashionMNIST",
    },
}


VARIANTS = {
    "original": "Original CREAM graph copied from the base DAG",
    "all_ones": "All entries set to True",
    "identity": "Only diagonal/self-connections set to True",
    "random_same_edges": "Random True entries with the same edge count as original",
}


def rel(path: Path) -> str:
    return "./" + path.relative_to(PROJECT_ROOT).as_posix()


def read_bool_dag(path: Path) -> tuple[list[str], list[list[bool]]]:
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    names = rows[0][1:]
    matrix = [[cell == "True" for cell in row[1:]] for row in rows[1:]]
    if len(matrix) != len(names) or any(len(row) != len(names) for row in matrix):
        raise ValueError(f"DAG must be square with matching labels: {path}")
    return names, matrix


def write_bool_dag(path: Path, names: list[str], matrix: list[list[bool]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + names)
        for name, row in zip(names, matrix):
            writer.writerow([name] + ["True" if value else "False" for value in row])


def make_dag_variants(names: list[str], original: list[list[bool]], seed: int) -> dict[str, list[list[bool]]]:
    size = len(names)
    n_edges = sum(sum(row) for row in original)
    rng = random.Random(seed)
    random_positions = set(rng.sample(range(size * size), n_edges))
    return {
        "original": original,
        "all_ones": [[True for _ in range(size)] for _ in range(size)],
        "identity": [[row == col for col in range(size)] for row in range(size)],
        "random_same_edges": [
            [(row * size + col) in random_positions for col in range(size)]
            for row in range(size)
        ],
    }


def upsert_experiment_name(config_text: str, experiment_name: str) -> str:
    lines = config_text.splitlines()
    for idx, line in enumerate(lines):
        if line.startswith("experiment_name:"):
            lines[idx] = f"experiment_name: {experiment_name}"
            return "\n".join(lines) + "\n"

    for idx, line in enumerate(lines):
        if line.startswith("seed:"):
            lines.insert(idx + 1, f"experiment_name: {experiment_name}")
            return "\n".join(lines) + "\n"

    lines.insert(0, f"experiment_name: {experiment_name}")
    return "\n".join(lines) + "\n"


def replace_dag_path(config_text: str, dag_path: Path) -> str:
    lines = config_text.splitlines()
    for idx, line in enumerate(lines):
        if line.strip().startswith("DAG_file:"):
            indent = line[: len(line) - len(line.lstrip())]
            lines[idx] = f"{indent}DAG_file: {rel(dag_path)}"
            return "\n".join(lines) + "\n"
    raise ValueError("Could not find paths.DAG_file in base config.")


def write_configs(dataset_key: str, meta: dict, dag_paths: dict[str, Path]) -> dict[str, Path]:
    config_out_dir = PROJECT_ROOT / "all_configs" / "sanity_checks" / f"{dataset_key}_graph_sanity"
    config_out_dir.mkdir(parents=True, exist_ok=True)
    base_config_path = PROJECT_ROOT / meta["base_config"]
    base_text = base_config_path.read_text()

    config_paths = {}
    for variant, dag_path in dag_paths.items():
        experiment_name = f"{meta['experiment_prefix']}/{variant}"
        text = upsert_experiment_name(base_text, experiment_name)
        text = replace_dag_path(text, dag_path)
        text = (
            text.rstrip()
            + "\n"
            + "sanity_check:\n"
            + f"  dataset: {dataset_key}\n"
            + f"  graph_variant: {variant}\n"
            + f"  description: {VARIANTS[variant]}\n"
            + f"  base_dag: {rel(PROJECT_ROOT / meta['base_dag'])}\n"
        )
        path = config_out_dir / f"{meta['experiment_prefix']}_{variant}.yaml"
        path.write_text(text)
        config_paths[variant] = path
        print(f"Wrote {path}")

    return config_paths


def write_dags(dataset_key: str, meta: dict, seed: int) -> dict[str, Path]:
    base_dag = PROJECT_ROOT / meta["base_dag"]
    names, original = read_bool_dag(base_dag)
    variants = make_dag_variants(names, original, seed)
    dag_out_dir = base_dag.parent / "graph_sanity"

    paths = {}
    for variant, matrix in variants.items():
        path = dag_out_dir / f"{meta['dag_prefix']}_{variant}.csv"
        write_bool_dag(path, names, matrix)
        paths[variant] = path
        print(f"Wrote {path} ({sum(sum(row) for row in matrix)} true entries)")
    return paths


def write_run_scripts(dataset_key: str, meta: dict, config_paths: dict[str, Path]) -> None:
    script_out_dir = PROJECT_ROOT / "server_scripts" / "cream_experiment" / meta["server_dir"] / "graph_sanity"
    script_out_dir.mkdir(parents=True, exist_ok=True)

    for variant, config_path in config_paths.items():
        run_name = f"run_{meta['experiment_prefix']}_{variant}.sh"
        sub_name = f"{meta['experiment_prefix']}_{variant}_job.sub"
        run_path = script_out_dir / run_name
        sub_path = script_out_dir / sub_name
        config_rel = config_path.relative_to(PROJECT_ROOT).as_posix()

        run_path.write_text(
            f"""#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="{PROJECT_ROOT_SERVER}"
CONDA_PYTHON="{CONDA_PYTHON_SERVER}"
CONFIG_PATH="{config_rel}"

if [ -x "$CONDA_PYTHON" ]; then
    PYTHON_BIN="$CONDA_PYTHON"
else
    echo "ERROR: Conda env not found at $CONDA_PYTHON" >&2
    exit 127
fi

cd "$PROJECT_ROOT"
echo "HOST=$(hostname)"
"$PYTHON_BIN" -V
nvidia-smi || true
"$PYTHON_BIN" -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available())"
"$PYTHON_BIN" -c "import pytorch_lightning, torchvision, yaml; print('deps_ok=1')"

echo "Running {meta['display']} CREAM graph sanity variant: {variant}"
"$PYTHON_BIN" simple_main.py --config "$CONFIG_PATH"

echo "Done!"
"""
        )

        log_prefix = f"{meta['experiment_prefix']}_{variant}"
        sub_path.write_text(
            f"""universe                = docker
docker_image            = {DOCKER_IMAGE}
initialdir              = {PROJECT_ROOT_SERVER}
executable              = {PROJECT_ROOT_SERVER}/server_scripts/cream_experiment/{meta['server_dir']}/graph_sanity/{run_name}

output                  = {PROJECT_ROOT_SERVER}/logs/{log_prefix}.$(ClusterId).$(ProcId).out
error                   = {PROJECT_ROOT_SERVER}/logs/{log_prefix}.$(ClusterId).$(ProcId).err
log                     = {PROJECT_ROOT_SERVER}/logs/{log_prefix}.$(ClusterId).log

request_GPUs            = 1
request_CPUs            = 8
request_memory          = 32G
requirements            = UidDomain == "cs.uni-saarland.de"
+WantGPUHomeMounted     = true
queue 1
"""
        )
        print(f"Wrote {run_path}")
        print(f"Wrote {sub_path}")

    submit_all = script_out_dir / f"submit_all_{meta['experiment_prefix']}.sh"
    submit_all.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        + "\n".join(
            f"condor_submit server_scripts/cream_experiment/{meta['server_dir']}/graph_sanity/{meta['experiment_prefix']}_{variant}_job.sub"
            for variant in config_paths
        )
        + "\n"
    )
    print(f"Wrote {submit_all}")


def write_notebook(dataset_keys: list[str]) -> None:
    notebook_out = PROJECT_ROOT / "notebook" / "graph_sanity_analysis.ipynb"
    dataset_roots = {
        key: (
            f"../experiments/{DATASETS[key]['display']}/train_cbm/"
            f"{DATASETS[key]['model_name']}/{key}_graph_sanity/{DATASETS[key]['experiment_prefix']}"
        )
        for key in dataset_keys
    }
    dag_roots = {
        key: str((PROJECT_ROOT / DATASETS[key]["base_dag"]).parent.relative_to(PROJECT_ROOT).as_posix() + "/graph_sanity")
        for key in dataset_keys
    }
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# CREAM Graph Sanity Checks\n\n", "Compares original, all-ones, identity, and random-same-edge-count graphs.\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "import pandas as pd\n",
                    f"dataset_roots = {json.dumps(dataset_roots, indent=2)}\n",
                    "rows = []\n",
                    "for dataset, root_str in dataset_roots.items():\n",
                    "    root = Path(root_str)\n",
                    "    for csv_path in sorted(root.glob('*/last_metrics/*.csv')):\n",
                    "        df = pd.read_csv(csv_path)\n",
                    "        if df.empty:\n",
                    "            continue\n",
                    "        row = df.iloc[0].to_dict()\n",
                    "        row['dataset'] = dataset\n",
                    "        row['variant'] = csv_path.parents[1].name\n",
                    "        row['csv_path'] = str(csv_path)\n",
                    "        rows.append(row)\n",
                    "results = pd.DataFrame(rows)\n",
                    "cols = [c for c in ['dataset', 'variant', 'test_task_accuracy', 'test_concept_accuracy', 'test_dropout_task_accuracy', 'PFI_concept_importance', 'PFI_side_importance', 'num_trainable_parameters'] if c in results.columns]\n",
                    "results[cols].sort_values(['dataset', 'variant']) if not results.empty else results\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import matplotlib.pyplot as plt\n",
                    "if not results.empty and 'test_task_accuracy' in results.columns:\n",
                    "    for dataset, df in results.groupby('dataset'):\n",
                    "        ax = df.sort_values('variant').plot.bar(x='variant', y='test_task_accuracy', legend=False, figsize=(7, 4))\n",
                    "        ax.set_ylabel('Test task accuracy')\n",
                    "        ax.set_xlabel('Graph variant')\n",
                    "        ax.set_title(f'{dataset} CREAM Graph Sanity Check')\n",
                    "        plt.tight_layout()\n",
                    "        plt.show()\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import csv\n",
                    f"dag_roots = {json.dumps(dag_roots, indent=2)}\n",
                    "edge_counts = []\n",
                    "for dataset, dag_root in dag_roots.items():\n",
                    "    for path in sorted(Path('..', dag_root).glob('*.csv')):\n",
                    "        with open(path, newline='') as f:\n",
                    "            rows = list(csv.reader(f))\n",
                    "        edge_counts.append({\n",
                    "            'dataset': dataset,\n",
                    "            'dag': path.name,\n",
                    "            'true_entries': sum(cell == 'True' for row in rows[1:] for cell in row[1:]),\n",
                    "        })\n",
                    "pd.DataFrame(edge_counts)\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_out.write_text(json.dumps(notebook, indent=2))
    print(f"Wrote {notebook_out}")


def generate(dataset_keys: list[str], seed: int) -> None:
    for key in dataset_keys:
        meta = DATASETS[key]
        print(f"\n=== {key} ===")
        dag_paths = write_dags(key, meta, seed)
        config_paths = write_configs(key, meta, dag_paths)
        write_run_scripts(key, meta, config_paths)
    write_notebook(dataset_keys)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default="celeba,cub,cfmnist",
        help=f"Comma-separated list from: {','.join(DATASETS)}",
    )
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    dataset_keys = [key.strip().lower() for key in args.datasets.split(",") if key.strip()]
    unknown = sorted(set(dataset_keys) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown dataset keys: {unknown}. Available: {sorted(DATASETS)}")
    generate(dataset_keys, args.random_seed)


if __name__ == "__main__":
    main()
