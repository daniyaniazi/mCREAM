import argparse
import csv
import json
import random
from pathlib import Path

from generate_graph_sanity import (
    CONDA_PYTHON_SERVER,
    DATASETS,
    DOCKER_IMAGE,
    PROJECT_ROOT,
    PROJECT_ROOT_SERVER,
    read_bool_dag,
    rel,
    replace_dag_path,
    upsert_experiment_name,
    write_bool_dag,
)


NOISE_LEVELS = list(range(10, 101, 10))


def flip_progressive_noise(
    original: list[list[bool]], noise_percent: int, seed: int
) -> tuple[list[list[bool]], dict]:
    size = len(original)
    total_positions = size * size
    n_flip = round(total_positions * noise_percent / 100)

    rng = random.Random(seed + noise_percent)
    flip_positions = set(rng.sample(range(total_positions), n_flip))

    perturbed = [row[:] for row in original]
    added = 0
    deleted = 0
    for pos in flip_positions:
        row = pos // size
        col = pos % size
        if perturbed[row][col]:
            deleted += 1
        else:
            added += 1
        perturbed[row][col] = not perturbed[row][col]

    original_edges = sum(sum(row) for row in original)
    perturbed_edges = sum(sum(row) for row in perturbed)
    graph_distance = n_flip / total_positions

    return perturbed, {
        "noise_percent": noise_percent,
        "different_edges": n_flip,
        "total_edge_positions": total_positions,
        "graph_distance": graph_distance,
        "added_edges": added,
        "deleted_edges": deleted,
        "original_edges": original_edges,
        "perturbed_edges": perturbed_edges,
    }


def write_dags(dataset_key: str, meta: dict, seed: int) -> tuple[dict[int, Path], list[dict]]:
    base_dag = PROJECT_ROOT / meta["base_dag"]
    names, original = read_bool_dag(base_dag)
    out_dir = base_dag.parent / "progressive_noise_graph"
    out_dir.mkdir(parents=True, exist_ok=True)

    dag_paths = {}
    metadata_rows = []
    for noise_percent in NOISE_LEVELS:
        perturbed, metadata = flip_progressive_noise(original, noise_percent, seed)
        path = out_dir / f"{meta['dag_prefix']}_noise_{noise_percent:03d}.csv"
        write_bool_dag(path, names, perturbed)
        dag_paths[noise_percent] = path
        metadata_rows.append(
            {
                "dataset": dataset_key,
                "dag_path": rel(path),
                **metadata,
            }
        )
        print(
            f"Wrote {path} "
            f"(noise={noise_percent}%, distance={metadata['graph_distance']:.3f}, "
            f"added={metadata['added_edges']}, deleted={metadata['deleted_edges']})"
        )

    metadata_path = out_dir / "progressive_noise_metadata.csv"
    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metadata_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metadata_rows)
    print(f"Wrote {metadata_path}")
    return dag_paths, metadata_rows


def write_configs(dataset_key: str, meta: dict, dag_paths: dict[int, Path]) -> dict[int, Path]:
    out_dir = PROJECT_ROOT / "all_configs" / "sanity_checks" / f"{dataset_key}_progressive_noise"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_text = (PROJECT_ROOT / meta["base_config"]).read_text()

    config_paths = {}
    for noise_percent, dag_path in dag_paths.items():
        experiment_name = f"CREAM_{dataset_key}_progressive_noise/noise_{noise_percent:03d}"
        text = upsert_experiment_name(base_text, experiment_name)
        text = replace_dag_path(text, dag_path)
        text = (
            text.rstrip()
            + "\n"
            + "sanity_check:\n"
            + f"  dataset: {dataset_key}\n"
            + "  experiment_type: progressive_noise\n"
            + f"  noise_percent: {noise_percent}\n"
            + "  perturbation: uniform_flip_any_edge_including_diagonal\n"
            + f"  base_dag: {rel(PROJECT_ROOT / meta['base_dag'])}\n"
        )
        path = out_dir / f"CREAM_{dataset_key}_progressive_noise_{noise_percent:03d}.yaml"
        path.write_text(text)
        config_paths[noise_percent] = path
        print(f"Wrote {path}")

    return config_paths


def write_run_scripts(dataset_key: str, meta: dict, config_paths: dict[int, Path]) -> None:
    out_dir = (
        PROJECT_ROOT
        / "server_scripts"
        / "cream_experiment"
        / meta["server_dir"]
        / "progressive_noise"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    for noise_percent, config_path in config_paths.items():
        run_name = f"run_CREAM_{dataset_key}_progressive_noise_{noise_percent:03d}.sh"
        sub_name = f"CREAM_{dataset_key}_progressive_noise_{noise_percent:03d}_job.sub"
        run_path = out_dir / run_name
        sub_path = out_dir / sub_name
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

echo "Running {meta['display']} CREAM progressive graph noise: {noise_percent}%"
"$PYTHON_BIN" simple_main.py --config "$CONFIG_PATH"

echo "Done!"
"""
        )

        log_prefix = f"CREAM_{dataset_key}_progressive_noise_{noise_percent:03d}"
        sub_path.write_text(
            f"""universe                = docker
docker_image            = {DOCKER_IMAGE}
initialdir              = {PROJECT_ROOT_SERVER}
executable              = {PROJECT_ROOT_SERVER}/server_scripts/cream_experiment/{meta['server_dir']}/progressive_noise/{run_name}

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

    submit_all = out_dir / f"submit_all_CREAM_{dataset_key}_progressive_noise.sh"
    submit_all.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        + "\n".join(
            f"condor_submit server_scripts/cream_experiment/{meta['server_dir']}/progressive_noise/CREAM_{dataset_key}_progressive_noise_{noise_percent:03d}_job.sub"
            for noise_percent in config_paths
        )
        + "\n"
    )
    print(f"Wrote {submit_all}")


def write_notebook(dataset_keys: list[str]) -> None:
    notebook_out = PROJECT_ROOT / "notebook" / "progressive_noise_analysis.ipynb"
    dataset_roots = {
        key: (
            f"../experiments/{DATASETS[key]['display']}/train_cbm/"
            f"{DATASETS[key]['model_name']}/{key}_progressive_noise/CREAM_{key}_progressive_noise"
        )
        for key in dataset_keys
    }
    metadata_paths = {
        key: "../" + str(
            (
                Path(DATASETS[key]["base_dag"]).parent
                / "progressive_noise_graph"
                / "progressive_noise_metadata.csv"
            ).as_posix()
        )
        for key in dataset_keys
    }
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# CREAM Progressive Graph Noise\n\n",
                    "Accuracy as a function of graph distance. Noise flips uniformly selected adjacency entries, including diagonal entries.\n",
                ],
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
                    f"metadata_paths = {json.dumps(metadata_paths, indent=2)}\n",
                    "rows = []\n",
                    "for dataset, root_str in dataset_roots.items():\n",
                    "    root = Path(root_str)\n",
                    "    meta = pd.read_csv(metadata_paths[dataset])\n",
                    "    for csv_path in sorted(root.glob('noise_*/last_metrics/*.csv')):\n",
                    "        df = pd.read_csv(csv_path)\n",
                    "        if df.empty:\n",
                    "            continue\n",
                    "        row = df.iloc[0].to_dict()\n",
                    "        row['dataset'] = dataset\n",
                    "        row['noise_percent'] = int(csv_path.parents[1].name.replace('noise_', ''))\n",
                    "        row['csv_path'] = str(csv_path)\n",
                    "        rows.append(row)\n",
                    "results = pd.DataFrame(rows)\n",
                    "metadata = pd.concat([pd.read_csv(path) for path in metadata_paths.values()], ignore_index=True)\n",
                    "merged = results.merge(metadata, on=['dataset', 'noise_percent'], how='left') if not results.empty else results\n",
                    "cols = [c for c in ['dataset', 'noise_percent', 'graph_distance', 'different_edges', 'added_edges', 'deleted_edges', 'perturbed_edges', 'test_task_accuracy', 'test_concept_accuracy', 'test_dropout_task_accuracy'] if c in merged.columns]\n",
                    "merged[cols].sort_values(['dataset', 'noise_percent']) if not merged.empty else merged\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import matplotlib.pyplot as plt\n",
                    "if not merged.empty and 'test_task_accuracy' in merged.columns:\n",
                    "    for dataset, df in merged.groupby('dataset'):\n",
                    "        df = df.sort_values('graph_distance')\n",
                    "        ax = df.plot(x='graph_distance', y='test_task_accuracy', marker='o', figsize=(7, 4), legend=False)\n",
                    "        ax.set_xlabel('Graph distance = different edge positions / total positions')\n",
                    "        ax.set_ylabel('Test task accuracy')\n",
                    "        ax.set_title(f'{dataset}: accuracy vs progressive graph noise')\n",
                    "        ax.grid(True, alpha=0.3)\n",
                    "        plt.tight_layout()\n",
                    "        plt.show()\n",
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
        dag_paths, _ = write_dags(key, meta, seed)
        config_paths = write_configs(key, meta, dag_paths)
        write_run_scripts(key, meta, config_paths)
    write_notebook(dataset_keys)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default="cub",
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
