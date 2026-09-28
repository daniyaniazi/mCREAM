import argparse
import csv
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


def edge_count(matrix: list[list[bool]]) -> int:
    return sum(sum(row) for row in matrix)


def make_random_full_matrix(size: int, n_edges: int, seed: int) -> list[list[bool]]:
    total_positions = size * size
    if n_edges < 0 or n_edges > total_positions:
        raise ValueError(f"n_edges must be in [0, {total_positions}], got {n_edges}")

    rng = random.Random(seed)
    positions = set(rng.sample(range(total_positions), n_edges))
    return [
        [(row * size + col) in positions for col in range(size)]
        for row in range(size)
    ]


def default_density_counts(original_edges: int, total_positions: int) -> list[int]:
    multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]
    counts = []
    for multiplier in multipliers:
        count = max(1, min(total_positions, round(original_edges * multiplier)))
        if count not in counts:
            counts.append(count)
    return counts


def parse_density_counts(raw: str | None, original_edges: int, total_positions: int) -> list[int]:
    if not raw:
        return default_density_counts(original_edges, total_positions)
    counts = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        count = int(part)
        if count < 0 or count > total_positions:
            raise ValueError(f"Density edge count {count} outside [0, {total_positions}]")
        if count not in counts:
            counts.append(count)
    return counts


def write_metadata(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def write_configs(
    dataset_key: str,
    meta: dict,
    experiment_family: str,
    graph_paths: dict[str, Path],
    metadata_by_variant: dict[str, dict],
) -> dict[str, Path]:
    config_out_dir = PROJECT_ROOT / "all_configs" / "sanity_checks" / f"{dataset_key}_{experiment_family}"
    config_out_dir.mkdir(parents=True, exist_ok=True)
    base_text = (PROJECT_ROOT / meta["base_config"]).read_text()

    config_paths = {}
    experiment_prefix = f"CREAM_{dataset_key}_{experiment_family}"
    for variant, dag_path in graph_paths.items():
        experiment_name = f"{experiment_prefix}/{variant}"
        text = upsert_experiment_name(base_text, experiment_name)
        text = replace_dag_path(text, dag_path)
        metadata = metadata_by_variant[variant]
        text = (
            text.rstrip()
            + "\n"
            + "sanity_check:\n"
            + f"  dataset: {dataset_key}\n"
            + f"  experiment_type: {experiment_family}\n"
            + f"  graph_variant: {variant}\n"
            + "  random_scope: full_dag_matrix_including_diagonal_and_task_rows\n"
            + f"  base_dag: {rel(PROJECT_ROOT / meta['base_dag'])}\n"
            + f"  original_edge_count: {metadata['original_edge_count']}\n"
            + f"  random_edge_count: {metadata['edge_count']}\n"
            + f"  total_positions: {metadata['total_positions']}\n"
            + f"  density: {metadata['density']}\n"
            + f"  random_seed: {metadata['random_seed']}\n"
        )
        path = config_out_dir / f"{experiment_prefix}_{variant}.yaml"
        path.write_text(text)
        config_paths[variant] = path
        print(f"Wrote {path}")

    return config_paths


def write_run_scripts(dataset_key: str, meta: dict, experiment_family: str, config_paths: dict[str, Path]) -> None:
    out_dir = (
        PROJECT_ROOT
        / "server_scripts"
        / "cream_experiment"
        / meta["server_dir"]
        / experiment_family
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    experiment_prefix = f"CREAM_{dataset_key}_{experiment_family}"

    for variant, config_path in config_paths.items():
        run_name = f"run_{experiment_prefix}_{variant}.sh"
        sub_name = f"{experiment_prefix}_{variant}_job.sub"
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

echo "Running {meta['display']} CREAM {experiment_family}: {variant}"
"$PYTHON_BIN" simple_main.py --config "$CONFIG_PATH"

echo "Done!"
"""
        )

        log_prefix = f"{experiment_prefix}_{variant}"
        sub_path.write_text(
            f"""universe                = docker
docker_image            = {DOCKER_IMAGE}
initialdir              = {PROJECT_ROOT_SERVER}
executable              = {PROJECT_ROOT_SERVER}/server_scripts/cream_experiment/{meta['server_dir']}/{experiment_family}/{run_name}

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

    submit_all = out_dir / f"submit_all_{experiment_prefix}.sh"
    submit_all.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        + "\n".join(
            f"condor_submit server_scripts/cream_experiment/{meta['server_dir']}/{experiment_family}/{experiment_prefix}_{variant}_job.sub"
            for variant in config_paths
        )
        + "\n"
    )
    print(f"Wrote {submit_all}")


def generate_for_dataset(
    dataset_key: str,
    topology_graphs: int,
    density_counts_raw: str | None,
    seed: int,
) -> None:
    meta = DATASETS[dataset_key]
    base_dag = PROJECT_ROOT / meta["base_dag"]
    names, original = read_bool_dag(base_dag)
    size = len(names)
    total_positions = size * size
    original_edges = edge_count(original)

    print(f"\n=== {dataset_key} ===")
    print(f"Full DAG size={size}x{size}, original edges={original_edges}, total positions={total_positions}")

    # Experiment 1: same edge count, multiple completely random topologies.
    topology_dir = base_dag.parent / "topology_random_full"
    topology_paths = {}
    topology_meta = {}
    topology_rows = []
    for idx in range(topology_graphs):
        variant = f"rand_topology_{idx:02d}"
        random_seed = seed + idx
        matrix = make_random_full_matrix(size, original_edges, random_seed)
        path = topology_dir / f"{meta['dag_prefix']}_{variant}_edges_{original_edges}.csv"
        write_bool_dag(path, names, matrix)
        topology_paths[variant] = path
        row = {
            "dataset": dataset_key,
            "experiment_type": "topology_random_full",
            "variant": variant,
            "dag_path": rel(path),
            "original_edge_count": original_edges,
            "edge_count": edge_count(matrix),
            "total_positions": total_positions,
            "density": edge_count(matrix) / total_positions,
            "random_seed": random_seed,
        }
        topology_meta[variant] = row
        topology_rows.append(row)
        print(f"Wrote {path} ({row['edge_count']} edges)")
    write_metadata(topology_dir / "topology_random_full_metadata.csv", topology_rows)
    topology_configs = write_configs(
        dataset_key, meta, "topology_random_full", topology_paths, topology_meta
    )
    write_run_scripts(dataset_key, meta, "topology_random_full", topology_configs)

    # Experiment 2: different edge counts, each with a random topology.
    density_dir = base_dag.parent / "density_random_full"
    density_counts = parse_density_counts(density_counts_raw, original_edges, total_positions)
    density_paths = {}
    density_meta = {}
    density_rows = []
    for idx, count in enumerate(density_counts):
        variant = f"edges_{count:05d}"
        random_seed = seed + 10_000 + idx
        matrix = make_random_full_matrix(size, count, random_seed)
        path = density_dir / f"{meta['dag_prefix']}_{variant}.csv"
        write_bool_dag(path, names, matrix)
        density_paths[variant] = path
        row = {
            "dataset": dataset_key,
            "experiment_type": "density_random_full",
            "variant": variant,
            "dag_path": rel(path),
            "original_edge_count": original_edges,
            "edge_count": edge_count(matrix),
            "total_positions": total_positions,
            "density": edge_count(matrix) / total_positions,
            "random_seed": random_seed,
        }
        density_meta[variant] = row
        density_rows.append(row)
        print(f"Wrote {path} ({row['edge_count']} edges)")
    write_metadata(density_dir / "density_random_full_metadata.csv", density_rows)
    density_configs = write_configs(
        dataset_key, meta, "density_random_full", density_paths, density_meta
    )
    write_run_scripts(dataset_key, meta, "density_random_full", density_configs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default="celeba,cub,cfmnist",
        help=f"Comma-separated list from: {','.join(DATASETS)}",
    )
    parser.add_argument("--topology_graphs", type=int, default=5)
    parser.add_argument(
        "--density_edge_counts",
        default=None,
        help="Comma-separated edge counts for density sweep. If omitted, uses 0.5x,0.75x,1x,1.25x,1.5x original edge count.",
    )
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    dataset_keys = [key.strip().lower() for key in args.datasets.split(",") if key.strip()]
    unknown = sorted(set(dataset_keys) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown dataset keys: {unknown}. Available: {sorted(DATASETS)}")
    if args.topology_graphs <= 0:
        raise ValueError("--topology_graphs must be positive")

    for dataset_key in dataset_keys:
        generate_for_dataset(
            dataset_key=dataset_key,
            topology_graphs=args.topology_graphs,
            density_counts_raw=args.density_edge_counts,
            seed=args.random_seed,
        )


if __name__ == "__main__":
    main()
