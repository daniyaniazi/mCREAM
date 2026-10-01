import argparse
import csv
import random
from copy import deepcopy
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


EXPERIMENT_FAMILY = "concept_concept_fixcount_topology"
DEFAULT_REPLACEMENT_PCTS = [0, 20, 50, 80, 100]
NUM_CLASSES = {
    "celeba": 1,
    "cub": 200,
    "cfmnist": 10,
}


def edge_count(matrix: list[list[bool]]) -> int:
    return sum(sum(row) for row in matrix)


def c2c_positions(matrix: list[list[bool]], num_concepts: int, value: bool) -> list[tuple[int, int]]:
    return [
        (row, col)
        for row in range(num_concepts)
        for col in range(num_concepts)
        if matrix[row][col] is value
    ]


def replace_c2c_edges(
    original: list[list[bool]],
    num_concepts: int,
    replacement_pct: int,
    seed: int,
) -> tuple[list[list[bool]], dict]:
    true_positions = c2c_positions(original, num_concepts, True)
    false_positions = c2c_positions(original, num_concepts, False)
    original_c2c_edges = len(true_positions)
    target_replace_count = round(original_c2c_edges * replacement_pct / 100)
    replace_count = min(target_replace_count, len(false_positions))
    capped = replace_count != target_replace_count

    rng = random.Random(seed)
    removed = set(rng.sample(true_positions, replace_count))
    added = set(rng.sample(false_positions, replace_count))

    matrix = deepcopy(original)
    for row, col in removed:
        matrix[row][col] = False
    for row, col in added:
        matrix[row][col] = True

    metadata = {
        "target_replacement_pct": replacement_pct,
        "actual_replacement_pct": replace_count / original_c2c_edges * 100 if original_c2c_edges else 0.0,
        "target_removed_c2c_edges": target_replace_count,
        "capped_by_available_non_edges": capped,
        "original_c2c_edges": original_c2c_edges,
        "available_original_c2c_non_edges": len(false_positions),
        "removed_c2c_edges": len(removed),
        "added_c2c_edges": len(added),
        "new_c2c_edges": len(c2c_positions(matrix, num_concepts, True)),
        "random_seed": seed,
    }
    return matrix, metadata


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
    graph_paths: dict[str, Path],
    metadata_by_variant: dict[str, dict],
) -> dict[str, Path]:
    config_out_dir = PROJECT_ROOT / "all_configs" / "sanity_checks" / f"{dataset_key}_{EXPERIMENT_FAMILY}"
    config_out_dir.mkdir(parents=True, exist_ok=True)
    base_text = (PROJECT_ROOT / meta["base_config"]).read_text()

    config_paths = {}
    experiment_prefix = f"CREAM_{dataset_key}_{EXPERIMENT_FAMILY}"
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
            + f"  experiment_type: {EXPERIMENT_FAMILY}\n"
            + f"  graph_variant: {variant}\n"
            + "  random_scope: concept_concept_block_only\n"
            + "  fixed_blocks: c2y_task_rows_and_all_non_c2c_entries\n"
            + f"  base_dag: {rel(PROJECT_ROOT / meta['base_dag'])}\n"
            + f"  target_replacement_pct: {metadata['target_replacement_pct']}\n"
            + f"  actual_replacement_pct: {metadata['actual_replacement_pct']}\n"
            + f"  capped_by_available_non_edges: {metadata['capped_by_available_non_edges']}\n"
            + f"  original_total_edges: {metadata['original_total_edges']}\n"
            + f"  new_total_edges: {metadata['new_total_edges']}\n"
            + f"  original_c2c_edges: {metadata['original_c2c_edges']}\n"
            + f"  available_original_c2c_non_edges: {metadata['available_original_c2c_non_edges']}\n"
            + f"  new_c2c_edges: {metadata['new_c2c_edges']}\n"
            + f"  target_removed_c2c_edges: {metadata['target_removed_c2c_edges']}\n"
            + f"  removed_c2c_edges: {metadata['removed_c2c_edges']}\n"
            + f"  added_c2c_edges: {metadata['added_c2c_edges']}\n"
            + f"  random_seed: {metadata['random_seed']}\n"
        )
        path = config_out_dir / f"{experiment_prefix}_{variant}.yaml"
        path.write_text(text)
        config_paths[variant] = path
        print(f"Wrote {path}")

    return config_paths


def write_run_scripts(dataset_key: str, meta: dict, config_paths: dict[str, Path]) -> None:
    out_dir = PROJECT_ROOT / "server_scripts" / "cream_experiment" / meta["server_dir"] / EXPERIMENT_FAMILY
    out_dir.mkdir(parents=True, exist_ok=True)
    experiment_prefix = f"CREAM_{dataset_key}_{EXPERIMENT_FAMILY}"

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

echo "Running {meta['display']} CREAM {EXPERIMENT_FAMILY}: {variant}"
"$PYTHON_BIN" simple_main.py --config "$CONFIG_PATH"

echo "Done!"
"""
        )

        log_prefix = f"{experiment_prefix}_{variant}"
        sub_path.write_text(
            f"""universe                = docker
docker_image            = {DOCKER_IMAGE}
initialdir              = {PROJECT_ROOT_SERVER}
executable              = {PROJECT_ROOT_SERVER}/server_scripts/cream_experiment/{meta['server_dir']}/{EXPERIMENT_FAMILY}/{run_name}

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
            f"condor_submit server_scripts/cream_experiment/{meta['server_dir']}/{EXPERIMENT_FAMILY}/{experiment_prefix}_{variant}_job.sub"
            for variant in config_paths
        )
        + "\n"
    )
    print(f"Wrote {submit_all}")


def parse_replacement_pcts(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 0 or value > 100:
            raise ValueError(f"Replacement percent must be in [0, 100], got {value}")
        if value not in values:
            values.append(value)
    return values


def generate_for_dataset(dataset_key: str, replacement_pcts: list[int], seed: int) -> None:
    meta = DATASETS[dataset_key]
    num_classes = NUM_CLASSES[dataset_key]
    base_dag = PROJECT_ROOT / meta["base_dag"]
    names, original = read_bool_dag(base_dag)
    size = len(names)
    num_concepts = size - num_classes
    original_total_edges = edge_count(original)
    original_c2c_edges = len(c2c_positions(original, num_concepts, True))

    print(f"\n=== {dataset_key} ===")
    print(
        f"Full DAG size={size}x{size}, concepts={num_concepts}, classes={num_classes}, "
        f"total edges={original_total_edges}, C--C edges={original_c2c_edges}"
    )

    out_dir = base_dag.parent / EXPERIMENT_FAMILY
    graph_paths = {}
    metadata_by_variant = {}
    rows = []
    for replacement_pct in replacement_pcts:
        variant = f"replace_{replacement_pct:03d}"
        random_seed = seed + replacement_pct
        matrix, metadata = replace_c2c_edges(
            original=original,
            num_concepts=num_concepts,
            replacement_pct=replacement_pct,
            seed=random_seed,
        )
        metadata.update(
            {
                "dataset": dataset_key,
                "experiment_type": EXPERIMENT_FAMILY,
                "variant": variant,
                "dag_path": rel(out_dir / f"{meta['dag_prefix']}_{variant}.csv"),
                "num_nodes": size,
                "num_concepts": num_concepts,
                "num_classes": num_classes,
                "original_total_edges": original_total_edges,
                "new_total_edges": edge_count(matrix),
            }
        )
        path = out_dir / f"{meta['dag_prefix']}_{variant}.csv"
        write_bool_dag(path, names, matrix)
        graph_paths[variant] = path
        metadata_by_variant[variant] = metadata
        rows.append(metadata)
        print(
            f"Wrote {path} | replacement={replacement_pct}% "
            f"removed={metadata['removed_c2c_edges']} added={metadata['added_c2c_edges']} "
            f"actual={metadata['actual_replacement_pct']:.2f}% "
            f"capped={metadata['capped_by_available_non_edges']} "
            f"C--C={metadata['new_c2c_edges']} total={metadata['new_total_edges']}"
        )

    write_metadata(out_dir / f"{EXPERIMENT_FAMILY}_metadata.csv", rows)
    config_paths = write_configs(dataset_key, meta, graph_paths, metadata_by_variant)
    write_run_scripts(dataset_key, meta, config_paths)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default="celeba,cub,cfmnist",
        help=f"Comma-separated list from: {','.join(DATASETS)}",
    )
    parser.add_argument(
        "--replacement_pcts",
        default=",".join(str(value) for value in DEFAULT_REPLACEMENT_PCTS),
        help="Comma-separated C--C replacement percentages. Default: 0,20,50,80,100.",
    )
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    dataset_keys = [key.strip().lower() for key in args.datasets.split(",") if key.strip()]
    unknown = sorted(set(dataset_keys) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown dataset keys: {unknown}. Available: {sorted(DATASETS)}")

    replacement_pcts = parse_replacement_pcts(args.replacement_pcts)
    for dataset_key in dataset_keys:
        generate_for_dataset(dataset_key, replacement_pcts, args.random_seed)


if __name__ == "__main__":
    main()
