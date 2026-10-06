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


DENSITY_FAMILY = "concept_task_density"
TOPOLOGY_FAMILY = "concept_task_fixcount_topology"
REDUCED_TOPOLOGY_FAMILY = "concept_task_reduced_topology"
DEFAULT_REPLACEMENT_PCTS = [0, 20, 50, 80, 100]
NUM_CLASSES = {
    "celeba": 1,
    "cub": 200,
    "cfmnist": 10,
}


def edge_count(matrix: list[list[bool]]) -> int:
    return sum(sum(row) for row in matrix)


def cy_positions(num_concepts: int, num_classes: int) -> list[tuple[int, int]]:
    return [
        (num_concepts + task_idx, concept_idx)
        for task_idx in range(num_classes)
        for concept_idx in range(num_concepts)
    ]


def cy_edge_count(matrix: list[list[bool]], num_concepts: int, num_classes: int) -> int:
    return sum(1 for row, col in cy_positions(num_concepts, num_classes) if matrix[row][col])


def default_density_counts(original_cy_edges: int, total_cy_positions: int) -> list[int]:
    counts = []
    for multiplier in [0.5, 0.75, 1.0, 1.25, 1.5]:
        count = max(1, min(total_cy_positions, round(original_cy_edges * multiplier)))
        if count not in counts:
            counts.append(count)
    return counts


def parse_density_counts(raw: str | None, original_cy_edges: int, total_cy_positions: int) -> list[int]:
    if not raw:
        return default_density_counts(original_cy_edges, total_cy_positions)
    counts = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        count = int(part)
        if count < 0 or count > total_cy_positions:
            raise ValueError(f"C--Y edge count {count} outside [0, {total_cy_positions}]")
        if count not in counts:
            counts.append(count)
    return counts


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


def make_random_cy_density_graph(
    original: list[list[bool]],
    num_concepts: int,
    num_classes: int,
    cy_edges: int,
    seed: int,
) -> list[list[bool]]:
    positions = cy_positions(num_concepts, num_classes)
    rng = random.Random(seed)
    chosen = set(rng.sample(positions, cy_edges))
    matrix = deepcopy(original)
    for row, col in positions:
        matrix[row][col] = (row, col) in chosen
    return matrix


def replace_cy_edges(
    original: list[list[bool]],
    num_concepts: int,
    num_classes: int,
    replacement_pct: int,
    seed: int,
) -> tuple[list[list[bool]], dict]:
    positions = cy_positions(num_concepts, num_classes)
    true_positions = [(row, col) for row, col in positions if original[row][col]]
    false_positions = [(row, col) for row, col in positions if not original[row][col]]
    original_cy_edges = len(true_positions)
    target_replace_count = round(original_cy_edges * replacement_pct / 100)
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

    return matrix, {
        "target_replacement_pct": replacement_pct,
        "actual_replacement_pct": replace_count / original_cy_edges * 100 if original_cy_edges else 0.0,
        "target_removed_cy_edges": target_replace_count,
        "capped_by_available_non_edges": capped,
        "original_cy_edges": original_cy_edges,
        "available_original_cy_non_edges": len(false_positions),
        "removed_cy_edges": len(removed),
        "added_cy_edges": len(added),
        "new_cy_edges": cy_edge_count(matrix, num_concepts, num_classes),
        "random_seed": seed,
    }


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
    experiment_prefix = f"CREAM_{dataset_key}_{experiment_family}"
    config_paths = {}

    for variant, dag_path in graph_paths.items():
        metadata = metadata_by_variant[variant]
        text = upsert_experiment_name(base_text, f"{experiment_prefix}/{variant}")
        text = replace_dag_path(text, dag_path)
        text = (
            text.rstrip()
            + "\n"
            + "sanity_check:\n"
            + f"  dataset: {dataset_key}\n"
            + f"  experiment_type: {experiment_family}\n"
            + f"  graph_variant: {variant}\n"
            + "  random_scope: concept_task_block_only\n"
            + "  fixed_blocks: c2c_and_all_non_cy_entries\n"
            + f"  base_dag: {rel(PROJECT_ROOT / meta['base_dag'])}\n"
            + f"  original_total_edges: {metadata['original_total_edges']}\n"
            + f"  new_total_edges: {metadata['new_total_edges']}\n"
            + f"  original_cy_edges: {metadata['original_cy_edges']}\n"
            + f"  new_cy_edges: {metadata['new_cy_edges']}\n"
            + f"  total_cy_positions: {metadata['total_cy_positions']}\n"
            + f"  random_seed: {metadata['random_seed']}\n"
        )
        if "cy_density" in metadata:
            text += (
                f"  cy_density: {metadata['cy_density']}\n"
                f"  cy_edge_multiplier: {metadata['cy_edge_multiplier']}\n"
            )
        if "target_replacement_pct" in metadata:
            text += (
                f"  target_replacement_pct: {metadata['target_replacement_pct']}\n"
                f"  actual_replacement_pct: {metadata['actual_replacement_pct']}\n"
                f"  capped_by_available_non_edges: {metadata['capped_by_available_non_edges']}\n"
                f"  removed_cy_edges: {metadata['removed_cy_edges']}\n"
                f"  added_cy_edges: {metadata['added_cy_edges']}\n"
            )
        path = config_out_dir / f"{experiment_prefix}_{variant}.yaml"
        path.write_text(text)
        config_paths[variant] = path
        print(f"Wrote {path}")
    return config_paths


def write_run_scripts(dataset_key: str, meta: dict, experiment_family: str, config_paths: dict[str, Path]) -> None:
    out_dir = PROJECT_ROOT / "server_scripts" / "cream_experiment" / meta["server_dir"] / experiment_family
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


def generate_density(dataset_key: str, density_counts_raw: str | None, seed: int) -> None:
    meta = DATASETS[dataset_key]
    num_classes = NUM_CLASSES[dataset_key]
    base_dag = PROJECT_ROOT / meta["base_dag"]
    names, original = read_bool_dag(base_dag)
    num_concepts = len(names) - num_classes
    original_total_edges = edge_count(original)
    original_cy_edges = cy_edge_count(original, num_concepts, num_classes)
    total_cy_positions = num_concepts * num_classes
    counts = parse_density_counts(density_counts_raw, original_cy_edges, total_cy_positions)

    out_dir = base_dag.parent / DENSITY_FAMILY
    graph_paths = {}
    metadata_by_variant = {}
    rows = []
    for idx, count in enumerate(counts):
        variant = f"cy_edges_{count:05d}"
        random_seed = seed + 20_000 + idx
        matrix = make_random_cy_density_graph(original, num_concepts, num_classes, count, random_seed)
        metadata = {
            "dataset": dataset_key,
            "experiment_type": DENSITY_FAMILY,
            "variant": variant,
            "dag_path": rel(out_dir / f"{meta['dag_prefix']}_{variant}.csv"),
            "num_nodes": len(names),
            "num_concepts": num_concepts,
            "num_classes": num_classes,
            "original_total_edges": original_total_edges,
            "new_total_edges": edge_count(matrix),
            "original_cy_edges": original_cy_edges,
            "new_cy_edges": cy_edge_count(matrix, num_concepts, num_classes),
            "total_cy_positions": total_cy_positions,
            "cy_density": count / total_cy_positions if total_cy_positions else 0.0,
            "cy_edge_multiplier": count / original_cy_edges if original_cy_edges else 0.0,
            "random_seed": random_seed,
        }
        path = out_dir / f"{meta['dag_prefix']}_{variant}.csv"
        write_bool_dag(path, names, matrix)
        graph_paths[variant] = path
        metadata_by_variant[variant] = metadata
        rows.append(metadata)
        print(f"Wrote {path} | C--Y edges={metadata['new_cy_edges']} total={metadata['new_total_edges']}")

    write_metadata(out_dir / f"{DENSITY_FAMILY}_metadata.csv", rows)
    configs = write_configs(dataset_key, meta, DENSITY_FAMILY, graph_paths, metadata_by_variant)
    write_run_scripts(dataset_key, meta, DENSITY_FAMILY, configs)


def generate_topology(dataset_key: str, replacement_pcts: list[int], seed: int) -> None:
    meta = DATASETS[dataset_key]
    num_classes = NUM_CLASSES[dataset_key]
    base_dag = PROJECT_ROOT / meta["base_dag"]
    names, original = read_bool_dag(base_dag)
    num_concepts = len(names) - num_classes
    original_total_edges = edge_count(original)
    original_cy_edges = cy_edge_count(original, num_concepts, num_classes)
    total_cy_positions = num_concepts * num_classes

    out_dir = base_dag.parent / TOPOLOGY_FAMILY
    graph_paths = {}
    metadata_by_variant = {}
    rows = []
    for replacement_pct in replacement_pcts:
        variant = f"replace_{replacement_pct:03d}"
        random_seed = seed + 30_000 + replacement_pct
        matrix, metadata = replace_cy_edges(original, num_concepts, num_classes, replacement_pct, random_seed)
        metadata.update(
            {
                "dataset": dataset_key,
                "experiment_type": TOPOLOGY_FAMILY,
                "variant": variant,
                "dag_path": rel(out_dir / f"{meta['dag_prefix']}_{variant}.csv"),
                "num_nodes": len(names),
                "num_concepts": num_concepts,
                "num_classes": num_classes,
                "original_total_edges": original_total_edges,
                "new_total_edges": edge_count(matrix),
                "total_cy_positions": total_cy_positions,
            }
        )
        path = out_dir / f"{meta['dag_prefix']}_{variant}.csv"
        write_bool_dag(path, names, matrix)
        graph_paths[variant] = path
        metadata_by_variant[variant] = metadata
        rows.append(metadata)
        print(
            f"Wrote {path} | replacement={replacement_pct}% "
            f"removed={metadata['removed_cy_edges']} added={metadata['added_cy_edges']} "
            f"actual={metadata['actual_replacement_pct']:.2f}% capped={metadata['capped_by_available_non_edges']}"
        )

    write_metadata(out_dir / f"{TOPOLOGY_FAMILY}_metadata.csv", rows)
    configs = write_configs(dataset_key, meta, TOPOLOGY_FAMILY, graph_paths, metadata_by_variant)
    write_run_scripts(dataset_key, meta, TOPOLOGY_FAMILY, configs)


def generate_reduced_topology(
    dataset_key: str,
    target_fraction: float,
    num_graphs: int,
    seed: int,
) -> None:
    meta = DATASETS[dataset_key]
    num_classes = NUM_CLASSES[dataset_key]
    base_dag = PROJECT_ROOT / meta["base_dag"]
    names, original = read_bool_dag(base_dag)
    num_concepts = len(names) - num_classes
    original_total_edges = edge_count(original)
    original_cy_edges = cy_edge_count(original, num_concepts, num_classes)
    total_cy_positions = num_concepts * num_classes
    target_cy_edges = max(1, min(total_cy_positions, round(original_cy_edges * target_fraction)))

    out_dir = base_dag.parent / REDUCED_TOPOLOGY_FAMILY
    graph_paths = {}
    metadata_by_variant = {}
    rows = []
    pct_label = int(round(target_fraction * 100))
    for idx in range(num_graphs):
        variant = f"cy_topology_{pct_label:03d}_{idx:02d}"
        random_seed = seed + 40_000 + idx
        matrix = make_random_cy_density_graph(
            original=original,
            num_concepts=num_concepts,
            num_classes=num_classes,
            cy_edges=target_cy_edges,
            seed=random_seed,
        )
        new_cy_edges = cy_edge_count(matrix, num_concepts, num_classes)
        metadata = {
            "dataset": dataset_key,
            "experiment_type": REDUCED_TOPOLOGY_FAMILY,
            "variant": variant,
            "dag_path": rel(out_dir / f"{meta['dag_prefix']}_{variant}.csv"),
            "num_nodes": len(names),
            "num_concepts": num_concepts,
            "num_classes": num_classes,
            "original_total_edges": original_total_edges,
            "new_total_edges": edge_count(matrix),
            "original_cy_edges": original_cy_edges,
            "new_cy_edges": new_cy_edges,
            "total_cy_positions": total_cy_positions,
            "target_cy_edge_fraction": target_fraction,
            "target_cy_edges": target_cy_edges,
            "cy_density": new_cy_edges / total_cy_positions if total_cy_positions else 0.0,
            "cy_edge_multiplier": new_cy_edges / original_cy_edges if original_cy_edges else 0.0,
            "random_seed": random_seed,
        }
        path = out_dir / f"{meta['dag_prefix']}_{variant}.csv"
        write_bool_dag(path, names, matrix)
        graph_paths[variant] = path
        metadata_by_variant[variant] = metadata
        rows.append(metadata)
        print(
            f"Wrote {path} | fixed C--Y topology density={target_fraction:.2f} "
            f"C--Y edges={new_cy_edges} total={metadata['new_total_edges']}"
        )

    write_metadata(out_dir / f"{REDUCED_TOPOLOGY_FAMILY}_metadata.csv", rows)
    configs = write_configs(dataset_key, meta, REDUCED_TOPOLOGY_FAMILY, graph_paths, metadata_by_variant)
    write_run_scripts(dataset_key, meta, REDUCED_TOPOLOGY_FAMILY, configs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="celeba,cub,cfmnist")
    parser.add_argument("--density_edge_counts", default=None)
    parser.add_argument(
        "--replacement_pcts",
        default=",".join(str(value) for value in DEFAULT_REPLACEMENT_PCTS),
    )
    parser.add_argument("--reduced_topology_fraction", type=float, default=0.75)
    parser.add_argument("--reduced_topology_graphs", type=int, default=5)
    parser.add_argument(
        "--include_degenerate_replace_topology",
        action="store_true",
        help="Also generate the old replace-style C--Y topology family. This is degenerate when the C--Y block is fully connected.",
    )
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    dataset_keys = [key.strip().lower() for key in args.datasets.split(",") if key.strip()]
    unknown = sorted(set(dataset_keys) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown dataset keys: {unknown}. Available: {sorted(DATASETS)}")
    replacement_pcts = parse_replacement_pcts(args.replacement_pcts)

    for dataset_key in dataset_keys:
        print(f"\n=== {dataset_key}: C--Y density ===")
        generate_density(dataset_key, args.density_edge_counts, args.random_seed)
        print(f"\n=== {dataset_key}: C--Y reduced-density topology ===")
        generate_reduced_topology(
            dataset_key=dataset_key,
            target_fraction=args.reduced_topology_fraction,
            num_graphs=args.reduced_topology_graphs,
            seed=args.random_seed,
        )
        if args.include_degenerate_replace_topology:
            print(f"\n=== {dataset_key}: C--Y replace-style topology ===")
            generate_topology(dataset_key, replacement_pcts, args.random_seed)


if __name__ == "__main__":
    main()
