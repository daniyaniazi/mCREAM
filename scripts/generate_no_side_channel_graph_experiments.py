import argparse
from pathlib import Path

from generate_graph_sanity import (
    CONDA_PYTHON_SERVER,
    DATASETS,
    DOCKER_IMAGE,
    PROJECT_ROOT,
    PROJECT_ROOT_SERVER,
    replace_dag_path,
    upsert_experiment_name,
)


NO_SIDE_SUFFIX = "_no_side_channel"
DEFAULT_FAMILIES = [
    "density_random_full",
    "topology_random_full",
    "concept_concept_density",
    "concept_concept_fixcount_topology",
    "concept_task_density",
    "concept_task_reduced_topology",
]

NO_SIDE_BASE_CONFIGS = {
    "celeba": "all_configs/best_hparams/CREAM_no_side_channel/CREAM_no_side_best_celeba.yaml",
    "cub": "all_configs/best_hparams/CREAM_no_side_channel/CREAM_no_side_best_cub_soft_config.yaml",
    "cfmnist": "all_configs/best_hparams/CREAM_no_side_channel/CREAM_no_side_best_cfmnist_soft_config.yaml",
}


def extract_dag_path(config_text: str) -> Path:
    for line in config_text.splitlines():
        if line.strip().startswith("DAG_file:"):
            raw = line.split(":", 1)[1].strip()
            return PROJECT_ROOT / raw.removeprefix("./")
    raise ValueError("Could not find paths.DAG_file in source config.")


def extract_sanity_block(config_text: str) -> str:
    lines = config_text.splitlines()
    for idx, line in enumerate(lines):
        if line.startswith("sanity_check:"):
            return "\n".join(lines[idx:]).rstrip() + "\n"
    return ""


def variant_from_config_name(dataset_key: str, family: str, path: Path) -> str:
    prefix = f"CREAM_{dataset_key}_{family}_"
    stem = path.stem
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected config name for {dataset_key}/{family}: {path.name}")
    return stem[len(prefix) :]


def make_config_text(dataset_key: str, family: str, variant: str, source_text: str) -> str:
    no_side_family = family + NO_SIDE_SUFFIX
    experiment_prefix = f"CREAM_{dataset_key}_{no_side_family}"
    experiment_name = f"{experiment_prefix}/{variant}"
    dag_path = extract_dag_path(source_text)
    sanity_block = extract_sanity_block(source_text)

    base_text = (PROJECT_ROOT / NO_SIDE_BASE_CONFIGS[dataset_key]).read_text()
    text = upsert_experiment_name(base_text, experiment_name)
    text = replace_dag_path(text, dag_path)
    if sanity_block:
        text = (
            text.rstrip()
            + "\n"
            + sanity_block.rstrip()
            + "\n"
            + "  no_side_channel: true\n"
            + f"  source_experiment_type: {family}\n"
        )
    return text


def write_run_scripts(dataset_key: str, meta: dict, family: str, config_paths: dict[str, Path]) -> None:
    no_side_family = family + NO_SIDE_SUFFIX
    experiment_prefix = f"CREAM_{dataset_key}_{no_side_family}"
    out_dir = PROJECT_ROOT / "server_scripts" / "cream_experiment" / meta["server_dir"] / no_side_family
    out_dir.mkdir(parents=True, exist_ok=True)

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

echo "Running {meta['display']} CREAM {no_side_family}: {variant}"
"$PYTHON_BIN" simple_main.py --config "$CONFIG_PATH"

echo "Done!"
"""
        )

        log_prefix = f"{experiment_prefix}_{variant}"
        sub_path.write_text(
            f"""universe                = docker
docker_image            = {DOCKER_IMAGE}
initialdir              = {PROJECT_ROOT_SERVER}
executable              = {PROJECT_ROOT_SERVER}/server_scripts/cream_experiment/{meta['server_dir']}/{no_side_family}/{run_name}

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
            f"condor_submit server_scripts/cream_experiment/{meta['server_dir']}/{no_side_family}/{experiment_prefix}_{variant}_job.sub"
            for variant in config_paths
        )
        + "\n"
    )
    print(f"Wrote {submit_all}")


def generate_for_family(dataset_key: str, family: str) -> None:
    meta = DATASETS[dataset_key]
    source_dir = PROJECT_ROOT / "all_configs" / "sanity_checks" / f"{dataset_key}_{family}"
    if not source_dir.exists():
        print(f"Skipping missing source config dir: {source_dir}")
        return

    out_dir = PROJECT_ROOT / "all_configs" / "sanity_checks" / f"{dataset_key}_{family}{NO_SIDE_SUFFIX}"
    out_dir.mkdir(parents=True, exist_ok=True)
    experiment_prefix = f"CREAM_{dataset_key}_{family}{NO_SIDE_SUFFIX}"
    config_paths = {}

    for source_path in sorted(source_dir.glob("*.yaml")):
        variant = variant_from_config_name(dataset_key, family, source_path)
        text = make_config_text(dataset_key, family, variant, source_path.read_text())
        out_path = out_dir / f"{experiment_prefix}_{variant}.yaml"
        out_path.write_text(text)
        config_paths[variant] = out_path
        print(f"Wrote {out_path}")

    if config_paths:
        write_run_scripts(dataset_key, meta, family, config_paths)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="celeba,cub,cfmnist")
    parser.add_argument("--families", default=",".join(DEFAULT_FAMILIES))
    args = parser.parse_args()

    dataset_keys = [key.strip().lower() for key in args.datasets.split(",") if key.strip()]
    family_keys = [key.strip() for key in args.families.split(",") if key.strip()]
    unknown = sorted(set(dataset_keys) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown dataset keys: {unknown}. Available: {sorted(DATASETS)}")

    for dataset_key in dataset_keys:
        for family in family_keys:
            generate_for_family(dataset_key, family)


if __name__ == "__main__":
    main()
