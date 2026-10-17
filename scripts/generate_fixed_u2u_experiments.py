import argparse
import copy
from pathlib import Path

import yaml

from generate_graph_sanity import (
    CONDA_PYTHON_SERVER,
    DATASETS,
    DOCKER_IMAGE,
    PROJECT_ROOT,
    PROJECT_ROOT_SERVER,
)


FAMILIES = [
    "density_random_full_no_side_channel",
    "topology_random_full_no_side_channel",
    "concept_concept_density_no_side_channel",
    "concept_concept_fixcount_topology_no_side_channel",
    "concept_task_density_no_side_channel",
    "concept_task_reduced_topology_no_side_channel",
]
CACHE_ROOT = "./data/fixed_u2u_feature_cache"
CONFIG_ROOT = PROJECT_ROOT / "all_configs" / "fixed_u2u"
SCRIPT_ROOT = PROJECT_ROOT / "server_scripts" / "cream_experiment" / "fixed_u2u"


def read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=False)


def write_job(dataset_key: str, name: str, command: str, server_dir: str) -> None:
    folder = SCRIPT_ROOT / server_dir
    folder.mkdir(parents=True, exist_ok=True)
    run_path = folder / f"run_{name}.sh"
    sub_path = folder / f"{name}_job.sub"
    run_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        f'PROJECT_ROOT="{PROJECT_ROOT_SERVER}"\n'
        f'CONDA_PYTHON="{CONDA_PYTHON_SERVER}"\n\n'
        'if [ ! -x "$CONDA_PYTHON" ]; then\n'
        '    echo "ERROR: Conda environment not found: $CONDA_PYTHON" >&2\n'
        "    exit 127\n"
        "fi\n"
        'cd "$PROJECT_ROOT"\n'
        '"$CONDA_PYTHON" -V\n'
        'nvidia-smi || true\n'
        f"{command}\n",
        encoding="utf-8",
    )
    run_path.chmod(0o755)
    sub_path.write_text(
        "universe                = docker\n"
        f"docker_image            = {DOCKER_IMAGE}\n"
        f"initialdir              = {PROJECT_ROOT_SERVER}\n"
        f"executable              = {PROJECT_ROOT_SERVER}/server_scripts/cream_experiment/fixed_u2u/{server_dir}/{run_path.name}\n\n"
        f"output                  = {PROJECT_ROOT_SERVER}/logs/{name}.$(ClusterId).$(ProcId).out\n"
        f"error                   = {PROJECT_ROOT_SERVER}/logs/{name}.$(ClusterId).$(ProcId).err\n"
        f"log                     = {PROJECT_ROOT_SERVER}/logs/{name}.$(ClusterId).log\n\n"
        "request_GPUs            = 1\n"
        "request_CPUs            = 8\n"
        "request_memory          = 32G\n"
        'requirements            = UidDomain == "cs.uni-saarland.de"\n'
        "+WantGPUHomeMounted     = true\n"
        "queue 1\n",
        encoding="utf-8",
    )
    sub_path.chmod(0o755)


def first_source_config(dataset_key: str) -> Path:
    folder = PROJECT_ROOT / "all_configs" / "sanity_checks" / f"{dataset_key}_concept_concept_density_no_side_channel"
    candidates = sorted(folder.glob("*.yaml"))
    if not candidates:
        raise FileNotFoundError(f"No density no-side-channel configs found in {folder}")
    return candidates[0]


def make_reference_config(dataset_key: str, source: dict) -> tuple[dict, Path]:
    meta = DATASETS[dataset_key]
    experiment = f"CREAM_{dataset_key}_graph_free_u2u_reference"
    config = copy.deepcopy(source)
    config["experiment_name"] = experiment
    config["feature_cache_dir"] = f"{CACHE_ROOT}/{dataset_key}"
    config.pop("debug_masking", None)
    config.pop("sanity_check", None)
    config["paths"] = {
        "default_root_dir": "./experiments/",
        "metric_dir": "./last_metrics/",
    }
    if "softmax_mask" in source.get("paths", {}):
        config["paths"]["softmax_mask"] = source["paths"]["softmax_mask"]
    model2 = config["hyperparameters_model2"]
    model2["masking_algorithm"] = "none"
    model2["last_layer_mask"] = False
    model2["side_dropout"] = False
    model2["dropout_prob"] = 0.0
    config["save_u2u_reference"] = (
        f"./experiments/{config['dataset_name']}/train_cbm/{config['model_name']}"
        f"/fixed_u2u_reference/{experiment}/shared_u2u.pt"
    )
    path = CONFIG_ROOT.parent / "fixed_u2u_reference" / f"{dataset_key}_graph_free_u2u.yaml"
    return config, path


def make_u2c_reference_config(dataset_key: str, source: dict) -> tuple[dict, Path]:
    config, _ = make_reference_config(dataset_key, source)
    experiment = f"CREAM_{dataset_key}_graph_free_u2u_u2c_reference"
    reference_root = (
        f"./experiments/{config['dataset_name']}/train_cbm/{config['model_name']}"
        f"/fixed_u2u_fixed_u2c_reference/{experiment}"
    )
    config["experiment_name"] = experiment
    config["save_u2u_reference"] = f"{reference_root}/shared_u2u.pt"
    config["save_u2c_reference"] = f"{reference_root}/shared_u2c.pt"
    path = (
        CONFIG_ROOT.parent
        / "fixed_u2u_fixed_u2c_reference"
        / f"{dataset_key}_graph_free_u2u_u2c.yaml"
    )
    return config, path


def generate_dataset(dataset_key: str) -> tuple[list[str], list[str]]:
    meta = DATASETS[dataset_key]
    source_path = first_source_config(dataset_key)
    source = read_yaml(source_path)
    cache_dir = f"{CACHE_ROOT}/{dataset_key}"

    cache_name = f"cache_{dataset_key}_frozen_backbone_features"
    cache_config_rel = source_path.relative_to(PROJECT_ROOT).as_posix()
    write_job(
        dataset_key,
        cache_name,
        f'"$CONDA_PYTHON" scripts/cache_frozen_backbone_features.py --config {cache_config_rel} --output-dir {cache_dir}',
        f"{meta['server_dir']}/cache",
    )

    reference, reference_path = make_reference_config(dataset_key, source)
    write_yaml(reference_path, reference)
    reference_name = f"train_{dataset_key}_graph_free_u2u_reference"
    write_job(
        dataset_key,
        reference_name,
        f'"$CONDA_PYTHON" simple_main.py --config {reference_path.relative_to(PROJECT_ROOT).as_posix()}',
        f"{meta['server_dir']}/reference",
    )

    fixed_u2c_reference, fixed_u2c_reference_path = make_u2c_reference_config(
        dataset_key, source
    )
    write_yaml(fixed_u2c_reference_path, fixed_u2c_reference)
    fixed_u2c_reference_name = f"train_{dataset_key}_graph_free_u2u_u2c_reference"
    write_job(
        dataset_key,
        fixed_u2c_reference_name,
        f'"$CONDA_PYTHON" simple_main.py --config {fixed_u2c_reference_path.relative_to(PROJECT_ROOT).as_posix()}',
        f"{meta['server_dir']}/reference_fixed_u2c",
    )

    submit_cache = SCRIPT_ROOT / f"submit_cache_{dataset_key}_features.sh"
    submit_cache.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"condor_submit server_scripts/cream_experiment/fixed_u2u/{meta['server_dir']}/cache/{cache_name}_job.sub\n",
        encoding="utf-8",
    )
    submit_cache.chmod(0o755)
    submit_reference = SCRIPT_ROOT / f"submit_reference_{dataset_key}_u2u.sh"
    submit_reference.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"condor_submit server_scripts/cream_experiment/fixed_u2u/{meta['server_dir']}/reference/{reference_name}_job.sub\n",
        encoding="utf-8",
    )
    submit_reference.chmod(0o755)
    submit_fixed_u2c_reference = (
        SCRIPT_ROOT / f"submit_reference_{dataset_key}_u2u_u2c.sh"
    )
    submit_fixed_u2c_reference.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"condor_submit server_scripts/cream_experiment/fixed_u2u/{meta['server_dir']}/reference_fixed_u2c/{fixed_u2c_reference_name}_job.sub\n",
        encoding="utf-8",
    )
    submit_fixed_u2c_reference.chmod(0o755)

    generated_submits = []
    generated_fixed_u2c_submits = []
    for family in FAMILIES:
        source_dir = PROJECT_ROOT / "all_configs" / "sanity_checks" / f"{dataset_key}_{family}"
        configs = sorted(source_dir.glob("*.yaml"))
        if not configs:
            print(f"Skipping missing family configs: {source_dir}")
            continue
        output_family = f"{family}_fixed_u2u"
        fixed_u2c_family = f"{family}_fixed_u2u_fixed_u2c"
        output_dir = CONFIG_ROOT / f"{dataset_key}_{output_family}"
        server_dir = PROJECT_ROOT / "server_scripts" / "cream_experiment" / meta["server_dir"] / output_family
        submit_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
        fixed_u2c_output_dir = (
            CONFIG_ROOT.parent / "fixed_u2u_fixed_u2c" / f"{dataset_key}_{fixed_u2c_family}"
        )
        fixed_u2c_server_dir = (
            PROJECT_ROOT
            / "server_scripts"
            / "cream_experiment"
            / meta["server_dir"]
            / fixed_u2c_family
        )
        fixed_u2c_submit_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
        for config_path in configs:
            config = read_yaml(config_path)
            old_experiment = config["experiment_name"]
            prefix, variant = old_experiment.split("/", 1)
            new_prefix = f"{prefix}_fixed_u2u"
            config["experiment_name"] = f"{new_prefix}/{variant}"
            config["feature_cache_dir"] = cache_dir
            config["paths"]["shared_u2u_file"] = reference["save_u2u_reference"]
            config["sanity_check"] = dict(config.get("sanity_check", {}))
            config["sanity_check"]["fixed_u2u_reference"] = reference["save_u2u_reference"]
            config["sanity_check"]["cached_backbone_features"] = cache_dir
            out_path = output_dir / config_path.name.replace(
                "_no_side_channel_", "_no_side_channel_fixed_u2u_"
            )
            if not out_path.exists():
                write_yaml(out_path, config)

            job_name = f"{new_prefix}_{variant}"
            run_path = server_dir / f"run_{job_name}.sh"
            sub_path = server_dir / f"{job_name}_job.sub"
            server_dir.mkdir(parents=True, exist_ok=True)
            config_rel = out_path.relative_to(PROJECT_ROOT).as_posix()
            run_path.write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\n\n"
                f'PROJECT_ROOT="{PROJECT_ROOT_SERVER}"\n'
                f'CONDA_PYTHON="{CONDA_PYTHON_SERVER}"\n'
                'cd "$PROJECT_ROOT"\n'
                '"$CONDA_PYTHON" simple_main.py --config '
                f'"{config_rel}"\n',
                encoding="utf-8",
            )
            run_path.chmod(0o755)
            sub_path.write_text(
                "universe                = docker\n"
                f"docker_image            = {DOCKER_IMAGE}\n"
                f"initialdir              = {PROJECT_ROOT_SERVER}\n"
                f"executable              = {PROJECT_ROOT_SERVER}/server_scripts/cream_experiment/{meta['server_dir']}/{output_family}/{run_path.name}\n\n"
                f"output                  = {PROJECT_ROOT_SERVER}/logs/{job_name}.$(ClusterId).$(ProcId).out\n"
                f"error                   = {PROJECT_ROOT_SERVER}/logs/{job_name}.$(ClusterId).$(ProcId).err\n"
                f"log                     = {PROJECT_ROOT_SERVER}/logs/{job_name}.$(ClusterId).log\n\n"
                "request_GPUs            = 1\nrequest_CPUs            = 8\nrequest_memory          = 32G\n"
                'requirements            = UidDomain == "cs.uni-saarland.de"\n'
                "+WantGPUHomeMounted     = true\nqueue 1\n",
                encoding="utf-8",
            )
            sub_path.chmod(0o755)
            submit_lines.append(
                f"condor_submit server_scripts/cream_experiment/{meta['server_dir']}/{output_family}/{sub_path.name}"
            )

            fixed_u2c_config = copy.deepcopy(config)
            fixed_u2c_prefix = f"{new_prefix}_fixed_u2c"
            fixed_u2c_config["experiment_name"] = f"{fixed_u2c_prefix}/{variant}"
            fixed_u2c_config["paths"]["shared_u2u_file"] = fixed_u2c_reference[
                "save_u2u_reference"
            ]
            fixed_u2c_config["paths"]["shared_u2c_file"] = fixed_u2c_reference[
                "save_u2c_reference"
            ]
            fixed_u2c_config["sanity_check"]["fixed_u2c_reference"] = (
                fixed_u2c_reference["save_u2c_reference"]
            )
            fixed_u2c_config["sanity_check"]["fixed_u2u_reference"] = (
                fixed_u2c_reference["save_u2u_reference"]
            )
            fixed_u2c_out_path = fixed_u2c_output_dir / config_path.name.replace(
                "_no_side_channel_fixed_u2u_",
                "_no_side_channel_fixed_u2u_fixed_u2c_",
            )
            write_yaml(fixed_u2c_out_path, fixed_u2c_config)

            fixed_u2c_job_name = f"{fixed_u2c_prefix}_{variant}"
            fixed_u2c_server_dir.mkdir(parents=True, exist_ok=True)
            fixed_u2c_run_path = fixed_u2c_server_dir / f"run_{fixed_u2c_job_name}.sh"
            fixed_u2c_sub_path = fixed_u2c_server_dir / f"{fixed_u2c_job_name}_job.sub"
            fixed_u2c_config_rel = fixed_u2c_out_path.relative_to(PROJECT_ROOT).as_posix()
            fixed_u2c_run_path.write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\n\n"
                f'PROJECT_ROOT="{PROJECT_ROOT_SERVER}"\n'
                f'CONDA_PYTHON="{CONDA_PYTHON_SERVER}"\n'
                'cd "$PROJECT_ROOT"\n'
                '"$CONDA_PYTHON" simple_main.py --config '
                f'"{fixed_u2c_config_rel}"\n',
                encoding="utf-8",
            )
            fixed_u2c_run_path.chmod(0o755)
            fixed_u2c_sub_path.write_text(
                "universe                = docker\n"
                f"docker_image            = {DOCKER_IMAGE}\n"
                f"initialdir              = {PROJECT_ROOT_SERVER}\n"
                f"executable              = {PROJECT_ROOT_SERVER}/server_scripts/cream_experiment/{meta['server_dir']}/{fixed_u2c_family}/{fixed_u2c_run_path.name}\n\n"
                f"output                  = {PROJECT_ROOT_SERVER}/logs/{fixed_u2c_job_name}.$(ClusterId).$(ProcId).out\n"
                f"error                   = {PROJECT_ROOT_SERVER}/logs/{fixed_u2c_job_name}.$(ClusterId).$(ProcId).err\n"
                f"log                     = {PROJECT_ROOT_SERVER}/logs/{fixed_u2c_job_name}.$(ClusterId).log\n\n"
                "request_GPUs            = 1\nrequest_CPUs            = 8\nrequest_memory          = 32G\n"
                'requirements            = UidDomain == "cs.uni-saarland.de"\n'
                "+WantGPUHomeMounted     = true\nqueue 1\n",
                encoding="utf-8",
            )
            fixed_u2c_sub_path.chmod(0o755)
            fixed_u2c_submit_lines.append(
                f"condor_submit server_scripts/cream_experiment/{meta['server_dir']}/{fixed_u2c_family}/{fixed_u2c_sub_path.name}"
            )
        submit_path = server_dir / f"submit_all_{dataset_key}_{output_family}.sh"
        submit_path.write_text("\n".join(submit_lines) + "\n", encoding="utf-8")
        submit_path.chmod(0o755)
        generated_submits.append(
            f"server_scripts/cream_experiment/{meta['server_dir']}/{output_family}/{submit_path.name}"
        )
        fixed_u2c_submit_path = fixed_u2c_server_dir / (
            f"submit_all_{dataset_key}_{fixed_u2c_family}.sh"
        )
        fixed_u2c_submit_path.parent.mkdir(parents=True, exist_ok=True)
        fixed_u2c_submit_path.write_text(
            "\n".join(fixed_u2c_submit_lines) + "\n", encoding="utf-8"
        )
        fixed_u2c_submit_path.chmod(0o755)
        generated_fixed_u2c_submits.append(
            f"server_scripts/cream_experiment/{meta['server_dir']}/{fixed_u2c_family}/{fixed_u2c_submit_path.name}"
        )

    return generated_submits, generated_fixed_u2c_submits, str(submit_fixed_u2c_reference.relative_to(PROJECT_ROOT).as_posix())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="celeba,cub,cfmnist")
    args = parser.parse_args()
    keys = [part.strip().lower() for part in args.datasets.split(",") if part.strip()]
    unknown = sorted(set(keys) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")

    all_submits = []
    all_fixed_u2c_submits = []
    fixed_u2c_reference_submits = []
    cache_submits = []
    reference_submits = []
    for key in keys:
        generated, generated_fixed_u2c, fixed_u2c_reference_submit = generate_dataset(key)
        all_submits.extend(generated)
        all_fixed_u2c_submits.extend(generated_fixed_u2c)
        fixed_u2c_reference_submits.append(fixed_u2c_reference_submit)
        meta = DATASETS[key]
        cache_submits.append(
            f"server_scripts/cream_experiment/fixed_u2u/submit_cache_{key}_features.sh"
        )
        reference_submits.append(
            f"server_scripts/cream_experiment/fixed_u2u/submit_reference_{key}_u2u.sh"
        )

    for name, lines in (
        ("submit_all_feature_caches.sh", cache_submits),
        ("submit_all_u2u_references.sh", reference_submits),
        ("submit_all_u2u_u2c_references.sh", fixed_u2c_reference_submits),
        ("submit_all_graph_variants_fixed_u2u.sh", all_submits),
        ("submit_all_graph_variants_fixed_u2u_fixed_u2c.sh", all_fixed_u2c_submits),
    ):
        path = SCRIPT_ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n\n"
            + "\n".join(f"bash {line}" for line in lines)
            + "\n",
            encoding="utf-8",
        )
        path.chmod(0o755)

    print(f"Generated fixed-U2U and fixed-U2U+U2C configs/jobs for {', '.join(keys)}")
    print("Run cache jobs first, then reference jobs, then the desired graph-variant jobs.")


if __name__ == "__main__":
    main()
