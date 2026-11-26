"""Generate fixed-backbone CREAM retraining configs/jobs from notebook selections."""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-config", required=True, type=Path)
    parser.add_argument("--selected-csv", type=Path, default=None,
                        help="Notebook-exported selections; defaults to output_dir/selected_edges_for_retraining.csv")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    analysis_path = args.analysis_config if args.analysis_config.is_absolute() else root / args.analysis_config
    analysis = yaml.safe_load(analysis_path.read_text(encoding="utf-8"))
    out_dir = Path(analysis["output_dir"])
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    selected_path = args.selected_csv or (out_dir / "selected_edges_for_retraining.csv")
    if not selected_path.is_absolute():
        selected_path = root / selected_path
    if not selected_path.is_file():
        raise FileNotFoundError(f"Notebook selection file not found: {selected_path}")
    selected = pd.read_csv(selected_path)
    required = {"model", "edge_id", "source_index", "target_index", "source_concept", "target_concept"}
    missing = required - set(selected.columns)
    if missing:
        raise ValueError(f"Selected CSV lacks required columns: {sorted(missing)}")

    model_specs = {m["name"]: m for m in analysis["models"]}
    seeds = [int(s) for s in analysis.get("training_seeds", [42])]
    generated_configs = []
    graph_root = out_dir / "generated_graphs"
    config_root = root / "all_configs" / "edge_sensitivity" / analysis["dataset"].lower().replace("_", "")
    config_root.mkdir(parents=True, exist_ok=True)
    graph_root.mkdir(parents=True, exist_ok=True)

    for _, row in selected.iterrows():
        model_name = str(row["model"])
        if model_name not in model_specs:
            raise ValueError(f"Selected model {model_name!r} is absent from analysis config")
        spec = model_specs[model_name]
        base_cfg_path = Path(spec["config"])
        if not base_cfg_path.is_absolute():
            base_cfg_path = root / base_cfg_path
        cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))
        dag_path = Path(cfg["paths"]["DAG_file"])
        if not dag_path.is_absolute():
            dag_path = root / dag_path
        dag = pd.read_csv(dag_path, index_col=0)
        source, target = int(row["source_index"]), int(row["target_index"])
        if bool(dag.iloc[target, source]) is False:
            raise ValueError(f"Edge {row['edge_id']} is not present at target-row/source-column [{target},{source}]")
        dag.iloc[target, source] = 0
        edge_id = str(row["edge_id"])
        graph_path = graph_root / model_name / f"remove_{edge_id}.csv"
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        dag.to_csv(graph_path)

        for seed in seeds:
            job_cfg = copy.deepcopy(cfg)
            job_cfg["seed"] = seed
            job_cfg["paths"]["DAG_file"] = f"./{graph_path.relative_to(root).as_posix()}"
            job_cfg["experiment_name"] = f"edge_sensitivity/{model_name}/remove_{edge_id}/seed_{seed}"
            output_cfg = config_root / f"{analysis['dataset'].lower()}_{model_name}_remove_{edge_id}_seed_{seed}.yaml"
            output_cfg.write_text(yaml.safe_dump(job_cfg, sort_keys=False), encoding="utf-8")
            generated_configs.append(output_cfg.relative_to(root).as_posix())

    plan_dir = out_dir / "retraining_plan"
    plan_dir.mkdir(parents=True, exist_ok=True)
    config_list = plan_dir / "condor_configs.txt"
    config_list.write_text("\n".join(generated_configs) + "\n", encoding="utf-8")
    run_script = root / "server_scripts/cream_experiment/analysis/run_edge_sensitivity_retrain.sh"
    sub_path = plan_dir / "edge_sensitivity_retrain.sub"
    sub_path.write_text(
        "\n".join([
            "universe = docker",
            "docker_image = pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime",
            f"initialdir = {root.as_posix()}",
            f"executable = {run_script.as_posix()}",
            'arguments = "$(config)"',
            f"output = {(root / 'logs/edge_retrain.$(ClusterId).$(ProcId).out').as_posix()}",
            f"error = {(root / 'logs/edge_retrain.$(ClusterId).$(ProcId).err').as_posix()}",
            f"log = {(root / 'logs/edge_retrain.$(ClusterId).log').as_posix()}",
            "request_GPUs = 1",
            "request_CPUs = 8",
            "request_memory = 32G",
            'requirements = UidDomain == "cs.uni-saarland.de"',
            "+WantGPUHomeMounted = true",
            f"queue config from {config_list.as_posix()}",
            "",
        ]), encoding="utf-8")
    submit = plan_dir / "submit_retraining.sh"
    submit.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {root.as_posix()}\n"
        f"chmod +x {run_script.as_posix()}\n"
        f"condor_submit {sub_path.as_posix()}\n", encoding="utf-8")
    print(f"Generated {len(generated_configs)} training configs under {config_root}")
    print(f"Submit after syncing to the server: bash {submit.relative_to(root).as_posix()}")


if __name__ == "__main__":
    main()
