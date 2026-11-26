"""Fixed-weight C--C edge ablation and sampled Shapley analysis for CREAM.

Run from the repository root with an analysis YAML. The model's backbone,
U2U, U2C, and task-head weights are frozen during this script. Edge ablation
zeros the complete U-group-to-concept block corresponding to one C--C edge.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.nn import BCEWithLogitsLoss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_u_information_sharing import (  # noqa: E402
    build_dataset,
    build_model,
    read_config,
    resolve_checkpoint,
    resolve_path,
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_model_checkpoint(spec: dict[str, Any], root: Path) -> Path:
    return resolve_checkpoint(spec, root)


def collect_u(model, loader, device: torch.device) -> dict[str, torch.Tensor]:
    """Cache learned U2U outputs and labels once; coalition evaluation reuses them."""
    us, concepts, tasks = [], [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            x, c, y = batch[:3]
            x = x.to(device)
            backbone = model.x_to_u(x)
            u = model.u_to_CY.u2u_model(backbone)
            us.append(u.detach().cpu())
            concepts.append(torch.as_tensor(c).detach().cpu().float())
            tasks.append(torch.as_tensor(y).detach().cpu())
    if not us:
        raise RuntimeError("A configured split produced no examples")
    return {"u": torch.cat(us), "c": torch.cat(concepts), "y": torch.cat(tasks)}


def find_u2c_weight(model, k: int, input_width: int) -> torch.nn.Parameter:
    wanted = (k, input_width)
    matches = []
    for module in model.u_to_CY.u2c_model.modules():
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.nn.Parameter) and tuple(weight.shape) == wanted:
            matches.append(weight)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one linear U2C weight of shape {wanted}; found {len(matches)}. "
            "This tool currently supports the configured single-layer masked U2C models."
        )
    return matches[0]


def graph_edges(config: dict[str, Any], root: Path, names: list[str]):
    dag = pd.read_csv(resolve_path(config["paths"]["DAG_file"], root), index_col=0)
    k = int(config["hyperparameters_model2"]["num_concepts"])
    if dag.shape[0] < k or dag.shape[1] < k:
        raise ValueError(f"DAG has shape {dag.shape}, smaller than {k} concepts")
    # CSV convention: target rows, source columns.
    c2c = dag.iloc[:k, :k].to_numpy(dtype=bool).T
    edges = []
    for source in range(k):
        for target in range(k):
            if c2c[source, target]:
                edges.append({
                    # Semantic ID stays identical across graph variants when
                    # both contain the same directed concept edge.
                    "edge_id": f"c{source:03d}_to_c{target:03d}",
                    "source_index": source,
                    "target_index": target,
                    "source_concept": names[source] if source < len(names) else f"concept_{source:03d}",
                    "target_concept": names[target] if target < len(names) else f"concept_{target:03d}",
                })
    return dag, edges


def concept_output(model, u: torch.Tensor, weight: torch.Tensor, enabled: set[str],
                   edges: list[dict[str, Any]], baseline_weight: torch.Tensor,
                   width: int, device: torch.device, batch_size: int = 1024):
    with torch.no_grad():
        weight.copy_(baseline_weight)
        disabled = [e for e in edges if e["edge_id"] not in enabled]
        for e in disabled:
            lo = e["source_index"] * width
            hi = lo + width
            weight[e["target_index"], lo:hi] = 0
        all_c = []
        uc_width = model.u_to_CY.num_exogenous - model.u_to_CY.num_side_channel
        for start in range(0, len(u), batch_size):
            ub = u[start:start + batch_size].to(device)
            c = model.u_to_CY.concept_activation_function(
                model.u_to_CY.u2c_model(ub[:, :uc_width])
            )
            all_c.append(c.detach().cpu())
        return torch.cat(all_c)


def make_intervention_masks(model, split_len: int, groups: list[list[int]] | None,
                            seed: int, group_mode: bool):
    masks = {}
    module = model.u_to_CY
    max_n = len(groups) if group_mode and groups else module.num_concepts
    for n in range(max_n + 1):
        # Make intervention choices identical for every evaluated coalition.
        seed_all(seed + 7919 * (n + 1) + (104729 if group_mode else 0))
        if group_mode:
            mask = module.generate_group_intervention_mask(n, split_len)
        else:
            mask = module.generate_intervention_mask(n, split_len)
        masks[n] = mask.bool()
    return masks


def task_correct(logits: torch.Tensor, y: torch.Tensor, num_classes: int) -> torch.Tensor:
    if num_classes == 1:
        pred = (logits.reshape(-1) > 0).long()
        target = y.reshape(-1).long()
    else:
        pred = logits.argmax(dim=1)
        target = y.reshape(-1).long()
    return (pred == target).float()


def metric_values(model, split_data: dict[str, torch.Tensor], c_all: torch.Tensor,
                  percentiles: tuple[torch.Tensor, torch.Tensor] | None,
                  intervention_masks: dict[tuple[str, int], torch.Tensor],
                  modes: list[str], batch_size: int, device: torch.device,
                  split: str) -> dict[str, float]:
    k = c_all.shape[1]
    threshold = 0.0 if isinstance(model.concept_loss_function, BCEWithLogitsLoss) else 0.5
    y = split_data["y"]
    c_true = split_data["c"] > 0.5
    task_correct_all, concept_correct_all = [], []
    per_concept_correct = []
    with torch.no_grad():
        for start in range(0, len(c_all), batch_size):
            end = min(start + batch_size, len(c_all))
            c = c_all[start:end].to(device)
            cb = c_true[start:end].to(device)
            yb = y[start:end].to(device)
            task_logits = model.u_to_CY.last_layer(c)
            task_correct_all.append(task_correct(task_logits, yb, model.num_classes).cpu())
            corr = ((c > threshold) == cb).float()
            concept_correct_all.append(corr.mean(dim=1).cpu())
            per_concept_correct.append(corr.mean(dim=0).cpu())
    result = {
        "task_accuracy": float(torch.cat(task_correct_all).mean()),
        "concept_accuracy": float(torch.cat(concept_correct_all).mean()),
    }
    # Weight per-concept means by examples, including a smaller final batch.
    for j in range(k):
        correct = []
        with torch.no_grad():
            for start in range(0, len(c_all), batch_size):
                end = min(start + batch_size, len(c_all))
                cb = c_true[start:end, j]
                c = c_all[start:end, j]
                correct.append(((c > threshold) == cb).float())
        result[f"concept_accuracy_{j:03d}"] = float(torch.cat(correct).mean())

    # Intervention accuracy curves use the same fixed random intervention masks
    # for all edge coalitions of a given model and split.
    low = high = None
    if percentiles is not None:
        low, high = (x.to(device) for x in percentiles)
    modes_to_run = ["individual"] + (["group"] if "group" in modes else [])
    for mode in modes_to_run:
        n_values = sorted(n for m, n in intervention_masks if m == mode)
        curve_task, curve_concept = [], []
        for n in n_values:
            correct_y, correct_c = [], []
            masks = intervention_masks[(mode, n)]
            for start in range(0, len(c_all), batch_size):
                end = min(start + batch_size, len(c_all))
                c_pred = c_all[start:end].to(device)
                c_targets = split_data["c"][start:end].to(device).float()
                yb = y[start:end].to(device)
                mask = masks[start:end].to(device)
                if low is not None and high is not None:
                    c_intervene = c_targets * high + (1 - c_targets) * low
                else:
                    c_intervene = c_targets
                c_after = c_pred.clone()
                c_after[mask] = c_intervene[mask]
                logits = model.u_to_CY.last_layer(c_after)
                correct_y.append(task_correct(logits, yb, model.num_classes).cpu())
                correct_c.append((((c_after > threshold) == (c_targets > 0.5)).float().mean(dim=1)).cpu())
            task_acc = float(torch.cat(correct_y).mean())
            concept_acc = float(torch.cat(correct_c).mean())
            curve_task.append(task_acc)
            curve_concept.append(concept_acc)
            result[f"intervention_task_accuracy_{mode}_{n:03d}"] = task_acc
            result[f"intervention_concept_accuracy_{mode}_{n:03d}"] = concept_acc
        if len(n_values) > 1:
            x = np.asarray(n_values, dtype=float) / max(n_values)
            result[f"intervention_task_auc_{mode}"] = float(np.trapz(curve_task, x))
            result[f"intervention_concept_auc_{mode}"] = float(np.trapz(curve_concept, x))
        elif n_values:
            result[f"intervention_task_auc_{mode}"] = curve_task[0]
            result[f"intervention_concept_auc_{mode}"] = curve_concept[0]
    return result


def flat_metric_rows(model_name: str, split: str, coalition_type: str,
                     coalition_id: str, values: dict[str, float],
                     edge: dict[str, Any] | None = None):
    rows = []
    for metric, value in values.items():
        row = {"model": model_name, "split": split, "coalition_type": coalition_type,
               "coalition_id": coalition_id, "metric": metric, "metric_value": value}
        if edge:
            row.update(edge)
        rows.append(row)
    return rows


def run_one_model(spec: dict[str, Any], root: Path, device: torch.device,
                  analysis: dict[str, Any], concept_names: list[str]):
    cfg_path = resolve_path(spec["config"], root)
    config = read_config(cfg_path)
    checkpoint = resolve_model_checkpoint(spec, root)
    print(f"Loading {spec['name']}: {checkpoint}")
    dataset = build_dataset(config, root)
    model = build_model(config, checkpoint, root, device)
    if config["hyperparameters_model2"].get("num_hidden_layers_in_maskedmlp", 0) != 0:
        raise ValueError("Edge analysis currently requires num_hidden_layers_in_maskedmlp: 0")
    if config["hyperparameters_model2"].get("num_side_channel", 0) != 0:
        raise ValueError("Edge analysis currently requires num_side_channel: 0")
    loaders = {"train": dataset.train_dataloader(), "val": dataset.val_dataloader(),
               "test": dataset.test_dataloader()}
    data = {split: collect_u(model, loader, device) for split, loader in loaders.items()}
    dag, edges = graph_edges(config, root, concept_names)
    k = int(config["hyperparameters_model2"]["num_concepts"])
    uc_width = int(config["hyperparameters_model2"]["num_exogenous"]
                   - config["hyperparameters_model2"].get("num_side_channel", 0))
    if uc_width % k:
        raise ValueError("U2C input width must be divisible by concept count")
    width = uc_width // k
    weight = find_u2c_weight(model, k, uc_width)
    baseline_weight = weight.detach().clone()
    if not edges:
        raise ValueError(f"No C--C edges found in DAG for model {spec['name']}")
    groups = None
    if config["paths"].get("softmax_mask"):
        groups = json.loads(resolve_path(config["paths"]["softmax_mask"], root).read_text(encoding="utf-8"))
        # The existing CREAM protocol adds grouped interventions only on datasets with this config.
    modes = ["individual"] + (["group"] if groups else [])
    split_names = list(analysis.get("evaluation_splits", ["val", "test"]))
    intervention_masks = {}
    seed = int(analysis.get("intervention_seed", 421))
    for split in split_names:
        n = len(data[split]["u"])
        for group_mode in (False, True) if groups else (False,):
            mode = "group" if group_mode else "individual"
            generated = make_intervention_masks(model, n, groups, seed, group_mode)
            for count, mask in generated.items():
                intervention_masks[(split, mode, count)] = mask

    # Fix the concept-scale percentile mapping from the trained model's train split,
    # matching simple_main.interventions and keeping it constant across ablations.
    train_c = concept_output(model, data["train"]["u"], weight, set(e["edge_id"] for e in edges),
                             edges, baseline_weight, width, device)
    representation = config["hyperparameters_model2"].get("concept_representation", "logits")
    percentiles = None
    if representation not in ("hard", "group_hard"):
        percentiles = (torch.quantile(train_c, 0.05, dim=0), torch.quantile(train_c, 0.95, dim=0))

    metric_cache: dict[tuple[str, ...], dict[str, dict[str, float]]] = {}
    coalition_rows = []

    def evaluate(enabled: set[str], coalition_type: str, coalition_id: str,
                 edge: dict[str, Any] | None = None):
        key = tuple(sorted(enabled))
        if key not in metric_cache:
            c_all = concept_output(model, data["test"]["u"], weight, enabled, edges,
                                   baseline_weight, width, device)
            metrics = {}
            # Shapley is configured on validation by default; don't pay to
            # recompute its many coalitions on test as well.
            splits_for_key = split_names if coalition_type != "shapley_coalition" else [shapley_split]
            for split in splits_for_key:
                if split == "test":
                    c_split = c_all
                else:
                    c_split = concept_output(model, data[split]["u"], weight, enabled, edges,
                                             baseline_weight, width, device)
                masks = {(mode, n): intervention_masks[(split, mode, n)]
                         for mode, n in [(m, n) for (s, m, n) in intervention_masks if s == split]}
                metrics[split] = metric_values(model, data[split], c_split, percentiles, masks,
                                               modes, int(analysis.get("metric_batch_size", 1024)),
                                               device, split)
            metric_cache[key] = metrics
        for split, metrics in metric_cache[key].items():
            coalition_rows.extend(flat_metric_rows(spec["name"], split, coalition_type,
                                                   coalition_id, metrics, edge))
        return metric_cache[key]

    all_edges = {e["edge_id"] for e in edges}
    evaluate(all_edges, "baseline", "all_edges_enabled")
    # Single-edge fixed-weight ablations on validation and held-out test splits.
    for edge in edges:
        evaluate(all_edges - {edge["edge_id"]}, "single_edge_removed", edge["edge_id"], edge)

    shapley_split = str(analysis.get("shapley_split", "val"))
    repetitions = int(analysis.get("shapley_permutations", 100))
    rng = np.random.default_rng(int(analysis.get("shapley_seed", 2026)))
    marginal_values: dict[str, dict[str, list[float]]] = {e["edge_id"]: {} for e in edges}
    permutation_rows = []
    for rep in range(repetitions):
        order = list(rng.permutation(len(edges)))
        enabled: set[str] = set()
        prev = evaluate(enabled, "shapley_coalition", f"p{rep:04d}_step000")[shapley_split]
        for step, idx in enumerate(order, start=1):
            edge = edges[int(idx)]
            enabled.add(edge["edge_id"])
            now = evaluate(enabled, "shapley_coalition", f"p{rep:04d}_step{step:03d}")[shapley_split]
            for metric in prev:
                marginal_values[edge["edge_id"]].setdefault(metric, []).append(now[metric] - prev[metric])
            permutation_rows.append({"model": spec["name"], "permutation": rep,
                                     "step": step, "edge_id": edge["edge_id"],
                                     "source_index": edge["source_index"],
                                     "target_index": edge["target_index"]})
            prev = now

    shapley_rows = []
    for edge in edges:
        for metric, samples in marginal_values[edge["edge_id"]].items():
            vals = np.asarray(samples, dtype=float)
            shapley_rows.append({
                "model": spec["name"], "split": shapley_split, "edge_id": edge["edge_id"],
                "source_index": edge["source_index"], "target_index": edge["target_index"],
                "source_concept": edge["source_concept"], "target_concept": edge["target_concept"],
                "metric": metric, "shapley_value": float(np.nanmean(vals)),
                "permutation_std": float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else float("nan"),
                "permutation_se": float(np.nanstd(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else float("nan"),
                "n_permutations": int(np.isfinite(vals).sum()),
            })
    return edges, coalition_rows, shapley_rows, permutation_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    analysis = read_config(args.analysis_config.resolve())
    device = torch.device(analysis.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    output = resolve_path(analysis["output_dir"], root)
    output.mkdir(parents=True, exist_ok=True)

    first_spec = analysis["models"][0]
    first_config = read_config(resolve_path(first_spec["config"], root))
    dataset = build_dataset(first_config, root)
    concepts = list(getattr(dataset, "concept_names", []) or [])
    if not concepts:
        dag = pd.read_csv(resolve_path(first_config["paths"]["DAG_file"], root), index_col=0)
        concepts = list(dag.index[:first_config["hyperparameters_model2"]["num_concepts"]].astype(str))

    all_edges, all_coalition_rows, all_shapley_rows, all_permutation_rows = [], [], [], []
    for spec in analysis["models"]:
        edges, coalition_rows, shapley_rows, permutation_rows = run_one_model(
            spec, root, device, analysis, concepts
        )
        all_edges.extend({"model": spec["name"], **e} for e in edges)
        all_coalition_rows.extend(coalition_rows)
        all_shapley_rows.extend(shapley_rows)
        all_permutation_rows.extend(permutation_rows)
    pd.DataFrame(all_edges).to_csv(output / "edge_list.csv", index=False)
    pd.DataFrame(all_coalition_rows).to_csv(output / "coalition_metrics.csv", index=False)
    pd.DataFrame(all_shapley_rows).to_csv(output / "shapley_values.csv", index=False)
    pd.DataFrame(all_permutation_rows).to_csv(output / "shapley_permutation_paths.csv", index=False)
    metadata = {"dataset": analysis["dataset"], "models": [m["name"] for m in analysis["models"]],
                "evaluation_splits": analysis.get("evaluation_splits", ["val", "test"]),
                "shapley_split": analysis.get("shapley_split", "val"),
                "shapley_permutations": int(analysis.get("shapley_permutations", 100)),
                "fixed_weights": True,
                "edge_removal": "zero the complete source-U-group to target-concept U2C weight block",
                "intervention_protocol": "same sampled intervention masks and baseline-train concept percentiles across coalitions"}
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved edge sensitivity and Shapley tables under {output}")


if __name__ == "__main__":
    main()
