"""Compare U2U representations from two trained CREAM topology variants.

Implements representation/probe analysis, matched-pair group patching, and cross-model transplants:
  * cache U2U vectors and concept logits/probabilities on train/val/test splits
  * per-group/per-concept logistic-probe AUROC matrices
  * graph-allowed joint probes
  * conditional predictive log-loss improvement I(C_i; U_k | C_k) proxy
  * matched within-model U-group patching and cross-model same-image transplants

Example (run from repository root):
python scripts/analyze_u_information_sharing.py \
  --dataset CelebA \
  --config-a all_configs/sanity_checks/celeba_concept_concept_fixcount_topology_no_side_channel/CREAM_celeba_concept_concept_fixcount_topology_no_side_channel_replace_000.yaml \
  --checkpoint-a experiments/CelebA/train_cbm/Standard_CelebA/.../checkpoints/best.ckpt \
  --config-b all_configs/sanity_checks/celeba_concept_concept_fixcount_topology_no_side_channel/CREAM_celeba_concept_concept_fixcount_topology_no_side_channel_replace_100.yaml \
  --checkpoint-b experiments/CelebA/train_cbm/Standard_CelebA/.../checkpoints/best.ckpt \
  --output-dir experiments/u_information_sharing/celeba_replace000_vs_replace100

The run configs determine the official dataset splits and optional frozen
backbone feature cache. Probe training uses the CREAM train split, model
selection uses validation, and final probe metrics use the CREAM test split.
"""

from __future__ import annotations

import argparse
import glob
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from src.fixed_feature_cache import CachedFeatureBackbone, CachedFeatureDataModule
from src.models import Template_CBM_MultiClass, UtoY_model
from src.utils import get_component_with_dicts


def read_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_checkpoint(spec: dict[str, Any], root: Path) -> Path:
    if spec.get("checkpoint"):
        checkpoint = resolve_path(spec["checkpoint"], root)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Configured checkpoint does not exist: {checkpoint}")
        return checkpoint
    if not spec.get("checkpoint_glob"):
        raise ValueError("Each model entry needs checkpoint or checkpoint_glob")
    pattern = resolve_path(spec["checkpoint_glob"], root)
    candidates = [Path(p) for p in glob.glob(str(pattern)) if Path(p).is_file()]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint matches configured pattern: {pattern}")
    # Checkpoint files are selected by latest modification time across Lightning versions.
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_path(path: str | Path, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def build_model(config: dict[str, Any], checkpoint_path: Path, root: Path, device: torch.device):
    h = config["hyperparameters_model2"]
    paths = config["paths"]
    model_class = get_component_with_dicts("model", config["model_name"])
    if config.get("feature_cache_dir"):
        backbone = CachedFeatureBackbone()
    else:
        backbone_ckpt = resolve_path(paths["input_model_path"], root)
        loaded_backbone = model_class.load_from_checkpoint(
            checkpoint_path=backbone_ckpt,
            dataset=config["dataset_name"],
            frozen=True,
        )
        backbone = loaded_backbone.concept_extractor

    graph = torch.as_tensor(
        pd.read_csv(resolve_path(paths["DAG_file"], root), index_col=0).values,
        dtype=torch.bool,
    )
    groups = None
    if paths.get("softmax_mask"):
        groups = json.loads(resolve_path(paths["softmax_mask"], root).read_text(encoding="utf-8"))
    u2c = UtoY_model(**h, causal_graph=graph, mutually_exclusive_concepts=groups)
    model = Template_CBM_MultiClass(
        backbone,
        u2c,
        num_exogenous=h["num_exogenous"],
        num_classes=h["num_classes"],
        num_concepts=h["num_concepts"],
        num_side_channel=h.get("num_side_channel", 0),
        frozen_model1=True,
        concept_representation=h.get("concept_representation", "logits"),
        learning_rate=config.get("hyperparameters", {}).get("learning_rate", 1e-4),
        lambda_weight=config.get("hyperparameters", {}).get("lambda_weight", 1.0),
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing_core = [k for k in missing if k.startswith(("u_to_CY.", "x_to_u."))]
    if missing_core:
        raise RuntimeError(f"Checkpoint {checkpoint_path} is missing model weights: {missing_core[:12]}")
    if unexpected:
        print(f"Note: ignoring {len(unexpected)} unrelated checkpoint keys for {checkpoint_path.name}")
    model.to(device).eval()
    return model


def build_dataset(config: dict[str, Any], root: Path):
    dataset_class = get_component_with_dicts("dataset", config["dataset_name"])
    seed = int(config.get("seed", 42))
    params = dict(config.get("dataset_params", {}))
    if "FMNIST" in config["dataset_name"]:
        if config["dataset_name"] == "Complete_Concept_FMNIST":
            dataset = dataset_class(**params, seed=seed, full_concepts=True)
        else:
            dataset = dataset_class(**params, seed=seed)
    else:
        dataset = dataset_class(**params)
    cache_dir = config.get("feature_cache_dir")
    if cache_dir:
        cache_dir = resolve_path(cache_dir, root)
        if not (cache_dir / "train.pt").exists() or not (cache_dir / "test.pt").exists():
            raise FileNotFoundError(f"Feature cache missing train.pt/test.pt under {cache_dir}")
        dataset = CachedFeatureDataModule(dataset, cache_dir)
    dataset.setup("fit")
    dataset.setup("test")
    return dataset


@torch.inference_mode()
def extract_split(model, loader, split: str, device: torch.device):
    all_u, all_logits, all_scores, all_y, all_c, all_task_targets = [], [], [], [], [], []
    for batch in loader:
        x, concepts, labels = batch[:3]
        x = x.to(device)
        # Match CREAM's forward path: frozen X2U/backbone -> learned U2U -> U2C.
        backbone_features = model.x_to_u(x)
        u = model.u_to_CY.u2u_model(backbone_features)
        n_concept_u = model.u_to_CY.num_exogenous - model.u_to_CY.num_side_channel
        uc = u[:, :n_concept_u]
        logits = model.u_to_CY.u2c_model(uc)
        scores = model.u_to_CY.concept_activation_function(logits)
        y_logits, _ = model.u_to_CY(backbone_features)
        all_u.append(u.cpu())
        all_logits.append(logits.cpu())
        all_scores.append(scores.cpu())
        all_y.append(y_logits.cpu())
        all_c.append(torch.as_tensor(concepts).cpu())
        all_task_targets.append(torch.as_tensor(labels).cpu())
    if not all_u:
        raise RuntimeError(f"No examples found in {split} loader")
    return {
        "u": torch.cat(all_u),
        "concept_logits": torch.cat(all_logits),
        "concept_scores": torch.cat(all_scores),
        "task_logits": torch.cat(all_y),
        "concept_targets": torch.cat(all_c),
        "task_targets": torch.cat(all_task_targets),
        # Stable row IDs are split-local positions in the exact dataloader order.
        "image_ids": torch.arange(sum(len(x) for x in all_c), dtype=torch.int64),
    }


def binary_target(y: np.ndarray) -> np.ndarray:
    return (np.asarray(y).reshape(-1) > 0.5).astype(np.int64)


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    y = binary_target(y)
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def bootstrap_auc_interval(y: np.ndarray, p: np.ndarray, repetitions: int, seed: int) -> tuple[float, float]:
    y = binary_target(y)
    if repetitions <= 0 or np.unique(y).size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(repetitions):
        ids = rng.integers(0, len(y), size=len(y))
        score = safe_auc(y[ids], p[ids])
        if np.isfinite(score):
            scores.append(score)
    if not scores:
        return float("nan"), float("nan")
    return tuple(float(x) for x in np.quantile(scores, [0.025, 0.975]))


def fit_probe(x_train, y_train, x_val, y_val, c_values):
    y_train = binary_target(y_train)
    y_val = binary_target(y_val)
    if x_train.ndim != 2 or x_train.shape[1] == 0:
        return None, float("nan"), float("nan")
    if np.unique(y_train).size < 2:
        return None, float("nan"), float("nan")
    best_model, best_score, best_c = None, -math.inf, None
    for c in c_values:
        probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=float(c), max_iter=2000, class_weight="balanced", solver="lbfgs"),
        )
        probe.fit(x_train, y_train)
        val_probability = probe.predict_proba(x_val)[:, 1]
        score = safe_auc(y_val, val_probability)
        if not np.isfinite(score):
            score = -float(log_loss(y_val, val_probability, labels=[0, 1]))
        if score > best_score:
            best_model, best_score, best_c = probe, score, float(c)
    return best_model, float(best_c), float(best_score)


def predict_probability(probe, x):
    return probe.predict_proba(x)[:, 1]


def sequential_loader(loader):
    """Use stable dataset order so both models see the same examples in each split."""
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size or 256,
        shuffle=False,
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
        drop_last=False,
    )


def _nearest_candidates(pool: np.ndarray, used: set[int], limit: int, rng: np.random.Generator) -> np.ndarray:
    candidates = np.asarray([int(i) for i in pool if int(i) not in used], dtype=np.int64)
    if len(candidates) > limit:
        candidates = rng.choice(candidates, size=limit, replace=False)
    else:
        rng.shuffle(candidates)
    return candidates


def make_matched_pairs(labels: np.ndarray, per_concept: int, seed: int):
    """Make shared positive-to-negative and same-label controls, matching other labels."""
    y = (np.asarray(labels) > 0.5).astype(np.int8)
    n, k = y.shape
    output = []
    for concept in range(k):
        rng = np.random.default_rng(seed + concept)
        positive = np.flatnonzero(y[:, concept] == 1)
        negative = np.flatnonzero(y[:, concept] == 0)
        rng.shuffle(positive)
        used_pos: set[int] = set()
        used_neg: set[int] = set()
        target_pairs = []
        for clean in positive:
            if len(target_pairs) >= per_concept:
                break
            candidates = _nearest_candidates(negative, used_neg, 4096, rng)
            if not len(candidates):
                break
            dist = np.count_nonzero(y[candidates][:, np.arange(k) != concept] !=
                                    y[int(clean), np.arange(k) != concept], axis=1)
            source = int(candidates[int(np.argmin(dist))])
            used_pos.add(int(clean)); used_neg.add(source)
            target_pairs.append((int(clean), source, int(dist.min())))

        control_pool = np.flatnonzero(y[:, concept] == 1)
        used_control: set[int] = set()
        controls = []
        other_cols = np.arange(k) != concept
        for desired_dist in [p[2] for p in target_pairs]:
            anchors = _nearest_candidates(control_pool, used_control, 4096, rng)
            if len(anchors) < 2:
                break
            anchor = int(anchors[0])
            candidates = anchors[1:]
            distances = np.count_nonzero(y[candidates][:, other_cols] != y[anchor, other_cols], axis=1)
            # Prefer a control with the same amount of label variation on other concepts.
            source = int(candidates[int(np.argmin(np.abs(distances - desired_dist)))])
            used_control.update((anchor, source))
            controls.append((anchor, source, int(np.count_nonzero(y[anchor, other_cols] != y[source, other_cols]))))

        for pair_id, (target_pair, control_pair) in enumerate(zip(target_pairs, controls)):
            for kind, pair in (("target_change", target_pair), ("control_same_target", control_pair)):
                clean, source, other_distance = pair
                output.append({
                    "changed_concept_index": concept,
                    "changed_concept": f"concept_{concept:03d}",
                    "pair_id": pair_id,
                    "pair_type": kind,
                    "clean_image_id": clean,
                    "source_image_id": source,
                    "clean_concept_value": int(y[clean, concept]),
                    "source_concept_value": int(y[source, concept]),
                    "other_concepts_hamming_distance": other_distance,
                })
    return output


def u2c_allowed_source_to_target(dag: pd.DataFrame, num_concepts: int) -> np.ndarray:
    """Convert CREAM DAG CSV convention (target rows, source columns) to [source,target]."""
    return dag.iloc[:num_concepts, :num_concepts].to_numpy(dtype=bool).T


def apply_u2c(model, u_concept: torch.Tensor, device: torch.device, chunk_size: int = 512):
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(u_concept), chunk_size):
            batch = u_concept[start:start + chunk_size].to(device)
            chunks.append(model.u_to_CY.u2c_model(batch).detach().cpu())
    return torch.cat(chunks, dim=0)


@torch.inference_mode()
def run_within_model_pair_patching(models, names, data, dags, concept_names, width, pairs, device):
    """Patch one source group's same-model activation from a matched source image."""
    k = len(concept_names)
    kinds = ("target_change", "control_same_target")
    max_pairs = max((int(r["pair_id"]) + 1 for r in pairs), default=0)
    shape = (len(names), k, k, k, len(kinds), max_pairs)
    signed = torch.full(shape, float("nan"), dtype=torch.float32)
    absolute = torch.full_like(signed, float("nan"))
    probability_delta = torch.full_like(signed, float("nan"))
    pair_rows = []
    pairs_by_concept_kind = {}
    for record in pairs:
        record["changed_concept"] = concept_names[record["changed_concept_index"]]
        pairs_by_concept_kind.setdefault((record["changed_concept_index"], record["pair_type"]), []).append(record)

    for m, (name, model, dag) in enumerate(zip(names, models, dags)):
        u_all = data[name]["test"]["u"]
        n_uc = model.u_to_CY.num_exogenous - model.u_to_CY.num_side_channel
        groups_u = u_all[:, :n_uc].reshape(len(u_all), k, width)
        base_logits = data[name]["test"]["concept_logits"]
        base_scores = data[name]["test"]["concept_scores"]
        adjacency = u2c_allowed_source_to_target(dag, k)
        for changed in range(k):
            for kind_index, kind in enumerate(kinds):
                recs = pairs_by_concept_kind.get((changed, kind), [])
                if not recs:
                    continue
                clean_ids = torch.tensor([r["clean_image_id"] for r in recs], dtype=torch.long)
                source_ids = torch.tensor([r["source_image_id"] for r in recs], dtype=torch.long)
                for source_group in range(k):
                    # Patch all pairs for this group together, then record permitted outputs only.
                    patched = groups_u[clean_ids].reshape(len(recs), k * width).clone()
                    lo, hi = source_group * width, (source_group + 1) * width
                    patched[:, lo:hi] = groups_u[source_ids, source_group, :]
                    patched_logits = apply_u2c(model, patched, device)
                    patched_scores = torch.cat([
                        model.u_to_CY.concept_activation_function(chunk.to(device)).detach().cpu()
                        for chunk in patched.split(512)
                    ], dim=0)
                    for target in range(k):
                        if not adjacency[source_group, target]:
                            continue
                        delta = patched_logits[:, target] - base_logits[clean_ids, target]
                        pdelta = patched_scores[:, target] - base_scores[clean_ids, target]
                        for r_i, rec in enumerate(recs):
                            pid = int(rec["pair_id"])
                            signed[m, changed, source_group, target, kind_index, pid] = delta[r_i]
                            absolute[m, changed, source_group, target, kind_index, pid] = delta[r_i].abs()
                            probability_delta[m, changed, source_group, target, kind_index, pid] = pdelta[r_i]
                            pair_rows.append({
                                "model": name, **rec, "source_group_index": source_group,
                                "source_group_concept": concept_names[source_group],
                                "predicted_concept_index": target, "predicted_concept": concept_names[target],
                                "baseline_logit": float(base_logits[clean_ids[r_i], target]),
                                "patched_logit": float(patched_logits[r_i, target]),
                                "signed_logit_change": float(delta[r_i]),
                                "absolute_logit_change": float(delta[r_i].abs()),
                                "baseline_score": float(base_scores[clean_ids[r_i], target]),
                                "patched_score": float(patched_scores[r_i, target]),
                                "signed_score_change": float(pdelta[r_i]),
                            })

    return signed, absolute, probability_delta, pd.DataFrame(pair_rows)


@torch.inference_mode()
def run_cross_model_patching(models, names, data, dags, concept_names, width, device, output_dir):
    """Transplant donor U groups into a receiver's U2C for the same test images."""
    from sklearn.pipeline import make_pipeline

    k = len(concept_names)
    results = {}
    alignment_parameters = {}
    summary_rows = []
    for receiver_i, receiver in enumerate(names):
        donor = names[1 - receiver_i]
        receiver_model = models[receiver_i]
        receiver_u = data[receiver]["test"]["u"][:, :k * width].reshape(-1, k, width).numpy()
        donor_u = data[donor]["test"]["u"][:, :k * width].reshape(-1, k, width).numpy()
        receiver_val = data[receiver]["val"]["u"][:, :k * width].reshape(-1, k, width).numpy()
        donor_val = data[donor]["val"]["u"][:, :k * width].reshape(-1, k, width).numpy()
        receiver_logits = data[receiver]["test"]["concept_logits"].numpy()
        receiver_scores = data[receiver]["test"]["concept_scores"].numpy()
        receiver_targets = data[receiver]["test"]["concept_targets"].numpy() > 0.5
        adjacency = u2c_allowed_source_to_target(dags[receiver_i], k)
        for mode in ("raw", "aligned"):
            delta = torch.full((len(receiver_u), k, k), float("nan"), dtype=torch.float32)
            score_delta = torch.full_like(delta, float("nan"))
            for group in range(k):
                if mode == "raw":
                    transplanted_group = donor_u[:, group, :]
                    alignment_r2 = float("nan")
                else:
                    aligner = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                    aligner.fit(donor_val[:, group, :], receiver_val[:, group, :])
                    transplanted_group = aligner.predict(donor_u[:, group, :])
                    alignment_r2 = float(aligner.score(donor_val[:, group, :], receiver_val[:, group, :]))
                    scaler = aligner.named_steps["standardscaler"]
                    ridge = aligner.named_steps["ridge"]
                    alignment_parameters[f"{receiver}_receives_{donor}_group_{group}"] = {
                        "source_mean": torch.as_tensor(scaler.mean_),
                        "source_scale": torch.as_tensor(scaler.scale_),
                        "ridge_coef": torch.as_tensor(ridge.coef_),
                        "ridge_intercept": torch.as_tensor(ridge.intercept_),
                        "validation_r2": alignment_r2,
                        "alpha": 1.0,
                    }
                for start in range(0, len(receiver_u), 512):
                    end = min(start + 512, len(receiver_u))
                    patched = receiver_u[start:end].reshape(end - start, k * width).copy()
                    patched[:, group * width:(group + 1) * width] = transplanted_group[start:end]
                    patched_logits = apply_u2c(receiver_model, torch.as_tensor(patched), device).numpy()
                    patched_scores = receiver_model.u_to_CY.concept_activation_function(
                        torch.as_tensor(patched_logits, device=device)
                    ).detach().cpu().numpy()
                    for target in np.flatnonzero(adjacency[group]):
                        delta[start:end, group, target] = torch.from_numpy(
                            patched_logits[:, target] - receiver_logits[start:end, target]
                        )
                        score_delta[start:end, group, target] = torch.from_numpy(
                            patched_scores[:, target] - receiver_scores[start:end, target]
                        )
                for target in range(k):
                    if not adjacency[group, target]:
                        continue
                    vals = delta[:, group, target].numpy()
                    score_vals = score_delta[:, group, target].numpy()
                    present = receiver_targets[:, target]
                    absent = ~present
                    summary_rows.append({"receiver": receiver, "donor": donor, "mode": mode,
                                         "source_group_index": group,
                                         "source_group_concept": concept_names[group],
                                         "target_concept_index": target,
                                         "target_concept": concept_names[target],
                                         "n_test_images": len(vals),
                                         "mean_signed_logit_change": float(np.nanmean(vals)),
                                         "mean_absolute_logit_change": float(np.nanmean(np.abs(vals))),
                                         "mean_signed_score_change": float(np.nanmean(score_vals)),
                                         "mean_absolute_score_change": float(np.nanmean(np.abs(score_vals))),
                                         "n_target_present": int(present.sum()),
                                         "mean_abs_logit_change_target_present": float(np.nanmean(np.abs(vals[present]))) if present.any() else float("nan"),
                                         "mean_abs_logit_change_target_absent": float(np.nanmean(np.abs(vals[absent]))) if absent.any() else float("nan"),
                                         "validation_alignment_r2": alignment_r2})
            results[f"{receiver}_receives_{donor}_{mode}"] = {
                "signed_logit_change": delta,
                "signed_score_change": score_delta,
                "axis_order": ["test_image", "source_group", "target_concept"],
                "receiver": receiver,
                "donor": donor,
                "mode": mode,
            }
    torch.save(results, output_dir / "cross_model_activation_patch_effects.pt")
    torch.save(alignment_parameters, output_dir / "cross_model_alignment_parameters.pt")
    pd.DataFrame(summary_rows).to_csv(output_dir / "cross_model_activation_patch_summary.csv", index=False)


def save_within_patch_outputs(signed, absolute, score_delta, pair_df, names, concepts, output_dir):
    torch.save({"signed_logit_change": signed, "absolute_logit_change": absolute,
                "signed_probability_change": score_delta,
                "axis_order": ["model", "changed_concept", "source_group", "predicted_concept", "pair_type", "pair_id"],
                "models": names, "concepts": concepts,
                "pair_types": ["target_change", "control_same_target"]},
               output_dir / "within_model_activation_patch_effects.pt")
    pair_df.to_csv(output_dir / "within_model_activation_patch_pairs.csv", index=False)
    summary = pair_df.groupby(["model", "changed_concept", "source_group_concept", "predicted_concept", "pair_type"], as_index=False).agg(
        n_pairs=("absolute_logit_change", "count"),
        mean_signed_logit_change=("signed_logit_change", "mean"),
        mean_absolute_logit_change=("absolute_logit_change", "mean"),
        mean_signed_score_change=("signed_score_change", "mean"),
    )
    summary.to_csv(output_dir / "within_model_activation_patch_summary.csv", index=False)
    means = summary.pivot_table(index=["model", "changed_concept", "source_group_concept", "predicted_concept"],
                                columns="pair_type", values="mean_absolute_logit_change").reset_index()
    if "target_change" in means and "control_same_target" in means:
        means["specificity_Q_target_minus_control"] = means["target_change"] - means["control_same_target"]
    means.to_csv(output_dir / "within_model_activation_patch_specificity.csv", index=False)


def probe_record(model_name, target_name, feature_name, x_train, y_train, x_val, y_val, x_test, y_test, c_values, bootstrap_repetitions):
    probe, best_c, val_score = fit_probe(x_train, y_train, x_val, y_val, c_values)
    if probe is None:
        return {"model": model_name, "target_concept": target_name, "feature_set": feature_name,
                "best_C": np.nan, "val_auc": np.nan, "test_auc": np.nan,
                "test_auc_ci_low": np.nan, "test_auc_ci_high": np.nan,
                "test_nll": np.nan, "probe": None}
    p_test = predict_probability(probe, x_test)
    y_test_bin = binary_target(y_test)
    ci_low, ci_high = bootstrap_auc_interval(y_test_bin, p_test, bootstrap_repetitions,
                                              seed=1000 + len(feature_name) + len(target_name))
    return {"model": model_name, "target_concept": target_name, "feature_set": feature_name,
            "best_C": best_c, "val_auc": val_score, "test_auc": safe_auc(y_test_bin, p_test),
            "test_auc_ci_low": ci_low, "test_auc_ci_high": ci_high,
            "test_nll": float(log_loss(y_test_bin, p_test, labels=[0, 1])), "probe": probe}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-config", type=Path,
                        help="YAML specifying dataset, model configs/checkpoint globs, and output directory")
    parser.add_argument("--dataset", help="Dataset label, e.g. CelebA; normally set in --analysis-config")
    parser.add_argument("--config-a", type=Path)
    parser.add_argument("--checkpoint-a", type=Path)
    parser.add_argument("--name-a")
    parser.add_argument("--config-b", type=Path)
    parser.add_argument("--checkpoint-b", type=Path)
    parser.add_argument("--name-b")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--probe-c-values", type=float, nargs="+")
    parser.add_argument("--bootstrap-repetitions", type=int, default=None,
                        help="Bootstrap samples for 95%% test-AUROC intervals; set 0 to skip")
    parser.add_argument("--pairs-per-concept", type=int, default=None,
                        help="Number of target-changing/control image-pair matches per concept")
    args = parser.parse_args()

    if args.analysis_config:
        analysis_path = args.analysis_config.resolve()
        analysis = read_config(analysis_path)
        root = args.root.resolve() if args.root else Path.cwd().resolve()
        models_config = analysis.get("models", {})
        if isinstance(models_config, list):
            models_config = {item["name"]: item for item in models_config}
        if len(models_config) != 2:
            raise ValueError("Analysis config must define exactly two models under 'models'")
        model_specs = list(models_config.values())
        args.dataset = args.dataset or analysis["dataset"]
        args.name_a = args.name_a or model_specs[0]["name"]
        args.name_b = args.name_b or model_specs[1]["name"]
        args.config_a = args.config_a or Path(model_specs[0]["config"])
        args.config_b = args.config_b or Path(model_specs[1]["config"])
        args.checkpoint_a = args.checkpoint_a or resolve_checkpoint(model_specs[0], root)
        args.checkpoint_b = args.checkpoint_b or resolve_checkpoint(model_specs[1], root)
        args.output_dir = args.output_dir or Path(analysis["output_dir"])
        args.probe_c_values = args.probe_c_values or analysis.get("probe_c_values", [0.01, 0.1, 1.0, 10.0])
        args.bootstrap_repetitions = (args.bootstrap_repetitions if args.bootstrap_repetitions is not None
                                      else analysis.get("bootstrap_repetitions", 300))
        args.pairs_per_concept = (args.pairs_per_concept if args.pairs_per_concept is not None
                                  else analysis.get("pairs_per_concept", 100))
        args.device = args.device or analysis.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Analysis config: {analysis_path}")
        print(f"Model A checkpoint: {args.checkpoint_a}")
        print(f"Model B checkpoint: {args.checkpoint_b}")
    else:
        required = ("dataset", "config_a", "checkpoint_a", "config_b", "checkpoint_b", "output_dir")
        missing = [key for key in required if getattr(args, key) is None]
        if missing:
            parser.error(f"Provide --analysis-config or all direct arguments; missing: {', '.join(missing)}")
        root = args.root.resolve() if args.root else Path.cwd().resolve()
        args.name_a = args.name_a or "replace_000"
        args.name_b = args.name_b or "replace_100"
        args.probe_c_values = args.probe_c_values or [0.01, 0.1, 1.0, 10.0]
        args.bootstrap_repetitions = args.bootstrap_repetitions if args.bootstrap_repetitions is not None else 300
        args.pairs_per_concept = args.pairs_per_concept if args.pairs_per_concept is not None else 100
        args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    outdir = resolve_path(args.output_dir, root)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    cfgs = [read_config(resolve_path(args.config_a, root)), read_config(resolve_path(args.config_b, root))]
    names = [args.name_a, args.name_b]
    checkpoints = [resolve_path(args.checkpoint_a, root), resolve_path(args.checkpoint_b, root)]
    for cfg, ckpt in zip(cfgs, checkpoints):
        if cfg["dataset_name"] != cfgs[0]["dataset_name"]:
            raise ValueError("Both model configs must use the same dataset")
        if not ckpt.is_file():
            raise FileNotFoundError(f"Trained model checkpoint not found: {ckpt}")
    if args.dataset.lower() not in cfgs[0]["dataset_name"].lower():
        raise ValueError(f"--dataset {args.dataset!r} does not match config dataset {cfgs[0]['dataset_name']!r}")

    # Both configs must describe the same dataset split and sample preprocessing.
    if cfgs[0].get("dataset_params", {}) != cfgs[1].get("dataset_params", {}):
        raise ValueError("Model configs have different dataset_params; use matching evaluation splits")
    cache_a, cache_b = cfgs[0].get("feature_cache_dir"), cfgs[1].get("feature_cache_dir")
    if bool(cache_a) != bool(cache_b):
        raise ValueError("One model config uses cached backbone features and the other does not")
    if cache_a and Path(cache_a) != Path(cache_b):
        raise ValueError("Both fixed-backbone models must use the same feature cache for paired comparison")
    dataset = build_dataset(cfgs[0], root)
    concept_names = list(getattr(dataset, "concept_names", []) or [])
    models = [build_model(cfg, ckpt, root, device) for cfg, ckpt in zip(cfgs, checkpoints)]
    data = {name: {} for name in names}
    loaders = (("train", sequential_loader(dataset.train_dataloader())),
               ("val", sequential_loader(dataset.val_dataloader())),
               ("test", sequential_loader(dataset.test_dataloader())))
    for split, loader in loaders:
        for name, model in zip(names, models):
            data[name][split] = extract_split(model, loader, split, device)
        # Check row alignment and target labels across both configs/models.
        a, b = (data[name][split] for name in names)
        if not torch.equal(a["concept_targets"], b["concept_targets"]):
            raise ValueError(f"Concept labels/order differ between models on {split} split")

    k = data[names[0]]["test"]["concept_targets"].shape[1]
    if not concept_names:
        concept_names = [f"concept_{i:03d}" for i in range(k)]
    if len(concept_names) != k:
        raise ValueError(f"Found {len(concept_names)} concept names for target width {k}")
    widths = []
    dags = []
    group_features: dict[str, dict[str, dict[int, np.ndarray]]] = {name: {} for name in names}
    for name, cfg in zip(names, cfgs):
        h = cfg["hyperparameters_model2"]
        n_uc = h["num_exogenous"] - h.get("num_side_channel", 0)
        if n_uc % h["num_concepts"]:
            raise ValueError(f"{name}: Uc width {n_uc} is not divisible by concept count {h['num_concepts']}")
        widths.append(n_uc // h["num_concepts"])
        dags.append(pd.read_csv(resolve_path(cfg["paths"]["DAG_file"], root), index_col=0).astype(bool))
    if widths[0] != widths[1] or k != cfgs[0]["hyperparameters_model2"]["num_concepts"] or k != cfgs[1]["hyperparameters_model2"]["num_concepts"]:
        raise ValueError("The models need matching concept counts and U group widths for a direct comparison")
    width = widths[0]

    mask_rows = []
    topology_masks = {}
    for name, dag in zip(names, dags):
        adjacency = u2c_allowed_source_to_target(dag, k)
        topology_masks[name] = {
            "concept_adjacency_source_to_target": torch.as_tensor(adjacency),
            "concept_names": concept_names,
        }
        for source in range(k):
            for target in range(k):
                mask_rows.append({"model": name, "source_group_index": source,
                                  "source_concept": concept_names[source],
                                  "target_concept_index": target,
                                  "target_concept": concept_names[target],
                                  "allowed": bool(adjacency[source, target])})
    torch.save(topology_masks, outdir / "topology_masks.pt")
    pd.DataFrame(mask_rows).to_csv(outdir / "topology_masks_long.csv", index=False)

    # Save reusable tensor caches and long-form CSVs for every official split.
    score_rows = []
    for name in names:
        for split, values in data[name].items():
            u = values["u"]
            uc = u[:, :k * width]
            values["u_concept_groups"] = uc.reshape(len(uc), k, width)
            torch.save(values, outdir / f"{name}_{split}_activations.pt")
            np.savez_compressed(
                outdir / f"{name}_{split}_activations.npz",
                image_ids=values["image_ids"].numpy(),
                u=values["u"].numpy(),
                u_concept_groups=values["u_concept_groups"].numpy(),
                concept_logits=values["concept_logits"].numpy(),
                concept_scores=values["concept_scores"].numpy(),
                concept_targets=values["concept_targets"].numpy(),
                task_logits=values["task_logits"].numpy(),
                task_targets=values["task_targets"].numpy(),
            )
            if split == "test":
                for i in range(len(values["concept_targets"])):
                    for j, cname in enumerate(concept_names):
                        score_rows.append({"model": name, "image_id": i, "concept_index": j,
                                           "concept_name": cname,
                                           "concept_logit": values["concept_logits"][i, j].item(),
                                           "concept_score": values["concept_scores"][i, j].item(),
                                           "concept_target": values["concept_targets"][i, j].item()})
            group_features[name][split] = {
                g: values["u_concept_groups"][:, g, :].numpy() for g in range(k)
            }
    pd.DataFrame(score_rows).to_csv(outdir / "test_concept_predictions_long.csv", index=False)
    concept_accuracy_rows = []
    for name in names:
        values = data[name]["test"]
        targets = values["concept_targets"].numpy() > 0.5
        predictions = values["concept_scores"].numpy() >= 0.5
        for j, cname in enumerate(concept_names):
            concept_accuracy_rows.append({"model": name, "concept_index": j, "concept_name": cname,
                                          "test_concept_accuracy": float((predictions[:, j] == targets[:, j]).mean())})
    pd.DataFrame(concept_accuracy_rows).to_csv(outdir / "test_concept_accuracy_by_concept.csv", index=False)

    # Per-group probes and four graph-aware joint probe families.
    probe_rows, params = [], []
    for name, dag in zip(names, dags):
        train = data[name]["train"]["concept_targets"].numpy()
        val = data[name]["val"]["concept_targets"].numpy()
        test = data[name]["test"]["concept_targets"].numpy()
        xsets = {split: group_features[name][split] for split in ("train", "val", "test")}
        for target in range(k):
            y_train, y_val, y_test = train[:, target], val[:, target], test[:, target]
            feature_sets: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            for source in range(k):
                feature_sets[f"group_{source}"] = tuple(xsets[s][source] for s in ("train", "val", "test"))
            # CREAM DAG CSV uses target rows and source columns (also used for C2Y).
            parent_groups = [g for g in range(k) if bool(dag.iloc[target, g])]
            allowed = sorted(set([target] + parent_groups))
            feature_sets["assigned_group"] = feature_sets[f"group_{target}"]
            feature_sets["other_allowed_groups"] = tuple(
                np.concatenate([xsets[s][g] for g in parent_groups if g != target], axis=1)
                if any(g != target for g in parent_groups) else xsets[s][target][:, :0]
                for s in ("train", "val", "test")
            )
            feature_sets["all_allowed_groups"] = tuple(
                np.concatenate([xsets[s][g] for g in allowed], axis=1) for s in ("train", "val", "test")
            )
            feature_sets["all_groups"] = tuple(
                np.concatenate([xsets[s][g] for g in range(k)], axis=1) for s in ("train", "val", "test")
            )
            for feature_name, (xtr, xva, xte) in feature_sets.items():
                result = probe_record(name, concept_names[target], feature_name,
                                      xtr, y_train, xva, y_val, xte, y_test,
                                      args.probe_c_values, args.bootstrap_repetitions)
                probe = result.pop("probe")
                result.update({"target_index": target,
                               "allowed_source_groups": ";".join(str(g) for g in allowed),
                               "allowed_source_concepts": ";".join(concept_names[g] for g in allowed)})
                probe_rows.append(result)
                if probe is not None:
                    clf = probe.named_steps["logisticregression"]
                    scaler = probe.named_steps["standardscaler"]
                    params.append({"model": name, "target_index": target, "target_concept": concept_names[target],
                                   "feature_set": feature_name, "coef": clf.coef_[0].tolist(),
                                   "intercept": clf.intercept_[0].item(), "scaler_mean": scaler.mean_.tolist(),
                                   "scaler_scale": scaler.scale_.tolist(), "best_C": result["best_C"]})

    probe_df = pd.DataFrame(probe_rows)
    probe_df.to_csv(outdir / "probe_auc_long.csv", index=False)
    # Convenient matrix files for Figure 1.
    for name in names:
        one = probe_df[(probe_df.model == name) & probe_df.feature_set.str.startswith("group_")]
        matrix = one.pivot(index="target_concept", columns="feature_set", values="test_auc").reindex(concept_names)
        matrix.to_csv(outdir / f"{name}_probe_auc_matrix.csv")
    (outdir / "probe_parameters.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    # Conditional predictive information proxy: test NLL(q0(C_i|C_k)) -
    # test NLL(q1(C_i|U_k,C_k)); positive values mean U_k adds predictive signal.
    conditional_rows = []
    for name in names:
        ytr = data[name]["train"]["concept_targets"].numpy()
        yva = data[name]["val"]["concept_targets"].numpy()
        yte = data[name]["test"]["concept_targets"].numpy()
        for target in range(k):
            for source in range(k):
                if target == source:
                    continue
                xtr0, xva0, xte0 = (y[:, source:source + 1] for y in (ytr, yva, yte))
                xtr1 = np.concatenate([xtr0, group_features[name]["train"][source]], axis=1)
                xva1 = np.concatenate([xva0, group_features[name]["val"][source]], axis=1)
                xte1 = np.concatenate([xte0, group_features[name]["test"][source]], axis=1)
                q0, c0, _ = fit_probe(xtr0, ytr[:, target], xva0, yva[:, target], args.probe_c_values)
                q1, c1, _ = fit_probe(xtr1, ytr[:, target], xva1, yva[:, target], args.probe_c_values)
                if q0 is None or q1 is None:
                    continue
                target_bin = binary_target(yte[:, target])
                nll0 = float(log_loss(target_bin, predict_probability(q0, xte0), labels=[0, 1]))
                nll1 = float(log_loss(target_bin, predict_probability(q1, xte1), labels=[0, 1]))
                conditional_rows.append({"model": name, "target_index": target, "target_concept": concept_names[target],
                                         "source_group_index": source, "source_concept": concept_names[source],
                                         "baseline_C": c0, "conditioned_C": c1, "nll_Ci_given_Ck": nll0,
                                         "nll_Ci_given_Uk_Ck": nll1, "conditional_info_proxy_nats": nll0 - nll1})
    conditional_df = pd.DataFrame(conditional_rows)
    conditional_df.to_csv(outdir / "conditional_information_long.csv", index=False)
    for name in names:
        mat = conditional_df[conditional_df.model == name].pivot(
            index="target_concept", columns="source_concept", values="conditional_info_proxy_nats"
        ).reindex(index=concept_names, columns=concept_names)
        np.fill_diagonal(mat.values, np.nan)
        mat.to_csv(outdir / f"{name}_conditional_information_matrix.csv")
    if len(names) == 2 and not conditional_df.empty:
        mats = []
        for name in names:
            mats.append(conditional_df[conditional_df.model == name].pivot(
                index="target_concept", columns="source_concept", values="conditional_info_proxy_nats"
            ).reindex(index=concept_names, columns=concept_names))
        difference = mats[1] - mats[0]
        np.fill_diagonal(difference.values, np.nan)
        difference.to_csv(outdir / f"delta_{names[1]}_minus_{names[0]}_conditional_information.csv")

    # Step 6--9: matched within-model patching and same-image cross-model transplant.
    test_labels = data[names[0]]["test"]["concept_targets"].numpy()
    matched_pairs = make_matched_pairs(test_labels, args.pairs_per_concept, seed=42)
    for pair in matched_pairs:
        pair["changed_concept"] = concept_names[pair["changed_concept_index"]]
    pair_df = pd.DataFrame(matched_pairs)
    pair_df.to_csv(outdir / "matched_image_pairs.csv", index=False)
    torch.save({"pairs": matched_pairs,
                "pair_types": ["target_change", "control_same_target"],
                "pairing_method": "greedy nearest match by Hamming distance on non-target concept labels",
                "concept_names": concept_names}, outdir / "matched_image_pairs.pt")
    signed, absolute, probability_delta, patch_pair_df = run_within_model_pair_patching(
        models, names, data, dags, concept_names, width, matched_pairs, device
    )
    save_within_patch_outputs(signed, absolute, probability_delta, patch_pair_df,
                              names, concept_names, outdir)
    run_cross_model_patching(models, names, data, dags, concept_names, width, device, outdir)

    manifest = {
        "dataset_argument": args.dataset,
        "dataset_name": cfgs[0]["dataset_name"],
        "models": [{"name": name, "config": str(cfg), "checkpoint": str(ckpt),
                    "dag": str(resolve_path(config["paths"]["DAG_file"], root)),
                    "num_concepts": config["hyperparameters_model2"]["num_concepts"],
                    "num_exogenous": config["hyperparameters_model2"]["num_exogenous"],
                    "num_side_channel": config["hyperparameters_model2"].get("num_side_channel", 0)}
                   for name, cfg, ckpt, config in zip(names, [args.config_a, args.config_b], checkpoints, cfgs)],
        "split_sizes": {split: len(data[names[0]][split]["concept_targets"]) for split in ("train", "val", "test")},
        "concept_names": concept_names,
        "group_width": width,
        "probe_c_values": args.probe_c_values,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "probe_fit_split": "train",
        "probe_selection_split": "val",
        "probe_evaluation_split": "test",
        "analysis_scope": "representation probes/conditional predictive information, within-model matched-pair U-group patching, and cross-model same-image U-group transplants; joint multi-group patching is not included",
        "pairs_per_concept_requested": args.pairs_per_concept,
        "matched_pairing_method": "greedy nearest match by Hamming distance on non-target concept labels; control pairs preserve the target label",
        "cross_model_patch_modes": ["raw donor group transplant", "validation-fitted ridge-aligned donor group transplant"],
        "cross_model_patch_caveat": "Cross-model patch changes receiver-model input activations; it measures compatibility/utilization, not alone semantic information redistribution.",
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved U activations, concept logits/scores, probe CSVs, and manifest to {outdir}")


if __name__ == "__main__":
    main()
