"""
Run NEC/ANEC + ADI on all saved CREAM checkpoints.

Usage:
    python evaluate_metrics.py --config all_configs/best_hparams/CREAM/CREAM_awa2_soft_seed0_config.yaml
    python evaluate_metrics.py --config all_configs/best_hparams/CREAM/CREAM_cub_soft_seed0_config.yaml --no_adi
"""
import argparse
import json
import torch
from pathlib import Path

import pytorch_lightning as pl
from src.utils import get_component_with_dicts, load_config
from src.models import Template_CBM_MultiClass, UtoY_model
from src.cream_metrics import evaluate_all, save_concept_saliency_maps
from src.saving_intermediate_utils import save_activation_percentiles


def find_checkpoint(exp_dir: Path) -> Path:
    ckpts = sorted(exp_dir.rglob('*.ckpt'))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found under {exp_dir}")
    # prefer 'best' over 'last'
    best = [c for c in ckpts if 'best' in c.name]
    return best[0] if best else ckpts[-1]


def get_concept_names(datamodule, dag_path: str, num_concepts: int, num_classes: int) -> list[str]:
    if hasattr(datamodule, 'concept_names'):
        names = list(datamodule.concept_names)
        if len(names) == num_concepts:
            return names

    try:
        import pandas as pd
        dag_names = list(pd.read_csv(dag_path, index_col=0).index)
        names = dag_names[:-num_classes] if num_classes > 0 else dag_names
        if len(names) == num_concepts:
            return names
    except Exception:
        pass

    return [f'concept_{i}' for i in range(num_concepts)]


def parse_heatmap_indices(indices: str | None) -> list[int] | None:
    if not indices:
        return None
    return [int(idx.strip()) for idx in indices.split(',') if idx.strip()]


def read_heatmap_ids(ids_path: str | None) -> list[str] | None:
    if not ids_path:
        return None
    with open(ids_path) as f:
        return [line.strip() for line in f if line.strip()]


def get_split_dataset_and_loader(datamodule, split: str):
    if split == 'train':
        return datamodule.train_dataset, datamodule.train_dataloader()
    if split == 'val':
        return datamodule.val_dataset, datamodule.val_dataloader()
    if split == 'test':
        return datamodule.test_data, datamodule.test_dataloader()
    raise ValueError(f"Unsupported split: {split}")


def parse_split_list(splits: str) -> list[str]:
    return [split.strip() for split in splits.split(',') if split.strip()]


def find_indices_by_image_ids(datamodule, image_ids: list[str], splits: list[str]) -> dict[str, list[int]]:
    from pathlib import Path

    matched_by_split = {split: [] for split in splits}
    missing_ids = []

    for image_id in image_ids:
        matched = False
        for split in splits:
            dataset, _ = get_split_dataset_and_loader(datamodule, split)
            if not hasattr(dataset, 'data'):
                raise RuntimeError(f"Could not inspect {split} dataset data for image-id matching.")

            for idx, item in enumerate(dataset.data):
                img_path = str(item.get('img_path', ''))
                stem = Path(img_path).stem
                name = Path(img_path).name
                if image_id == stem or image_id == name or image_id in img_path:
                    matched_by_split[split].append(idx)
                    matched = True
                    break
            if matched:
                break

        if not matched:
            missing_ids.append(image_id)

    if missing_ids:
        print(f"WARNING: Could not find these requested image IDs in splits {splits}:")
        for image_id in missing_ids:
            print(f"  {image_id}")

    matched_count = sum(len(indices) for indices in matched_by_split.values())
    print(f"Matched {matched_count}/{len(image_ids)} requested image IDs.")
    for split, indices in matched_by_split.items():
        print(f"  {split}: {len(indices)} matches")
    return matched_by_split


def random_split_indices(datamodule, split: str, n: int, seed: int) -> list[int]:
    if n <= 0:
        return []
    dataset, _ = get_split_dataset_and_loader(datamodule, split)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=generator)
    return perm[:min(n, len(dataset))].tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--no_adi', action='store_true', help='Skip ADI (faster)')
    parser.add_argument('--no_heatmaps', action='store_true', help='Skip CAM heatmap PNG/PT export')
    parser.add_argument('--heatmap_images', type=int, default=10, help='Number of test samples to export CAM heatmaps for')
    parser.add_argument('--heatmap_indices', default=None, help='Comma-separated test-set indices to export, e.g. 12,45,88')
    parser.add_argument('--heatmap_ids', default=None, help='Text file with one image ID/path fragment per line')
    parser.add_argument('--heatmap_split', default='test', choices=['train', 'val', 'test'], help='Split used with --heatmap_images/--heatmap_indices')
    parser.add_argument('--heatmap_search_splits', default='test,val', help='Comma-separated splits searched with --heatmap_ids')
    parser.add_argument('--heatmap_random_val', type=int, default=0, help='Also export this many random validation samples')
    parser.add_argument('--heatmap_top_k', type=int, default=None, help='Save only the top-k predicted concept heatmaps per image')
    parser.add_argument('--only_heatmaps', action='store_true', help='Export CAM heatmaps and skip NEC/ANEC/ADI')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device)

    dataset_name = config['dataset_name']
    model_name   = config['model_name']
    seed         = config['seed']
    mode         = config['mode']
    config_folder = Path(args.config).parent.name

    # experiment_name for finding the right folder
    exp_name = config.get('experiment_name',
                          Path(args.config).stem.split('_')[0] + '_' +
                          Path(args.config).stem.split('_')[1])

    exp_dir = (Path(config['paths']['default_root_dir'])
               / dataset_name / mode / model_name / config_folder / exp_name)

    print(f"Dataset: {dataset_name}  Model: {model_name}  Seed: {seed}")
    print(f"Exp dir: {exp_dir}")

    # ── Load checkpoint ────────────────────────────────────────────────────
    ckpt_path = find_checkpoint(exp_dir)
    print(f"Loading checkpoint: {ckpt_path}")

    from pandas import read_csv
    dag_path = config['paths']['DAG_file']
    df_dag   = read_csv(dag_path, index_col=0)
    causal_graph = torch.tensor(df_dag.values.astype(bool), dtype=torch.bool)

    softmax_mask = None
    if 'softmax_mask' in config['paths']:
        with open(config['paths']['softmax_mask']) as f:
            softmax_mask = json.load(f)

    # Build model1 and model2 — needed because load_from_checkpoint
    # can't reconstruct them (saved with ignore=["model1","model2"])
    model_class = get_component_with_dicts('model', model_name)
    hyperparams_model2 = config['hyperparameters_model2']
    hyperparams        = config['hyperparameters']

    # model1 — backbone (weights will be overwritten by checkpoint)
    model1 = model_class(
        num_classes=hyperparams_model2['num_classes'],
        learning_rate=hyperparams.get('learning_rate', 1e-4),
        frozen=hyperparams.get('frozen_model1', True),
    )

    # model2 — UtoY
    model2_kwargs = dict(**hyperparams_model2, causal_graph=causal_graph)
    if softmax_mask is not None:
        model2_kwargs['mutually_exclusive_concepts'] = softmax_mask
    model2 = UtoY_model(**model2_kwargs)

    # num_hyperparameters for Template_CBM_MultiClass
    num_hparams = {k: hyperparams_model2[k] for k in
                   ['num_classes', 'num_exogenous', 'num_side_channel',
                    'num_concepts', 'concept_representation']}

    # Template_CBM_MultiClass is constructed with model1.concept_extractor,
    # not the full model1 — see simple_main.py line 268
    model = Template_CBM_MultiClass.load_from_checkpoint(
        ckpt_path,
        map_location=device,
        strict=True,
        model1=model1.concept_extractor,
        model2=model2,
        **num_hparams,
        **hyperparams,
    )
    model.eval().to(device)

    # ── Load dataset ───────────────────────────────────────────────────────
    pl.seed_everything(seed)
    dataset_class = get_component_with_dicts('dataset', dataset_name)
    datamodule = dataset_class(**config['dataset_params'])
    datamodule.setup(stage='fit')
    datamodule.setup(stage='test')

    out_dir = exp_dir / 'last_metrics'
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_heatmaps:
        concept_names = get_concept_names(
            datamodule=datamodule,
            dag_path=dag_path,
            num_concepts=hyperparams_model2['num_concepts'],
            num_classes=hyperparams_model2['num_classes'],
        )

        heatmap_jobs: dict[str, list[int] | None] = {}
        heatmap_ids = read_heatmap_ids(args.heatmap_ids)
        if heatmap_ids is not None:
            search_splits = parse_split_list(args.heatmap_search_splits)
            heatmap_jobs.update(find_indices_by_image_ids(datamodule, heatmap_ids, search_splits))
        else:
            heatmap_jobs[args.heatmap_split] = parse_heatmap_indices(args.heatmap_indices)

        if args.heatmap_random_val > 0:
            val_indices = random_split_indices(datamodule, 'val', args.heatmap_random_val, seed)
            existing_val_indices = heatmap_jobs.get('val') or []
            heatmap_jobs['val'] = sorted(set(existing_val_indices + val_indices))
            print(f"Random validation indices: {val_indices}")

        for split, heatmap_indices in heatmap_jobs.items():
            if heatmap_indices == []:
                continue
            _, loader = get_split_dataset_and_loader(datamodule, split)
            heatmap_dir = out_dir / 'heatmaps' / split
            print(f"Saving CAM heatmap PNG/PT files to {heatmap_dir}...")
            save_concept_saliency_maps(
                model=model,
                dataloader=loader,
                device=device,
                concept_names=concept_names,
                save_dir=str(heatmap_dir),
                n_images=args.heatmap_images,
                save_pt=True,
                sample_indices=heatmap_indices,
                top_k_concepts=args.heatmap_top_k,
            )

        if args.only_heatmaps:
            print("Heatmap export complete; skipping NEC/ANEC/ADI.")
            return

    # ── Compute activation percentiles for NEC ────────────────────────────
    print("Computing activation percentiles for NEC intervention mapping...")
    if config['hyperparameters_model2']['concept_representation'] in ('soft', 'group_soft', 'logits'):
        perc_df = save_activation_percentiles(
            dataset=datamodule,
            dataset_name=dataset_name,
            model=model,
            DAG_path=dag_path,
        )
        model.intervention_percentile_df = perc_df

    # ── Run all metrics ────────────────────────────────────────────────────
    budgets = [5, 10, 15, 20, 25, 30] if dataset_name == 'CUB' else [5, 10, 15, 20, 25, 30]
    results = evaluate_all(
        model=model,
        datamodule=datamodule,
        device=device,
        budgets=budgets,
        run_adi=not args.no_adi,
    )
    results['seed']    = seed
    results['dataset'] = dataset_name

    # ── Save results ───────────────────────────────────────────────────────
    out_path = out_dir / 'nec_anec_adi.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
