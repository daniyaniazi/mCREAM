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
from src.cream_metrics import evaluate_all
from src.saving_intermediate_utils import save_activation_percentiles


def find_checkpoint(exp_dir: Path) -> Path:
    ckpts = sorted(exp_dir.rglob('*.ckpt'))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found under {exp_dir}")
    # prefer 'best' over 'last'
    best = [c for c in ckpts if 'best' in c.name]
    return best[0] if best else ckpts[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--no_adi', action='store_true', help='Skip ADI (faster)')
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
    out_dir = exp_dir / 'last_metrics'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'nec_anec_adi.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
