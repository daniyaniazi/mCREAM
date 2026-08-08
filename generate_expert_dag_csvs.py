"""
Convert expert graph .pt files (u2c + c2y) into full DAG CSVs
matching the format expected by simple_main.py (same as adjacency_v1.csv).

For each expert, combines u2c and c2y into a full (K+C)×(K+C) boolean matrix
and saves as CSV with TRUE/FALSE strings.

Usage:
    python generate_expert_dag_csvs.py --dataset CUB
    python generate_expert_dag_csvs.py --dataset CelebA
    python generate_expert_dag_csvs.py --dataset cfmnist
    python generate_expert_dag_csvs.py --dataset all
"""
import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path


DATASET_CONFIG = {
    'CUB': {
        'graphs_root': './data/CUB/expert_graphs/consensus',
        'gt_dag':      './data/CUB/CUB_DAG_only_Gc.csv',
        'num_concepts': 112,
        'num_classes':  200,
    },
    'CelebA': {
        'graphs_root': './data/CelebA/expert_graphs/consensus',
        'gt_dag':      './data/CelebA/final_DAG_unfair.csv',
        'num_concepts': 7,
        'num_classes':  1,
    },
    'cfmnist': {
        'graphs_root': './data/FashionMNIST/expert_graphs/consensus',
        'gt_dag':      './data/FashionMNIST/Complete_Concept_FMNIST_DAG.csv',
        'num_concepts': 11,
        'num_classes':  10,
    },
}


def pt_to_full_dag_csv(u2c_pt, c2y_pt, gt_dag_path, num_concepts, num_classes, out_path):
    """Build full (K+C)×(K+C) DAG CSV from u2c and c2y .pt tensors."""
    gt_df = pd.read_csv(gt_dag_path, index_col=0)
    row_names = list(gt_df.index)
    col_names = list(gt_df.columns)

    # Load tensors
    u2c = torch.load(u2c_pt, weights_only=True).bool().numpy()  # (K, K)
    c2y = torch.load(c2y_pt, weights_only=True).bool().numpy()  # (C, K) or (C, K+C)

    K = num_concepts
    C = num_classes
    N = K + C

    # Build full matrix — start from GT to preserve structure of unused blocks
    full = (gt_df.values != 0).astype(bool) if gt_df.dtypes.iloc[0] == object else gt_df.values.astype(bool)
    full = np.zeros((N, N), dtype=bool)

    # Fill u2c block (top-left K×K)
    full[:K, :K] = u2c[:K, :K]

    # Fill c2y block (bottom C rows, first K cols)
    c2y_part = c2y[:C, :K]
    full[K:K+C, :K] = c2y_part

    # Convert to TRUE/FALSE strings
    str_mat = np.where(full, 'TRUE', 'FALSE')
    out_df = pd.DataFrame(str_mat, index=row_names[:N], columns=col_names[:N])
    out_df.to_csv(out_path)


def process_dataset(ds_name, levels=('low', 'medium', 'high'), n_experts=5):
    cfg = DATASET_CONFIG[ds_name]
    graphs_root = Path(cfg['graphs_root'])
    gt_dag      = cfg['gt_dag']
    K           = cfg['num_concepts']
    C           = cfg['num_classes']

    for level in levels:
        level_dir = graphs_root / level
        u2c_dir   = level_dir / 'u2c'
        c2y_dir   = level_dir / 'c2y'

        if not u2c_dir.exists():
            print(f'  [SKIP] {u2c_dir} not found')
            continue

        for i in range(n_experts):
            u2c_pt = u2c_dir / f'expert_{i}.pt'
            c2y_pt = c2y_dir / f'expert_{i}.pt'
            out    = level_dir / f'expert_graph_{i}.csv'

            if not u2c_pt.exists():
                print(f'  [SKIP] {u2c_pt} not found')
                continue
            if not c2y_pt.exists():
                print(f'  [SKIP] {c2y_pt} not found — using GT c2y')
                # fall back to GT c2y
                gt_df  = pd.read_csv(gt_dag, index_col=0)
                gt_mat = (gt_df.values != 0).astype(bool) if gt_df.dtypes.iloc[0] == object else gt_df.values.astype(bool)
                c2y_pt_fallback = None
            else:
                c2y_pt_fallback = c2y_pt

            if c2y_pt_fallback is None:
                # write with GT c2y
                gt_df  = pd.read_csv(gt_dag, index_col=0)
                u2c    = torch.load(u2c_pt, weights_only=True).bool().numpy()
                N      = K + C
                full   = np.zeros((N, N), dtype=bool)
                full[:K, :K] = u2c[:K, :K]
                gt_mat = (gt_df.values != 0).astype(bool) if gt_df.dtypes.iloc[0] == object else gt_df.values.astype(bool)
                full[K:, :] = gt_mat[K:, :]
                row_names = list(gt_df.index)[:N]
                col_names = list(gt_df.columns)[:N]
                str_mat   = np.where(full, 'TRUE', 'FALSE')
                pd.DataFrame(str_mat, index=row_names, columns=col_names).to_csv(out)
            else:
                pt_to_full_dag_csv(u2c_pt, c2y_pt_fallback, gt_dag, K, C, out)

            print(f'  Saved: {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='all', choices=list(DATASET_CONFIG.keys()) + ['all'])
    parser.add_argument('--levels', nargs='+', default=['low', 'medium', 'high'])
    parser.add_argument('--n_experts', type=int, default=5)
    args = parser.parse_args()

    datasets = list(DATASET_CONFIG.keys()) if args.dataset == 'all' else [args.dataset]
    for ds in datasets:
        print(f'\nProcessing {ds}...')
        process_dataset(ds, levels=args.levels, n_experts=args.n_experts)
