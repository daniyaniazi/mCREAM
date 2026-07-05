"""
Generate consensus expert graphs for mCREAM Alpha experiment.

Scheme:
  1. Start from GT graph
  2. Apply BASE noise p to get shared base graph (same for all experts)
  3. Each expert adds PRIVATE noise q on top (different per expert)
  4. Result: any two experts agree on ~(1 - 2q(1-q)) of edges

  For q=0.05 → pairwise consensus ≈ 90.25%

Noise levels (base corruption p):
  low    = 0.25  (25% of GT edges flipped in base)
  medium = 0.50  (50% of GT edges flipped in base)
  high   = 0.75  (75% of GT edges flipped in base)

Noise type: mixed — flip picks uniformly from {add absent, delete existing}

Usage:
  python generate_consensus_expert_graphs.py --dataset cfmnist
  python generate_consensus_expert_graphs.py --dataset celeba
  python generate_consensus_expert_graphs.py --dataset cfmnist --consensus 0.90
"""

import argparse
import numpy as np
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.expert_graphs.generation import load_and_split_dag, save_expert_graphs


# ── Dataset configs ───────────────────────────────────────────────────────────
DATASETS = {
    "cfmnist": {
        "dag_path":    "data/FashionMNIST/Complete_Concept_FMNIST_DAG.csv",
        "num_classes": 10,
        "num_experts": 5,
        "output_base": "data/FashionMNIST/expert_graphs/consensus",
    },
    "celeba": {
        "dag_path":    "data/CelebA/final_DAG_unfair.csv",
        "num_classes": 1,
        "num_experts": 5,
        "output_base": "data/CelebA/expert_graphs/consensus",
    },
    "cub": {
        "dag_path":    "data/CUB/CUB_DAG_only_Gc.csv",
        "num_classes": 200,
        "num_experts": 5,
        "output_base": "data/CUB/expert_graphs/consensus",
    },
}

# Base noise = how much GT is corrupted before private flips
BASE_NOISE = {"low": 0.15, "medium": 0.25, "high": 0.50}


# ── Core generation ───────────────────────────────────────────────────────────

def flip_edges(G: np.ndarray, p_flip: float, rng: np.random.Generator,
               preserve_diagonal: bool = True) -> np.ndarray:
    """
    Flip p_flip fraction of ALL positions (add absent OR delete existing).
    No distinction between add/delete — each selected position is simply toggled.
    Returns new graph with same shape.
    """
    G_out = G.copy()
    K = G.shape[0]
    total = K * K
    positions = np.arange(total)

    if preserve_diagonal:
        # Remove diagonal positions from candidate set
        diag_pos = np.array([i * K + i for i in range(K)])
        positions = np.setdiff1d(positions, diag_pos)

    n_flip = max(1, int(len(positions) * p_flip))
    chosen = rng.choice(positions, size=n_flip, replace=False)
    rows, cols = chosen // K, chosen % K

    for r, c in zip(rows, cols):
        G_out[r, c] = 1 - G_out[r, c]   # toggle: 0→1 (add) or 1→0 (delete)

    return G_out


def generate_consensus_experts(
    G_star: np.ndarray,       # [K, K] GT u2c graph as numpy bool/int
    M: int,                    # number of experts
    p_base: float,             # base noise: fraction of GT edges corrupted (shared)
    p_private: float,          # private noise per expert (causes disagreement)
    base_seed: int = 42,
    preserve_diagonal: bool = True,
) -> list[np.ndarray]:
    """
    Generate M expert graphs with controlled consensus.

    Step 1: corrupt GT by p_base → shared base graph
    Step 2: each expert independently corrupts base by p_private

    Pairwise consensus ≈ 1 - 2*p_private*(1-p_private)
    For p_private=0.05 → consensus ≈ 90.25%

    Returns list of M numpy bool arrays.
    """
    rng_base    = np.random.default_rng(base_seed)
    G_base      = flip_edges(G_star.astype(int), p_base, rng_base, preserve_diagonal)

    experts = []
    for m in range(M):
        rng_m = np.random.default_rng(base_seed + m + 1)
        G_m   = flip_edges(G_base, p_private, rng_m, preserve_diagonal)
        experts.append(G_m.astype(bool))

    return experts, G_base


def verify_consensus(experts: list[np.ndarray]) -> float:
    """Compute mean pairwise agreement across all expert pairs."""
    M = len(experts)
    agreements = []
    for i in range(M):
        for j in range(i + 1, M):
            agree = (experts[i] == experts[j]).mean()
            agreements.append(agree)
    return float(np.mean(agreements))


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_for_dataset(dataset_name: str, consensus: float, M: int = None,
                         levels: list = None):
    cfg        = DATASETS[dataset_name]
    dag_path   = cfg["dag_path"]
    num_classes = cfg["num_classes"]
    M          = M or cfg["num_experts"]
    output_base = cfg["output_base"]

    # private flip q that gives target consensus: 1 - 2q(1-q) = consensus
    # → q = (1 - sqrt(2*consensus - 1)) / 2   (smaller root)
    # Simpler: just use fixed q=0.05 for ~90% consensus, q=0.10 for ~82%
    p_private = (1 - (2 * consensus - 1) ** 0.5) / 2
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}  |  M={M}")
    print(f"Target consensus: {consensus:.0%}  →  private flip q={p_private:.3f}")
    print(f"{'='*60}")

    # Load GT
    u2c_star, c2y_star = load_and_split_dag(dag_path, num_classes)
    G_star = u2c_star.numpy().astype(int)
    K = G_star.shape[0]
    print(f"GT u2c: {K}×{K}, {G_star.sum()} edges")

    run_levels = {k: v for k, v in BASE_NOISE.items() if levels is None or k in levels}
    for level, p_base in run_levels.items():
        output_dir = Path(output_base) / f"{level}"
        if (output_dir / "config.yaml").exists():
            print(f"  [SKIP] {level} — already exists at {output_dir}")
            continue

        print(f"\n  Generating {level} (p_base={p_base}, p_private={p_private:.3f})...")

        experts, G_base = generate_consensus_experts(
            G_star, M,
            p_base=p_base,
            p_private=p_private,
            base_seed=42,
        )

        # Verify
        actual_consensus = verify_consensus(experts)
        base_vs_gt = (G_base == G_star).mean()
        print(f"    Base vs GT agreement:      {base_vs_gt:.1%}  (target: {1-p_base:.0%})")
        print(f"    Pairwise expert consensus: {actual_consensus:.1%}  (target: {consensus:.0%})")

        # Print per-expert corruption vs GT
        for m, G_m in enumerate(experts):
            vs_gt  = (G_m == G_star).mean()
            vs_base = (G_m == G_base).mean()
            adds   = ((G_m.astype(int) - G_star) == 1).sum()
            dels   = ((G_star - G_m.astype(int)) == 1).sum()
            print(f"    expert_{m}: vs_GT={vs_gt:.1%}  vs_base={vs_base:.1%}  "
                  f"(+{adds} edges, -{dels} edges)")

        # Convert to tensors for save_expert_graphs
        expert_u2c = [torch.tensor(G_m, dtype=torch.bool) for G_m in experts]
        # c2y: all experts use GT c2y (only u2c is perturbed)
        expert_c2y = [c2y_star.bool() for _ in range(M)]

        save_config = {
            "dag_path":         dag_path,
            "num_classes":      num_classes,
            "num_experts":      M,
            "noise_type":       "consensus",
            "noise_level":      level,
            "p_base":           p_base,
            "p_private":        float(p_private),
            "target_consensus": consensus,
            "actual_consensus": float(actual_consensus),
            "seed":             42,
        }

        save_expert_graphs(expert_u2c, expert_c2y, output_dir, save_config)

        # Save GT alongside
        gt_dir = output_dir / "ground_truth"
        gt_dir.mkdir(exist_ok=True)
        torch.save(u2c_star.bool(), gt_dir / "u2c_star.pt")
        torch.save(c2y_star.bool(), gt_dir / "c2y_star.pt")

        print(f"    Saved to: {output_dir}")

    print(f"\nDone: {dataset_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   choices=["cfmnist", "celeba", "cub", "all"], default="cfmnist")
    parser.add_argument("--consensus", type=float, default=0.90,
                        help="Target pairwise consensus (default 0.90 = 90%%)")
    parser.add_argument("--num_experts", type=int, default=None)
    parser.add_argument("--levels", nargs="+", choices=["low","medium","high"], default=None,
                        help="Which noise levels to generate (default: all three)")
    args = parser.parse_args()

    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        generate_for_dataset(ds, args.consensus, args.num_experts, args.levels)


if __name__ == "__main__":
    main()
