import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_undirected_simple_graph(dag_path: Path) -> np.ndarray:
    dag = pd.read_csv(dag_path, index_col=0).astype(bool).values
    adj = np.logical_or(dag, dag.T)
    np.fill_diagonal(adj, False)
    return adj.astype(np.int8)


def sample_gnm_graph(n_nodes: int, n_edges: int, rng: np.random.Generator) -> np.ndarray:
    n_possible = n_nodes * (n_nodes - 1) // 2
    chosen = rng.choice(n_possible, size=n_edges, replace=False)
    rows, cols = np.triu_indices(n_nodes, k=1)
    adj = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    adj[rows[chosen], cols[chosen]] = 1
    adj[cols[chosen], rows[chosen]] = 1
    return adj


def graph_metrics(adj: np.ndarray) -> dict[str, float]:
    degrees = adj.sum(axis=1).astype(float)
    n_nodes = adj.shape[0]

    # triangles = trace(A^3) / 6 for an undirected simple graph.
    a2 = adj @ adj
    triangles = float(np.trace(a2 @ adj) / 6.0)

    clustering_values = []
    for node in range(n_nodes):
        neighbors = np.flatnonzero(adj[node])
        degree = len(neighbors)
        if degree < 2:
            clustering_values.append(0.0)
            continue
        subgraph = adj[np.ix_(neighbors, neighbors)]
        neighbor_edges = subgraph.sum() / 2.0
        clustering_values.append(neighbor_edges / (degree * (degree - 1) / 2.0))

    eigenvalues = np.linalg.eigvalsh(adj.astype(float))
    lambda_1 = float(eigenvalues[-1])
    lambda_2 = float(eigenvalues[-2]) if len(eigenvalues) > 1 else 0.0

    return {
        "average_degree": float(degrees.mean()),
        "degree_variance": float(degrees.var()),
        "clustering_coefficient": float(np.mean(clustering_values)),
        "num_triangles": triangles,
        "largest_eigenvalue": lambda_1,
        "spectral_gap": lambda_1 - lambda_2,
    }


def z_score(original: float, mean: float, std: float) -> float:
    if std == 0:
        return float("nan")
    return (original - mean) / std


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dag", default="data/CUB/CUB_DAG_only_Gc.csv")
    parser.add_argument("--output_dir", default="experiments/CUB/graph_randomness_er")
    parser.add_argument("--n_graphs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dag_path = Path(args.dag)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_adj = load_undirected_simple_graph(dag_path)
    n_nodes = original_adj.shape[0]
    n_edges = int(original_adj.sum() // 2)
    er_p = 2 * n_edges / (n_nodes * (n_nodes - 1))

    print(f"Original graph: N={n_nodes}, E={n_edges}, p={er_p:.6f}")
    original_metrics = graph_metrics(original_adj)

    rng = np.random.default_rng(args.seed)
    ensemble_rows = []
    for graph_idx in range(args.n_graphs):
        er_adj = sample_gnm_graph(n_nodes, n_edges, rng)
        metrics = graph_metrics(er_adj)
        metrics.update(
            {
                "graph_idx": graph_idx,
                "n_nodes": n_nodes,
                "n_edges": int(er_adj.sum() // 2),
            }
        )
        ensemble_rows.append(metrics)
        print(f"Computed ER graph {graph_idx + 1}/{args.n_graphs}")

    original_df = pd.DataFrame([{**original_metrics, "n_nodes": n_nodes, "n_edges": n_edges, "er_p": er_p}])
    ensemble_df = pd.DataFrame(ensemble_rows)

    metric_names = list(original_metrics.keys())
    z_rows = []
    for metric in metric_names:
        mean = float(ensemble_df[metric].mean())
        std = float(ensemble_df[metric].std(ddof=1))
        original = float(original_metrics[metric])
        z_rows.append(
            {
                "metric": metric,
                "original": original,
                "er_mean": mean,
                "er_std": std,
                "z_score": z_score(original, mean, std),
            }
        )

    z_df = pd.DataFrame(z_rows)

    original_df.to_csv(output_dir / "original_metrics.csv", index=False)
    ensemble_df.to_csv(output_dir / "er_ensemble_metrics.csv", index=False)
    z_df.to_csv(output_dir / "z_scores.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "dag": str(dag_path),
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "er_p": er_p,
                "n_graphs": args.n_graphs,
                "seed": args.seed,
                "graph_model": "Uniform undirected fixed-edge graph G(n, m), equivalent to ER baseline with matched edge count",
                "preprocessing": "symmetrized adjacency with diagonal removed",
            },
            indent=2,
        )
    )

    print(f"Saved results to {output_dir}")
    print(z_df)


if __name__ == "__main__":
    main()
