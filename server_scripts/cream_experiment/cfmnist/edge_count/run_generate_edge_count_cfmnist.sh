#!/usr/bin/env bash
# Step 1: Generate DAG CSVs and configs for edge count experiments (cfmnist)
# Run this ONCE before submitting CREAM jobs.
# No GPU needed.
set -euo pipefail

PROJECT_ROOT="/home/dani00003/mCREAM"
CONDA_PYTHON="/home/dani00003/miniconda3/envs/mcream/bin/python"

if [ -x "$CONDA_PYTHON" ]; then PYTHON_BIN="$CONDA_PYTHON"
else echo "ERROR: Conda env not found" >&2; exit 127; fi

cd "$PROJECT_ROOT"
echo "HOST=$(hostname)"; "$PYTHON_BIN" -V

echo "Generating edge count experiment graphs (cfmnist, delta=5, 5 seeds)..."
"$PYTHON_BIN" generate_edge_count_experiments.py --dataset cfmnist --delta 5 --n_seeds 5

echo "Generating single-edge perturbation graphs (cfmnist, 10 additions)..."
"$PYTHON_BIN" generate_single_edge_perturbation.py --dataset cfmnist --n_additions 10

echo "Done! Graphs saved under:"
echo "  data/FashionMNIST/edge_count_experiments/"
echo "  data/FashionMNIST/single_edge_perturbation/"
echo "Configs saved under:"
echo "  all_configs/mcream_configs/cfmnist/edge_count_experiments/"
echo "  all_configs/mcream_configs/cfmnist/single_edge_perturbation/"
