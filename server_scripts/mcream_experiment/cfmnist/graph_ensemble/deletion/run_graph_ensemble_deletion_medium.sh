#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="/home/dani00003/mCREAM"
CONDA_PYTHON="/home/dani00003/miniconda3/envs/mcream/bin/python"
if [ -x "$CONDA_PYTHON" ]; then PYTHON_BIN="$CONDA_PYTHON"
else echo "ERROR: Conda env not found" >&2; exit 127; fi
cd "$PROJECT_ROOT"
echo "HOST=$(hostname)"; "$PYTHON_BIN" -V; nvidia-smi || true

EXPERT_DIR="$PROJECT_ROOT/data/FashionMNIST/expert_graphs/ensemble/deletion_medium"
if [ ! -f "$EXPERT_DIR/config.yaml" ]; then
    echo "Generating expert graphs for deletion/medium..."
    "$PYTHON_BIN" generate_ensemble_expert_graphs.py --dataset cfmnist --num_experts 5
fi

echo "Running mCREAM Graph Ensemble (deletion/medium)..."
"$PYTHON_BIN" mcream_graph_ensemble_main.py     --config all_configs/mcream_graph_ensemble_configs/cfmnist/deletion/graph_ensemble_deletion_medium.yaml
echo "Done!"
