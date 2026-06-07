#!/usr/bin/env bash
# Convert ensemble expert graphs (.pt) to DAG CSV files
# required by simple_main.py (standalone CREAM).
# Run this ONCE before submitting cream_noisy_ensemble jobs.
set -euo pipefail

PROJECT_ROOT="/home/dani00003/mCREAM"
CONDA_PYTHON="/home/dani00003/miniconda3/envs/mcream/bin/python"

if [ -x "$CONDA_PYTHON" ]; then
    PYTHON_BIN="$CONDA_PYTHON"
else
    echo "ERROR: Conda env not found at $CONDA_PYTHON" >&2
    exit 127
fi

cd "$PROJECT_ROOT"

echo "=============================================="
echo "Converting expert graphs to DAG CSVs"
echo "=============================================="
echo "HOST=$(hostname)"
"$PYTHON_BIN" -V

"$PYTHON_BIN" convert_expert_graphs_to_csv.py --dataset all

echo ""
echo "=============================================="
echo "Conversion complete!"
echo "CSVs saved under:"
echo "  data/FashionMNIST/expert_graphs/ensemble/*/cream_noisy_dags/"
echo "  data/CelebA/expert_graphs/ensemble/*/cream_noisy_dags/"
echo "=============================================="
