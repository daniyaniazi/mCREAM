#!/usr/bin/env bash
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
echo "HOST=$(hostname)"
"$PYTHON_BIN" -V

echo "Running CUB ER graph randomness analysis..."
"$PYTHON_BIN" scripts/er_graph_randomness_analysis.py \
    --dag data/CUB/CUB_DAG_only_Gc.csv \
    --output_dir experiments/CUB/graph_randomness_er \
    --n_graphs 100 \
    --seed 42

echo "Done!"
