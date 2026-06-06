#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/dani00003/mCREAM"
CONDA_PYTHON="/home/dani00003/miniconda3/envs/mcream/bin/python"

if [ -x "$CONDA_PYTHON" ]; then
    PYTHON_BIN="$CONDA_PYTHON"
else
    echo "ERROR: Conda env not found" >&2; exit 127
fi

cd "$PROJECT_ROOT"
echo "HOST=$(hostname)"
"$PYTHON_BIN" -V
nvidia-smi || true

# Convert expert graphs to DAG CSV if not done yet
DAG_CSV="$PROJECT_ROOT/data/CelebA/expert_graphs/ensemble/reversal_low/cream_noisy_dags/expert_0.csv"
if [ ! -f "$DAG_CSV" ]; then
    echo "Converting expert graphs to DAG CSVs..."
    "$PYTHON_BIN" convert_expert_graphs_to_csv.py --dataset celeba
fi

echo "Running standalone CREAM on reversal/low/expert0..."
"$PYTHON_BIN" simple_main.py --config all_configs/mcream_configs/celeba/cream_noisy_ensemble/cream_noisy_reversal_low_expert0.yaml
echo "Done!"
