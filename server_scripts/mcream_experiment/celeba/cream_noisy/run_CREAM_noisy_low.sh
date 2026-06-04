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
nvidia-smi || true

if [ ! -f "data/CelebA/noisy_dags/noisy_dag_low.csv" ]; then
    echo "Generating noisy DAG CSV..."
    "$PYTHON_BIN" generate_single_noisy_dags.py --dataset celeba --level low
fi

"$PYTHON_BIN" simple_main.py --config all_configs/mcream_configs/celeba/cream_noisy/CREAM_noisy_low.yaml
echo "Done!"
