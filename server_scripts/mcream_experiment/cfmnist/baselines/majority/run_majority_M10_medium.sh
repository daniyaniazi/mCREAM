#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

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
echo "mCREAM: Majority (M=10, Medium) - CFMNIST"
echo "=============================================="
echo "HOST=$(hostname)"
"$PYTHON_BIN" -V
nvidia-smi || true

echo ""
echo "Running mCREAM with Majority aggregation (M=10, medium)..."
"$PYTHON_BIN" mcream_main.py --config all_configs/mcream_configs/cfmnist/baselines/majority_M10_medium.yaml

echo "Done!"
