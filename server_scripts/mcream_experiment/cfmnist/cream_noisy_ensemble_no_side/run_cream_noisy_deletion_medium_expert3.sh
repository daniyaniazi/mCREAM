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


echo "Running standalone CREAM on deletion/medium/expert3..."
"$PYTHON_BIN" simple_main.py --config all_configs/mcream_configs/cfmnist/cream_noisy_ensemble_no_side/cream_noisy_deletion_medium_expert3_no_side.yaml
echo "Done!"
