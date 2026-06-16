#!/usr/bin/env bash
# CREAM with ground truth DAG, NO side channel — CelebA
set -euo pipefail
PROJECT_ROOT="/home/dani00003/mCREAM"
CONDA_PYTHON="/home/dani00003/miniconda3/envs/mcream/bin/python"
if [ -x "$CONDA_PYTHON" ]; then PYTHON_BIN="$CONDA_PYTHON"
else echo "ERROR: Conda env not found" >&2; exit 127; fi
cd "$PROJECT_ROOT"
echo "HOST=$(hostname)"; "$PYTHON_BIN" -V; nvidia-smi || true
echo "Running CREAM no-side-channel (celeba)..."
"$PYTHON_BIN" simple_main.py \
    --config all_configs/best_hparams/CREAM_no_side_channel/CREAM_no_side_best_celeba.yaml
echo "Done!"
