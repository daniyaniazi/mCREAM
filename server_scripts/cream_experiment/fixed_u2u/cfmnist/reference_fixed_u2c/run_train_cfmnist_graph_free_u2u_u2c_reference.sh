#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/dani00003/mCREAM"
CONDA_PYTHON="/home/dani00003/miniconda3/envs/mcream/bin/python"

if [ ! -x "$CONDA_PYTHON" ]; then
    echo "ERROR: Conda environment not found: $CONDA_PYTHON" >&2
    exit 127
fi
cd "$PROJECT_ROOT"
"$CONDA_PYTHON" -V
nvidia-smi || true
"$CONDA_PYTHON" simple_main.py --config all_configs/fixed_u2u_fixed_u2c_reference/cfmnist_graph_free_u2u_u2c.yaml
