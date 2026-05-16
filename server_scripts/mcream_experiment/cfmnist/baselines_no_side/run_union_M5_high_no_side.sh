#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/dani00003/mCREAM"
CONDA_PYTHON="/home/dani00003/miniconda3/envs/mcream/bin/python"

if [ -x "$CONDA_PYTHON" ]; then
    PYTHON_BIN="$CONDA_PYTHON"
else
    echo "ERROR: Conda env not found" >&2
    exit 127
fi

cd "$PROJECT_ROOT"
echo "HOST=$(hostname)"
"$PYTHON_BIN" -V
nvidia-smi || true
"$PYTHON_BIN" -c "import torch; print(torch.__version__, torch.cuda.is_available())"

echo Running: union_M5_high_no_side
"$PYTHON_BIN" mcream_main.py --config all_configs/mcream_configs/cfmnist/baselines_no_side/union_M5_high_no_side.yaml
echo Done!