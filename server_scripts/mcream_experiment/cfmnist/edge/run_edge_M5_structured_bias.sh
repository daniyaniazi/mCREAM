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

echo "=============================================="
echo "mCREAM: Edge-level with Structured Expert Bias"
echo "Experts: 2 conservative, 2 liberal, 1 balanced"
echo "=============================================="
echo "HOST=$(hostname)"
"$PYTHON_BIN" -V
nvidia-smi || true

"$PYTHON_BIN" -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available())"

echo ""
echo "Running mCREAM with Edge aggregation (structured bias)..."
"$PYTHON_BIN" mcream_main.py --config all_configs/mcream_configs/cfmnist/edge/edge_M5_structured_bias.yaml

echo "Done!"
