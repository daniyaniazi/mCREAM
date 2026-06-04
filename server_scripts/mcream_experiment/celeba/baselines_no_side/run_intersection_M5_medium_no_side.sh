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
echo "mCREAM: Intersection Baseline (M=5, Medium) - CelebA - NO SIDE CHANNEL"
echo "=============================================="
echo "HOST=$(hostname)"
"$PYTHON_BIN" -V
nvidia-smi || true

"$PYTHON_BIN" -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available())"

EXPERT_DIR="$PROJECT_ROOT/data/CelebA/expert_graphs/M5/medium"
if [ ! -d "$EXPERT_DIR" ]; then
    echo "ERROR: Expert graphs not found at $EXPERT_DIR"
    exit 1
fi

echo ""
echo "Running mCREAM with Intersection aggregation (no side channel)..."
"$PYTHON_BIN" mcream_main.py --config all_configs/mcream_configs/celeba/baselines_no_side/intersection_M5_medium_no_side.yaml

echo "Done!"
