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

echo "================================================"
echo "mCREAM Ensemble: cfmnist | soft_average | addition | low"
echo "================================================"
echo "HOST=$(hostname)"
"$PYTHON_BIN" -V
nvidia-smi || true
"$PYTHON_BIN" -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available())"

# Ensure expert graphs exist (generates if missing)
EXPERT_DIR="$PROJECT_ROOT/data/FashionMNIST/expert_graphs/ensemble/addition_low"
if [ ! -f "$EXPERT_DIR/config.yaml" ]; then
    echo "Generating expert graphs for addition/low..."
    "$PYTHON_BIN" generate_ensemble_expert_graphs.py --dataset cfmnist --num_experts 5
fi

echo ""
echo "Running mCREAM Ensemble (soft_average, addition, low)..."
"$PYTHON_BIN" mcream_ensemble_main.py     --config all_configs/mcream_ensemble_configs/cfmnist/addition/soft_average_addition_low.yaml

echo "Done!"
