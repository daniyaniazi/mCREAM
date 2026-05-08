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
echo "mCREAM: Intersection Baseline (M=5, Medium)"
echo "=============================================="
echo "HOST=$(hostname)"
"$PYTHON_BIN" -V
nvidia-smi || true

"$PYTHON_BIN" -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available())"
"$PYTHON_BIN" -c "import pytorch_lightning, torchvision, yaml; print('deps_ok=1')"

# Check if expert graphs exist
EXPERT_DIR="$PROJECT_ROOT/data/FashionMNIST/expert_graphs/M5/medium"
if [ ! -d "$EXPERT_DIR" ]; then
    echo "ERROR: Expert graphs not found at $EXPERT_DIR"
    echo "Run generate_expert_graphs_job.sub first!"
    exit 1
fi

echo ""
echo "Running mCREAM with Intersection aggregation..."
"$PYTHON_BIN" mcream_main.py --config all_configs/mcream_configs/cfmnist/baselines/intersection/intersection_M5_medium.yaml

echo "Done!"
