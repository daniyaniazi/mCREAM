#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/dani00003/mCREAM"
CONDA_PYTHON="/home/dani00003/miniconda3/envs/mcream/bin/python"
CONFIG_PATH="all_configs/sanity_checks/cub_progressive_noise/CREAM_cub_progressive_noise_100.yaml"

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
"$PYTHON_BIN" -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available())"
"$PYTHON_BIN" -c "import pytorch_lightning, torchvision, yaml; print('deps_ok=1')"

echo "Running CUB CREAM progressive graph noise: 100%"
"$PYTHON_BIN" simple_main.py --config "$CONFIG_PATH"

echo "Done!"
