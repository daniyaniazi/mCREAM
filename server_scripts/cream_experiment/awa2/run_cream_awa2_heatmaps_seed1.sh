#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/dani00003/mCREAM"
CONDA_PYTHON="/home/dani00003/miniconda3/envs/mcream/bin/python"
CONFIG_PATH="all_configs/best_hparams/CREAM/CREAM_awa2_soft_seed1_config.yaml"

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

echo "Generating CREAM AWA2 seed=1 heatmaps for 5 test samples..."
"$PYTHON_BIN" evaluate_metrics.py \
    --config "$CONFIG_PATH" \
    --heatmap_images 5 \
    --only_heatmaps

echo "Done!"
