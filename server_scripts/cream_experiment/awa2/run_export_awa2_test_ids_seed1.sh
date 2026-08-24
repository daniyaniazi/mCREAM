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

echo "Exporting AWA2 seed=1 test image IDs..."
"$PYTHON_BIN" scripts/export_test_image_ids.py \
    --config "$CONFIG_PATH" \
    --split test

echo "Done!"
