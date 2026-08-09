#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="/home/dani00003/mCREAM"
PYTHON_BIN="/home/dani00003/miniconda3/envs/mcream/bin/python"
cd "$PROJECT_ROOT"
echo "HOST=$(hostname)"; nvidia-smi || true
"$PYTHON_BIN" evaluate_metrics.py     --config all_configs/best_hparams/CREAM/CREAM_awa2_soft_seed2_config.yaml
echo "Done!"
