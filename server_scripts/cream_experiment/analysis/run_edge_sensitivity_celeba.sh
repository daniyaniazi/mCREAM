#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="/home/dani00003/mCREAM"
CONDA_PYTHON="/home/dani00003/miniconda3/envs/mcream/bin/python"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$CONDA_PYTHON" scripts/edge_sensitivity_shapley.py --analysis-config all_configs/analysis/edge_sensitivity/celeba_replace_000_vs_replace_100.yaml
