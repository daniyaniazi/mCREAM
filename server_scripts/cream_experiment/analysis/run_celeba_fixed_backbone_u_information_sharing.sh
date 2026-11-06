#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/dani00003/mCREAM"
CONDA_PYTHON="/home/dani00003/miniconda3/envs/mcream/bin/python"
ANALYSIS_CONFIG="all_configs/analysis/u_information_sharing/celeba_fixed_backbone_replace_000_vs_replace_100.yaml"

cd "$PROJECT_ROOT"
"$CONDA_PYTHON" scripts/analyze_u_information_sharing.py --analysis-config "$ANALYSIS_CONFIG"
