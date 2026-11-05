#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/dani00003/mCREAM"
CONDA_PYTHON="/home/dani00003/miniconda3/envs/mcream/bin/python"
cd "$PROJECT_ROOT"
"$CONDA_PYTHON" simple_main.py --config "all_configs/fixed_u2u_fixed_u2c/cub_density_random_full_no_side_channel_fixed_u2u_fixed_u2c/CREAM_cub_density_random_full_no_side_channel_edges_37610.yaml"
