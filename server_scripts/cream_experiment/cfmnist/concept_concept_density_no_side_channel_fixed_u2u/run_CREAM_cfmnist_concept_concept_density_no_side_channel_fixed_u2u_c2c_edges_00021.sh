#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/dani00003/mCREAM"
CONDA_PYTHON="/home/dani00003/miniconda3/envs/mcream/bin/python"
cd "$PROJECT_ROOT"
"$CONDA_PYTHON" simple_main.py --config "all_configs/fixed_u2u/cfmnist_concept_concept_density_no_side_channel_fixed_u2u/CREAM_cfmnist_concept_concept_density_no_side_channel_fixed_u2u_c2c_edges_00021.yaml"
