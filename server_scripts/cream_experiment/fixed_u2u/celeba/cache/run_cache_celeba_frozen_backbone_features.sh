#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/dani00003/mCREAM"
CONDA_PYTHON="/home/dani00003/miniconda3/envs/mcream/bin/python"

if [ ! -x "$CONDA_PYTHON" ]; then
    echo "ERROR: Conda environment not found: $CONDA_PYTHON" >&2
    exit 127
fi
cd "$PROJECT_ROOT"
"$CONDA_PYTHON" -V
nvidia-smi || true
"$CONDA_PYTHON" scripts/cache_frozen_backbone_features.py --config all_configs/sanity_checks/celeba_concept_concept_density_no_side_channel/CREAM_celeba_concept_concept_density_no_side_channel_c2c_edges_00005.yaml --output-dir ./data/fixed_u2u_feature_cache/celeba
