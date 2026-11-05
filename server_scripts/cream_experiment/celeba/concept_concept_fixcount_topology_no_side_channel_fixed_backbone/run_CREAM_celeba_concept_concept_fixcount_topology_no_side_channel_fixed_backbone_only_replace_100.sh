#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/dani00003/mCREAM"
CONDA_PYTHON="/home/dani00003/miniconda3/envs/mcream/bin/python"
cd "$PROJECT_ROOT"
"$CONDA_PYTHON" simple_main.py --config "all_configs/fixed_backbone/celeba_concept_concept_fixcount_topology_no_side_channel_fixed_backbone/CREAM_celeba_concept_concept_fixcount_topology_no_side_channel_fixed_backbone_only_replace_100.yaml"
