#!/usr/bin/env bash
# Submit mCREAM Graph Ensemble edge count experiments — CelebA (u2c ±5, GT=10).
# 11 configs (5..15 edges) × 5 seeds per job = 11 jobs total.
#
# PREREQUISITES (run once on server):
#   python generate_edge_count_experiments.py --dataset celeba --delta 5 --n_seeds 5
#   python gen_graph_ensemble_edge_count_configs_celeba.py
#
# USAGE:
#   ./submit_graph_ensemble_edge_count_celeba.sh

set -euo pipefail
cd ~/mCREAM

CONFIG_DIR="all_configs/mcream_graph_ensemble_configs/celeba/edge_count"
PYTHON="/home/dani00003/miniconda3/envs/mcream/bin/python"
COUNT=0

if [ ! -d "$CONFIG_DIR" ] || [ -z "$(ls $CONFIG_DIR/*.yaml 2>/dev/null)" ]; then
    echo "ERROR: No configs found in $CONFIG_DIR"
    echo "Run first:"
    echo "  python generate_edge_count_experiments.py --dataset celeba --delta 5 --n_seeds 5"
    echo "  python gen_graph_ensemble_edge_count_configs_celeba.py"
    exit 1
fi

echo "=============================================="
echo "mCREAM Graph Ensemble — CelebA Edge Count"
echo "  GT=10 u2c edges, range 5..15"
echo "  configs: $CONFIG_DIR"
echo "=============================================="

for CONFIG in "$CONFIG_DIR"/gensemble_edge_count_u2c_*.yaml; do
    [ -f "$CONFIG" ] || continue
    BASE=$(basename "$CONFIG" .yaml)

    echo "universe                = docker
docker_image            = pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
executable              = $PYTHON
arguments               = mcream_graph_ensemble_main.py --config ${CONFIG}
initialdir              = /home/dani00003/mCREAM

output                  = /home/dani00003/mCREAM/logs/${BASE}.\$(ClusterId).\$(ProcId).out
error                   = /home/dani00003/mCREAM/logs/${BASE}.\$(ClusterId).\$(ProcId).err
log                     = /home/dani00003/mCREAM/logs/${BASE}.\$(ClusterId).log

request_GPUs            = 1
request_CPUs            = 8
request_memory          = 32G
requirements            = UidDomain == \"cs.uni-saarland.de\"
+WantGPUHomeMounted     = true
queue 1" | condor_submit

    COUNT=$((COUNT + 1))
done

echo ""
echo "Submitted $COUNT jobs (each runs 5 seeds sequentially)"
echo "Results: experiments/CelebA/train_cbm/mCREAM_GraphEnsemble/gensemble_edge_count_*/"
condor_q
