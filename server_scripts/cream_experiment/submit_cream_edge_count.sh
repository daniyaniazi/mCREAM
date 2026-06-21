#!/usr/bin/env bash
# Submit CREAM on edge count graphs — cfmnist (u2c only, ±5 from GT=17).
# 11 edge counts × 5 seeds = 55 jobs total.
# Each job = one CREAM on one noisy graph (different random edge selection).
#
# PREREQUISITE: edge count CSV graphs must already exist under
#   data/FashionMNIST/expert_graphs/graph_ensemble_edge_count/u2c_*edges/
#
# USAGE:
#   ./submit_cream_edge_count.sh        # all 55 jobs

set -euo pipefail
cd ~/mCREAM

CONFIG_DIR="all_configs/mcream_configs/cfmnist/edge_count_experiments"
PYTHON="/home/dani00003/miniconda3/envs/mcream/bin/python"
COUNT=0

if [ ! -d "$CONFIG_DIR" ] || [ -z "$(ls $CONFIG_DIR/*.yaml 2>/dev/null)" ]; then
    echo "ERROR: No configs found in $CONFIG_DIR"
    exit 1
fi

echo "=============================================="
echo "CREAM on Edge Count Graphs — cfmnist (u2c ±5)"
echo "  configs: $CONFIG_DIR"
echo "=============================================="

for CONFIG in "$CONFIG_DIR"/edge_count_u2c_*.yaml; do
    [ -f "$CONFIG" ] || continue
    BASE=$(basename "$CONFIG" .yaml)

    echo "universe                = docker
docker_image            = pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
executable              = $PYTHON
arguments               = simple_main.py --config ${CONFIG}
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
echo "Submitted $COUNT jobs"
echo "Results: experiments/.../train_cbm/edge_count_u2c_*/"
condor_q
