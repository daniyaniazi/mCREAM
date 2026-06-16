#!/usr/bin/env bash
# Submit mCREAM Graph Ensemble edge count experiments.
# Mirrors edge_perturbation_analysis (scenario 1) but with GraphEnsemble model.
#
# PREREQUISITE:
#   python generate_edge_count_experiments.py --dataset cfmnist --delta 5
#   (graphs already exist if you ran CREAM edge count experiments)
#
# USAGE:
#   ./submit_graph_ensemble_edge_count.sh              # all (165 jobs)
#   ./submit_graph_ensemble_edge_count.sh --u2c-only   # only u2c (55 jobs)
#   ./submit_graph_ensemble_edge_count.sh --c2y-only   # only c2y (110 jobs)

set -euo pipefail
cd ~/mCREAM

RUN_U2C=true
RUN_C2Y=true

for arg in "$@"; do
    case $arg in
        --u2c-only) RUN_C2Y=false ;;
        --c2y-only) RUN_U2C=false ;;
        *) echo "Unknown: $arg" >&2; exit 1 ;;
    esac
done

CONFIG_DIR="all_configs/mcream_graph_ensemble_configs/cfmnist/edge_count"
PYTHON="/home/dani00003/miniconda3/envs/mcream/bin/python"
COUNT=0

echo "=============================================="
echo "mCREAM Graph Ensemble — Edge Count Experiments"
echo "  u2c: $RUN_U2C  |  c2y: $RUN_C2Y"
echo "=============================================="

for CONFIG in "$CONFIG_DIR"/gensemble_edge_count_*.yaml; do
    [ -f "$CONFIG" ] || continue
    BASE=$(basename "$CONFIG" .yaml)

    # Filter by type
    if [[ "$BASE" == *"_u2c_"* ]] && [ "$RUN_U2C" = false ]; then continue; fi
    if [[ "$BASE" == *"_c2y_"* ]] && [ "$RUN_C2Y" = false ]; then continue; fi

    EXP_NAME="${BASE}"

    echo "universe                = docker
docker_image            = pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
executable              = $PYTHON
arguments               = mcream_graph_ensemble_main.py --config ${CONFIG}
initialdir              = /home/dani00003/mCREAM

output                  = /home/dani00003/mCREAM/logs/${EXP_NAME}.\$(ClusterId).\$(ProcId).out
error                   = /home/dani00003/mCREAM/logs/${EXP_NAME}.\$(ClusterId).\$(ProcId).err
log                     = /home/dani00003/mCREAM/logs/${EXP_NAME}.\$(ClusterId).log

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
condor_q
