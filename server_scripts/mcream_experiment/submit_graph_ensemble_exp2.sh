#!/usr/bin/env bash
# Experiment 2: mCREAM Graph Ensemble on single-edge perturbation graphs.
# Each experiment: all 5 experts share the SAME graph with 1 edge changed.
# 39 jobs total (29 deletions + 10 additions).
#
# PREREQUISITE (run once):
#   python generate_single_edge_perturbation.py --dataset cfmnist --n_additions 10
#   python generate_graph_ensemble_single_edge_configs.py --dataset cfmnist
#
# USAGE:
#   ./submit_graph_ensemble_exp2.sh
#   ./submit_graph_ensemble_exp2.sh --del-only
#   ./submit_graph_ensemble_exp2.sh --add-only

set -euo pipefail
cd ~/mCREAM

RUN_DEL=true
RUN_ADD=true

for arg in "$@"; do
    case $arg in
        --del-only) RUN_ADD=false ;;
        --add-only) RUN_DEL=false ;;
        *) echo "Unknown: $arg" >&2; exit 1 ;;
    esac
done

CONFIG_DIR="all_configs/mcream_graph_ensemble_configs/cfmnist/single_edge"
PYTHON="/home/dani00003/miniconda3/envs/mcream/bin/python"
COUNT=0

# Check configs exist
if [ ! -d "$CONFIG_DIR" ] || [ -z "$(ls $CONFIG_DIR/*.yaml 2>/dev/null)" ]; then
    echo "ERROR: No configs found in $CONFIG_DIR"
    echo "Run first:"
    echo "  python generate_single_edge_perturbation.py --dataset cfmnist"
    echo "  python generate_graph_ensemble_single_edge_configs.py --dataset cfmnist"
    exit 1
fi

echo "=============================================="
echo "Graph Ensemble Exp2: Single-Edge Perturbation"
echo "  deletions: $RUN_DEL  |  additions: $RUN_ADD"
echo "=============================================="

for CONFIG in "$CONFIG_DIR"/gensingle_*.yaml; do
    [ -f "$CONFIG" ] || continue
    BASE=$(basename "$CONFIG" .yaml)

    if [[ "$BASE" == *"del_edge"* ]] && [ "$RUN_DEL" = false ]; then continue; fi
    if [[ "$BASE" == *"add_edge"* ]] && [ "$RUN_ADD" = false ]; then continue; fi

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
echo "Submitted $COUNT Exp2 jobs"
echo "Results: experiments/.../mCREAM_GraphEnsemble/gensingle_*/"
condor_q
