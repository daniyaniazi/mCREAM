#!/usr/bin/env bash
# Experiment 3: mCREAM Graph Ensemble with MIXED noise levels.
# Each expert gets a graph from a different noise level (low/medium/high).
# Lambda should learn: high-noise expert gets higher weight.
# 3 jobs (deletion/addition/reversal).
#
# PREREQUISITE:
#   condor_submit server_scripts/mcream_experiment/generate_ensemble_graphs_job.sub
#   (graphs must exist for all 3 levels of each action)
#
# USAGE:
#   ./submit_graph_ensemble_exp3.sh
#   ./submit_graph_ensemble_exp3.sh --deletion-only
#   ./submit_graph_ensemble_exp3.sh --addition-only
#   ./submit_graph_ensemble_exp3.sh --reversal-only

set -euo pipefail
cd ~/mCREAM

RUN_DELETION=true
RUN_ADDITION=true
RUN_REVERSAL=true

for arg in "$@"; do
    case $arg in
        --deletion-only) RUN_ADDITION=false; RUN_REVERSAL=false ;;
        --addition-only) RUN_DELETION=false; RUN_REVERSAL=false ;;
        --reversal-only) RUN_DELETION=false; RUN_ADDITION=false ;;
        *) echo "Unknown: $arg" >&2; exit 1 ;;
    esac
done

CONFIG_DIR="all_configs/mcream_graph_ensemble_configs/cfmnist/mixed_levels"
PYTHON="/home/dani00003/miniconda3/envs/mcream/bin/python"
COUNT=0

echo "=============================================="
echo "Graph Ensemble Exp3: Mixed Noise Levels"
echo "  deletion: $RUN_DELETION"
echo "  addition: $RUN_ADDITION"
echo "  reversal: $RUN_REVERSAL"
echo "  Expert assignment: E0=low, E1=medium, E2=high, E3=low, E4=high"
echo "  Expected: lambda_2 > lambda_1 > lambda_0"
echo "=============================================="

# Fix line endings
sed -i 's/\r$//' "$CONFIG_DIR"/*.yaml 2>/dev/null || true

for ACTION in deletion addition reversal; do
    CONFIG="$CONFIG_DIR/graph_ensemble_mixed_${ACTION}.yaml"
    [ -f "$CONFIG" ] || continue

    case $ACTION in
        deletion) [ "$RUN_DELETION" = false ] && continue ;;
        addition) [ "$RUN_ADDITION" = false ] && continue ;;
        reversal) [ "$RUN_REVERSAL" = false ] && continue ;;
    esac

    BASE="graph_ensemble_mixed_${ACTION}"

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

    echo "  Submitted: $ACTION (seeds: [42,7,1,134,89])"
    COUNT=$((COUNT + 1))
done

echo ""
echo "Submitted $COUNT Exp3 jobs"
echo "Results: experiments/.../mCREAM_GraphEnsemble/graph_ensemble_mixed_*/"
echo "Key metric to check: lambda_0..lambda_4 in results CSV"
condor_q
