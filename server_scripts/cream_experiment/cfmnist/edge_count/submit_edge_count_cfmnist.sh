#!/usr/bin/env bash
# Submit all CREAM edge-count + single-edge perturbation jobs for cfmnist.
#
# STEP 1 (run once, no GPU):
#   condor_submit server_scripts/cream_experiment/cfmnist/edge_count/generate_edge_count_cfmnist_job.sub
#   Wait for completion.
#
# STEP 2 (run this script):
#   ./server_scripts/cream_experiment/cfmnist/edge_count/submit_edge_count_cfmnist.sh
#
# Usage:
#   ./submit_edge_count_cfmnist.sh              # both scenarios
#   ./submit_edge_count_cfmnist.sh --count-only # only edge count experiments
#   ./submit_edge_count_cfmnist.sh --perturb-only # only single-edge perturbation

set -euo pipefail
cd ~/mCREAM

RUN_COUNT=true
RUN_PERTURB=true

for arg in "$@"; do
    case $arg in
        --count-only)   RUN_PERTURB=false ;;
        --perturb-only) RUN_COUNT=false ;;
        *) echo "Unknown: $arg" >&2; exit 1 ;;
    esac
done

COUNT=0

run_job() {
    local CONFIG=$1
    local EXP_NAME
    EXP_NAME=$(basename "$CONFIG" .yaml)

    # Generate a .sub file on the fly and submit
    SUB_CONTENT="universe                = docker
docker_image            = pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
executable              = /home/dani00003/miniconda3/envs/mcream/bin/python
arguments               = simple_main.py --config ${CONFIG}
initialdir              = /home/dani00003/mCREAM

output                  = /home/dani00003/mCREAM/logs/${EXP_NAME}.\$(ClusterId).\$(ProcId).out
error                   = /home/dani00003/mCREAM/logs/${EXP_NAME}.\$(ClusterId).\$(ProcId).err
log                     = /home/dani00003/mCREAM/logs/${EXP_NAME}.\$(ClusterId).log

request_GPUs            = 1
request_CPUs            = 8
request_memory          = 32G
requirements            = UidDomain == \"cs.uni-saarland.de\"
+WantGPUHomeMounted     = true
queue 1"

    echo "$SUB_CONTENT" | condor_submit
    COUNT=$((COUNT + 1))
}

echo "=============================================="
echo "Submitting cfmnist edge perturbation jobs"
echo "  edge_count:  $RUN_COUNT"
echo "  single_edge: $RUN_PERTURB"
echo "=============================================="

if [ "$RUN_COUNT" = true ]; then
    echo ""
    echo "=== Scenario 1: Edge Count experiments ==="
    for CONFIG in all_configs/mcream_configs/cfmnist/edge_count_experiments/*.yaml; do
        [ -f "$CONFIG" ] || continue
        echo "  Submitting $CONFIG"
        run_job "$CONFIG"
    done
fi

if [ "$RUN_PERTURB" = true ]; then
    echo ""
    echo "=== Scenario 2: Single-Edge Perturbation ==="
    for CONFIG in all_configs/mcream_configs/cfmnist/single_edge_perturbation/*.yaml; do
        [ -f "$CONFIG" ] || continue
        echo "  Submitting $CONFIG"
        run_job "$CONFIG"
    done
fi

echo ""
echo "=============================================="
echo "Submitted $COUNT jobs total"
echo "=============================================="
condor_q
