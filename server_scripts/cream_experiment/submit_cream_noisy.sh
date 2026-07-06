#!/usr/bin/env bash
# Submit CREAM on noisy expert graphs (expert_0) — cfmnist.
# Covers addition/deletion/reversal × low/medium/high (9 jobs, 5 seeds each).
#
# PREREQUISITE (run once on server):
#   python generate_ensemble_expert_graphs.py --dataset cfmnist
#   python generate_single_noisy_dags.py --dataset cfmnist --action addition --level low
#   python generate_single_noisy_dags.py --dataset cfmnist --action addition --level medium
#   python generate_single_noisy_dags.py --dataset cfmnist --action addition --level high
#   python generate_single_noisy_dags.py --dataset cfmnist --action deletion --level low
#   python generate_single_noisy_dags.py --dataset cfmnist --action deletion --level medium
#   python generate_single_noisy_dags.py --dataset cfmnist --action deletion --level high
#   python generate_single_noisy_dags.py --dataset cfmnist --action reversal --level low
#   python generate_single_noisy_dags.py --dataset cfmnist --action reversal --level medium
#   python generate_single_noisy_dags.py --dataset cfmnist --action reversal --level high
#
# USAGE:
#   ./submit_cream_noisy.sh                  # all 9
#   ./submit_cream_noisy.sh --addition-only
#   ./submit_cream_noisy.sh --deletion-only
#   ./submit_cream_noisy.sh --reversal-only

set -euo pipefail
cd ~/mCREAM

RUN_ADDITION=true
RUN_DELETION=true
RUN_REVERSAL=true

for arg in "$@"; do
    case $arg in
        --addition-only) RUN_DELETION=false; RUN_REVERSAL=false ;;
        --deletion-only) RUN_ADDITION=false; RUN_REVERSAL=false ;;
        --reversal-only) RUN_ADDITION=false; RUN_DELETION=false ;;
        *) echo "Unknown: $arg" >&2; exit 1 ;;
    esac
done

CONFIG_DIR="all_configs/cream_noisy_cfmnist"
PYTHON="/home/dani00003/miniconda3/envs/mcream/bin/python"
COUNT=0

echo "=============================================="
echo "CREAM on Noisy Expert Graphs — cfmnist"
echo "  addition: $RUN_ADDITION  deletion: $RUN_DELETION  reversal: $RUN_REVERSAL"
echo "=============================================="

for CONFIG in "$CONFIG_DIR"/*.yaml; do
    [ -f "$CONFIG" ] || continue
    BASE=$(basename "$CONFIG" .yaml)

    if [[ "$BASE" == *"addition"* ]] && [ "$RUN_ADDITION" = false ]; then continue; fi
    if [[ "$BASE" == *"deletion"* ]] && [ "$RUN_DELETION" = false ]; then continue; fi
    if [[ "$BASE" == *"reversal"* ]] && [ "$RUN_REVERSAL" = false ]; then continue; fi

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
echo "Submitted $COUNT jobs (each = 5 seeds)"
echo "Results: experiments/.../train_cbm/cream_noisy_*/"
condor_q
