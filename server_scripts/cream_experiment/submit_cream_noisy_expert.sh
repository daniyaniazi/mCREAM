#!/usr/bin/env bash
# CREAM on individual noisy expert graphs (consensus 90%).
# Runs simple_main.py — one CREAM model per expert graph.
#
# USAGE:
#   ./submit_cream_noisy_expert.sh                        # cub, all levels, all experts
#   ./submit_cream_noisy_expert.sh --dataset celeba       # celeba
#   ./submit_cream_noisy_expert.sh --dataset all          # cub + celeba + cfmnist
#   ./submit_cream_noisy_expert.sh --level low            # low noise only
#   ./submit_cream_noisy_expert.sh --dataset cub --level high

set -euo pipefail
cd ~/mCREAM

PYTHON="/home/dani00003/miniconda3/envs/mcream/bin/python"
COUNT=0
DATASET="cub"
LEVELS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --level)
            shift
            while [[ $# -gt 0 ]] && [[ "$1" != --* ]]; do
                LEVELS="$LEVELS $1"; shift
            done
            LEVELS="${LEVELS# }"
            ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [ "$DATASET" = "all" ]; then
    DATASETS="cub celeba cfmnist"
else
    DATASETS="$DATASET"
fi

echo "=============================================="
echo "CREAM on noisy expert graphs (consensus 90%)"
echo "  datasets: $DATASETS"
echo "  levels:   ${LEVELS:-all}"
echo "  script:   simple_main.py"
echo "=============================================="

for DS in $DATASETS; do
    DIR="all_configs/mcream_graph_ensemble_configs/${DS}/cream_noisy_expert"
    if [ ! -d "$DIR" ]; then echo "  [SKIP] $DIR not found"; continue; fi
    echo "  Submitting from: $DIR"
    submitted=0
    for CONFIG in "$DIR"/*.yaml; do
        [ -f "$CONFIG" ] || continue
        BASE=$(basename "$CONFIG" .yaml)

        if [ -n "$LEVELS" ]; then
            MATCH=false
            for LV in $LEVELS; do
                [[ "$BASE" == *"_${LV}_"* ]] && MATCH=true && break
            done
            [ "$MATCH" = false ] && continue
        fi

        echo "  → $BASE"
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
        submitted=$((submitted + 1))
    done
    echo "    submitted $submitted jobs from $DIR"
done

echo ""
echo "Submitted $COUNT jobs total"
echo "Results: experiments/{CUB,CelebA}/train_cbm/Standard_{CUB,CelebA}/CREAM/cream_with_consensus_noisy_expert_*/"
[ $COUNT -gt 0 ] && condor_q
