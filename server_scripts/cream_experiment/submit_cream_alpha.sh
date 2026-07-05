#!/usr/bin/env bash
# Submit CREAM on alpha-recovered graphs (Exp5).
# Reads configs from all_configs/cream_alpha_{dataset}/
#
# PREREQUISITES:
#   python generate_alpha_recovered_graphs.py --dataset cfmnist --threshold 0.5
#
# USAGE:
#   ./submit_cream_alpha.sh                    # cfmnist, all levels
#   ./submit_cream_alpha.sh --dataset celeba   # celeba only
#   ./submit_cream_alpha.sh --dataset cub      # cub only
#   ./submit_cream_alpha.sh --dataset all      # all datasets
#   ./submit_cream_alpha.sh --level low        # low only

set -euo pipefail
cd ~/mCREAM

PYTHON="/home/dani00003/miniconda3/envs/mcream/bin/python"
COUNT=0
DATASET="cfmnist"
LEVELS="low medium high"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --level)
            LEVELS=""
            shift
            while [[ $# -gt 0 ]] && [[ "$1" != --* ]]; do
                LEVELS="$LEVELS $1"; shift
            done
            LEVELS="${LEVELS# }"
            ;;
        *) echo "Unknown: $1" >&2; exit 1 ;;
    esac
done

if [ "$DATASET" = "all" ]; then
    DATASETS="cfmnist celeba cub"
else
    DATASETS="$DATASET"
fi

echo "=============================================="
echo "CREAM on Alpha-Recovered Graphs (Exp5)"
echo "  datasets: $DATASETS  |  levels: $LEVELS"
echo "=============================================="

for DS in $DATASETS; do
    CONFIG_DIR="all_configs/cream_alpha_${DS}"
    if [ ! -d "$CONFIG_DIR" ]; then
        echo "  [SKIP] $CONFIG_DIR not found — run generate_alpha_recovered_graphs.py first"
        continue
    fi
    echo "  Submitting from: $CONFIG_DIR"
    for CONFIG in "$CONFIG_DIR"/*.yaml; do
        [ -f "$CONFIG" ] || continue
        BASE=$(basename "$CONFIG" .yaml)
        # filter by level
        if [ -n "$LEVELS" ]; then
            MATCH=false
            for LV in $LEVELS; do
                [[ "$BASE" == *"_${LV}" ]] && MATCH=true && break
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
    done
done

echo ""
echo "Submitted $COUNT jobs (each runs 5 seeds)"
[ $COUNT -gt 0 ] && condor_q
