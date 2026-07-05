#!/usr/bin/env bash
# Exp4: mCREAM Graph Ensemble — consensus expert graphs with shared alpha matrix.
# 3 noise levels (low/medium/high), 5 training seeds each = 3 jobs per dataset.
# Each expert graph has 90% pairwise consensus with others.
# use_alpha=True: shared learnable edge importance matrix trained end-to-end.
#
# PREREQUISITES (run once on server):
#   python generate_consensus_expert_graphs.py --dataset cfmnist --consensus 0.90
#   python generate_consensus_expert_graphs.py --dataset celeba  --consensus 0.90
#
# USAGE:
#   ./submit_graph_ensemble_consensus.sh                          # cfmnist, all levels
#   ./submit_graph_ensemble_consensus.sh --dataset celeba         # celeba, all levels
#   ./submit_graph_ensemble_consensus.sh --dataset all            # both datasets
#   ./submit_graph_ensemble_consensus.sh --level low              # cfmnist, low only
#   ./submit_graph_ensemble_consensus.sh --dataset celeba --level high medium

set -euo pipefail
cd ~/mCREAM

PYTHON="/home/dani00003/miniconda3/envs/mcream/bin/python"
COUNT=0
DATASET="cfmnist"          # default
LEVELS="low medium high"   # default: all three

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --level)
            LEVELS=""
            shift
            while [[ $# -gt 0 ]] && [[ "$1" != --* ]]; do
                LEVELS="$LEVELS $1"; shift
            done
            LEVELS="${LEVELS# }"  # trim leading space
            ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# Resolve datasets list
if [ "$DATASET" = "all" ]; then
    DATASETS="cfmnist celeba cub"
else
    DATASETS="$DATASET"
fi

echo "=============================================="
echo "mCREAM Graph Ensemble — Exp4 Consensus 90%"
echo "  datasets: $DATASETS"
echo "  levels:   $LEVELS"
echo "  use_alpha=True  |  5 seeds per job"
echo "=============================================="

submit_dir() {
    SD_DIR="$1"
    SD_FILTER="$2"   # space-separated levels to include, empty=all
    if [ ! -d "$SD_DIR" ]; then
        echo "  [SKIP] Dir not found: $SD_DIR"; return
    fi
    CONFIG_DIR="$SD_DIR"
    echo "  Submitting from: $CONFIG_DIR"
    for CONFIG in "$CONFIG_DIR"/*.yaml; do
        [ -f "$CONFIG" ] || continue
        BASE=$(basename "$CONFIG" .yaml)
        # Filter by level if specified
        if [ -n "$SD_FILTER" ]; then
            MATCH=false
            for LV in $SD_FILTER; do
                [[ "$BASE" == *"_${LV}" ]] && MATCH=true && break
            done
            [ "$MATCH" = false ] && continue
        fi
        echo "  → $BASE"
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
}

for DS in $DATASETS; do
    submit_dir "all_configs/mcream_graph_ensemble_configs/${DS}/consensus_0.9" "$LEVELS"
done

echo ""
echo "Submitted $COUNT jobs (each runs 5 seeds, saves alpha matrix per seed)"
echo "Results: experiments/.../mCREAM_GraphEnsemble/graph_ensemble_consensus_*/"
echo "Alpha:   experiments/.../seed_*/lightning_logs/version_0/alpha_prob_seed*.csv"
[ $COUNT -gt 0 ] && condor_q
