#!/usr/bin/env bash
# mCREAM Graph Ensemble — Consensus expert graphs with shared alpha matrix.
#
# Modes:
#   static   (Exp4): fixed consensus graphs, 3 levels
#   dynamic  (Exp5): graphs refreshed every N epochs, 3 levels
#   hparam   (Exp5-hparam): dynamic + grid over epochs/lambda/loss_type
#
# PREREQUISITES (run once on server):
#   python generate_consensus_expert_graphs.py --dataset cfmnist --consensus 0.90 --force
#   python generate_consensus_expert_graphs.py --dataset celeba  --consensus 0.90 --force
#   python generate_consensus_expert_graphs.py --dataset cub     --consensus 0.90 --force
#   (--force needed because BASE_NOISE changed to low=0.10 medium=0.15 high=0.35)
#
# USAGE:
#   ./submit_graph_ensemble_consensus.sh                          # cfmnist, static, all levels
#   ./submit_graph_ensemble_consensus.sh --dynamic                # cfmnist, dynamic, all levels
#   ./submit_graph_ensemble_consensus.sh --hparam                 # cfmnist, dynamic hparam grid
#   ./submit_graph_ensemble_consensus.sh --hparam --dataset all   # all datasets, hparam grid
#   ./submit_graph_ensemble_consensus.sh --dataset celeba         # celeba, static
#   ./submit_graph_ensemble_consensus.sh --dataset all            # all datasets, static
#   ./submit_graph_ensemble_consensus.sh --level low              # low noise only
#   ./submit_graph_ensemble_consensus.sh --hparam --level high    # hparam grid, high only
#   ./submit_graph_ensemble_consensus.sh --hparam --epochs 300    # hparam grid, 300 epochs only
#   ./submit_graph_ensemble_consensus.sh --hparam --loss orig     # hparam grid, per_expert loss only

set -euo pipefail
cd ~/mCREAM

PYTHON="/home/dani00003/miniconda3/envs/mcream/bin/python"
COUNT=0
DATASET="cfmnist"
LEVELS=""          # empty = all
DYNAMIC=false
HPARAM=false
UNIFORM_LAMBDA=false
FILTER_EPOCHS=""   # empty = all
FILTER_LOSS=""     # empty = all

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --dynamic) DYNAMIC=true; shift ;;
        --hparam)  HPARAM=true; DYNAMIC=true; shift ;;
        --level)
            shift
            while [[ $# -gt 0 ]] && [[ "$1" != --* ]]; do
                LEVELS="$LEVELS $1"; shift
            done
            LEVELS="${LEVELS# }"
            ;;
        --epochs)
            shift
            while [[ $# -gt 0 ]] && [[ "$1" != --* ]]; do
                FILTER_EPOCHS="$FILTER_EPOCHS $1"; shift
            done
            FILTER_EPOCHS="${FILTER_EPOCHS# }"
            ;;
        --loss)
            shift
            while [[ $# -gt 0 ]] && [[ "$1" != --* ]]; do
                FILTER_LOSS="$FILTER_LOSS $1"; shift
            done
            FILTER_LOSS="${FILTER_LOSS# }"
            ;;
        --uniform-lambda) UNIFORM_LAMBDA=true; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# Resolve datasets
if [ "$DATASET" = "all" ]; then
    DATASETS="cfmnist celeba cub"
else
    DATASETS="$DATASET"
fi

echo "=============================================="
echo "mCREAM Graph Ensemble — Consensus 90%"
echo "  datasets: $DATASETS"
echo "  levels:   ${LEVELS:-all}"
echo "  mode:     $([ "$HPARAM" = true ] && echo "hparam-grid" || ([ "$DYNAMIC" = true ] && echo "dynamic" || echo "static"))"
[ -n "$FILTER_EPOCHS" ] && echo "  epochs filter: $FILTER_EPOCHS"
[ -n "$FILTER_LOSS"   ] && echo "  loss filter:   $FILTER_LOSS"
echo "  use_alpha=True  |  5 seeds per job"
echo "=============================================="

# ── submit_dir: submit all matching .yaml files from a config directory ───────
submit_dir() {
    local DIR="$1"
    if [ ! -d "$DIR" ]; then
        echo "  [SKIP] Dir not found: $DIR"; return
    fi
    echo "  Submitting from: $DIR"
    local submitted=0
    for CONFIG in "$DIR"/*.yaml; do
        [ -f "$CONFIG" ] || continue
        BASE=$(basename "$CONFIG" .yaml)

        # ── Level filter ──────────────────────────────────────────────────────
        if [ -n "$LEVELS" ]; then
            MATCH=false
            for LV in $LEVELS; do
                # match _low / _low_ (for hparam names like ..._low_ep300_...)
                [[ "$BASE" == *"_${LV}_"* ]] && MATCH=true && break
                [[ "$BASE" == *"_${LV}" ]]   && MATCH=true && break
            done
            [ "$MATCH" = false ] && continue
        fi

        # ── Epochs filter (hparam mode only) ─────────────────────────────────
        if [ -n "$FILTER_EPOCHS" ]; then
            MATCH=false
            for EP in $FILTER_EPOCHS; do
                [[ "$BASE" == *"_ep${EP}_"* ]] && MATCH=true && break
            done
            [ "$MATCH" = false ] && continue
        fi

        # ── Loss type filter (hparam mode only) ───────────────────────────────
        if [ -n "$FILTER_LOSS" ]; then
            MATCH=false
            for LS in $FILTER_LOSS; do
                [[ "$BASE" == *"_${LS}" ]] && MATCH=true && break
            done
            [ "$MATCH" = false ] && continue
        fi

        # For uniform lambda: patch config on the fly — flip flag and prefix experiment_name
        SUBMIT_CONFIG="$CONFIG"
        if [ "$UNIFORM_LAMBDA" = true ]; then
            UNIFORM_BASE="uniform_lambda_${BASE}"
            TMP_CONFIG="$(dirname "$CONFIG")/${UNIFORM_BASE}.yaml"
            sed \
                -e "s/uniform_lambda: false/uniform_lambda: true/" \
                -e "s/experiment_name: ${BASE}/experiment_name: uniform_lambda_${BASE}/" \
                "$CONFIG" > "$TMP_CONFIG"
            SUBMIT_CONFIG="$TMP_CONFIG"
            BASE="$UNIFORM_BASE"
        fi

        echo "  → $BASE"
        echo "universe                = docker
docker_image            = pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
executable              = $PYTHON
arguments               = mcream_graph_ensemble_main.py --config ${SUBMIT_CONFIG}
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
}

# ── Main loop ─────────────────────────────────────────────────────────────────
# --uniform-lambda reuses consensus_0.9 configs but patches them on the fly
for DS in $DATASETS; do
    if [ "$HPARAM" = true ]; then
        submit_dir "all_configs/mcream_graph_ensemble_configs/${DS}/consensus_dynamic_hparam"
    elif [ "$DYNAMIC" = true ]; then
        submit_dir "all_configs/mcream_graph_ensemble_configs/${DS}/consensus_dynamic"
    else
        submit_dir "all_configs/mcream_graph_ensemble_configs/${DS}/consensus_0.9"
    fi
done

echo ""
echo "Submitted $COUNT jobs total (each runs 5 seeds)"
echo "Results: experiments/.../mCREAM_GraphEnsemble/graph_ensemble_consensus_dynamic_*/"
echo "Alpha:   experiments/.../seed_*/lightning_logs/version_0/alpha_prob_seed*.csv"
[ $COUNT -gt 0 ] && condor_q
