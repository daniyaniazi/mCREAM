#!/usr/bin/env bash
# Submit mCREAM Ensemble NO SIDE CHANNEL experiments.
# Same usage flags as submit_all_ensemble.sh
#
# PREREQUISITE:
#   condor_submit server_scripts/mcream_experiment/generate_ensemble_graphs_job.sub
#
# USAGE:
#   ./submit_all_ensemble_no_side.sh
#   ./submit_all_ensemble_no_side.sh --cfmnist-only
#   ./submit_all_ensemble_no_side.sh --celeba-only
#   ./submit_all_ensemble_no_side.sh --average
#   ./submit_all_ensemble_no_side.sh --weighted
#   ./submit_all_ensemble_no_side.sh --soft-average
#   ./submit_all_ensemble_no_side.sh --soft-weighted
#   ./submit_all_ensemble_no_side.sh --celeba-only --weighted

set -euo pipefail
cd ~/mCREAM

RUN_CFMNIST=true
RUN_CELEBA=true
RUN_AVERAGE=true
RUN_WEIGHTED=true
RUN_SOFT_AVERAGE=true
RUN_SOFT_WEIGHTED=true

for arg in "$@"; do
    case $arg in
        --cfmnist-only)   RUN_CELEBA=false ;;
        --celeba-only)    RUN_CFMNIST=false ;;
        --average)        RUN_WEIGHTED=false; RUN_SOFT_AVERAGE=false; RUN_SOFT_WEIGHTED=false ;;
        --weighted)       RUN_AVERAGE=false;  RUN_SOFT_AVERAGE=false; RUN_SOFT_WEIGHTED=false ;;
        --soft-average)   RUN_AVERAGE=false;  RUN_WEIGHTED=false;     RUN_SOFT_WEIGHTED=false ;;
        --soft-weighted)  RUN_AVERAGE=false;  RUN_WEIGHTED=false;     RUN_SOFT_AVERAGE=false ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

echo "=============================================="
echo "mCREAM Ensemble NO SIDE CHANNEL — Submission"
echo "  cfmnist:      $RUN_CFMNIST"
echo "  celeba:       $RUN_CELEBA"
echo "  average:      $RUN_AVERAGE"
echo "  weighted:     $RUN_WEIGHTED"
echo "  soft_average: $RUN_SOFT_AVERAGE"
echo "  soft_weighted:$RUN_SOFT_WEIGHTED"
echo "=============================================="

find server_scripts/mcream_experiment/cfmnist/ensemble/no_side -name "*.sh" -exec sed -i 's/\r$//' {} \;
find server_scripts/mcream_experiment/celeba/ensemble/no_side  -name "*.sh" -exec sed -i 's/\r$//' {} \;
chmod +x server_scripts/mcream_experiment/cfmnist/ensemble/no_side/*.sh
chmod +x server_scripts/mcream_experiment/celeba/ensemble/no_side/*.sh

ACTIONS="deletion addition reversal"
LEVELS="low medium high"
COUNT=0

submit_job() {
    local JOB=$1
    if [ -f "$JOB" ]; then
        condor_submit "$JOB"
        COUNT=$((COUNT + 1))
    else
        echo "  WARNING: $JOB not found" >&2
    fi
}

submit_for_dataset() {
    local DATASET=$1
    local BASE="server_scripts/mcream_experiment/${DATASET}/ensemble/no_side"

    echo ""
    echo "=== $DATASET (no side channel) ==="

    for ACTION in $ACTIONS; do
        for LEVEL in $LEVELS; do
            if [ "$RUN_AVERAGE" = true ]; then
                submit_job "${BASE}/no_side_average_${ACTION}_${LEVEL}_job.sub"
            fi
            if [ "$RUN_WEIGHTED" = true ]; then
                submit_job "${BASE}/no_side_weighted_${ACTION}_${LEVEL}_job.sub"
            fi
            if [ "$RUN_SOFT_AVERAGE" = true ]; then
                submit_job "${BASE}/no_side_soft_average_${ACTION}_${LEVEL}_job.sub"
            fi
            if [ "$RUN_SOFT_WEIGHTED" = true ]; then
                submit_job "${BASE}/no_side_soft_weighted_${ACTION}_${LEVEL}_job.sub"
            fi
        done
    done
}

[ "$RUN_CFMNIST" = true ] && submit_for_dataset cfmnist
[ "$RUN_CELEBA"  = true ] && submit_for_dataset celeba

echo ""
echo "=============================================="
echo "Submitted $COUNT no-side-channel ensemble jobs"
echo "=============================================="
condor_q
