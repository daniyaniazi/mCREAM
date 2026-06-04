#!/usr/bin/env bash
# Submit mCREAM Ensemble experiments with flexible filtering.
#
# PREREQUISITE — run graph generation once before any experiments:
#   condor_submit server_scripts/mcream_experiment/generate_ensemble_graphs_job.sub
#
# USAGE EXAMPLES:
#
#   All experiments, all datasets:
#     ./submit_all_ensemble.sh
#
#   One dataset only:
#     ./submit_all_ensemble.sh --cfmnist-only
#     ./submit_all_ensemble.sh --celeba-only
#
#   One ensemble type only (across all datasets):
#     ./submit_all_ensemble.sh --average
#     ./submit_all_ensemble.sh --weighted
#     ./submit_all_ensemble.sh --soft-average
#     ./submit_all_ensemble.sh --soft-weighted
#
#   Combine dataset + type filters:
#     ./submit_all_ensemble.sh --celeba-only --average
#     ./submit_all_ensemble.sh --celeba-only --weighted
#     ./submit_all_ensemble.sh --celeba-only --soft-average
#     ./submit_all_ensemble.sh --celeba-only --soft-weighted
#     ./submit_all_ensemble.sh --cfmnist-only --soft-weighted

set -euo pipefail
cd ~/mCREAM

# -----------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------
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
echo "mCREAM Ensemble — Submission"
echo "  cfmnist:       $RUN_CFMNIST"
echo "  celeba:        $RUN_CELEBA"
echo "  average:       $RUN_AVERAGE"
echo "  weighted:      $RUN_WEIGHTED"
echo "  soft_average:  $RUN_SOFT_AVERAGE"
echo "  soft_weighted: $RUN_SOFT_WEIGHTED"
echo "=============================================="

# Fix line endings and permissions
find server_scripts/mcream_experiment/cfmnist/ensemble -name "*.sh" -exec sed -i 's/\r$//' {} \;
find server_scripts/mcream_experiment/celeba/ensemble  -name "*.sh" -exec sed -i 's/\r$//' {} \;
chmod +x server_scripts/mcream_experiment/cfmnist/ensemble/**/*.sh
chmod +x server_scripts/mcream_experiment/celeba/ensemble/**/*.sh

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
    local BASE="server_scripts/mcream_experiment/${DATASET}/ensemble"

    echo ""
    echo "=== $DATASET ==="

    for ACTION in $ACTIONS; do
        for LEVEL in $LEVELS; do
            if [ "$RUN_AVERAGE" = true ]; then
                echo "  [average]       ${ACTION}/${LEVEL}"
                submit_job "${BASE}/${ACTION}/average_${ACTION}_${LEVEL}_job.sub"
            fi
            if [ "$RUN_WEIGHTED" = true ]; then
                echo "  [weighted]      ${ACTION}/${LEVEL}"
                submit_job "${BASE}/${ACTION}/weighted_${ACTION}_${LEVEL}_job.sub"
            fi
            if [ "$RUN_SOFT_AVERAGE" = true ]; then
                echo "  [soft_average]  ${ACTION}/${LEVEL}"
                submit_job "${BASE}/${ACTION}/soft_average_${ACTION}_${LEVEL}_job.sub"
            fi
            if [ "$RUN_SOFT_WEIGHTED" = true ]; then
                echo "  [soft_weighted] ${ACTION}/${LEVEL}"
                submit_job "${BASE}/${ACTION}/soft_weighted_${ACTION}_${LEVEL}_job.sub"
            fi
        done
    done
}

[ "$RUN_CFMNIST" = true ] && submit_for_dataset cfmnist
[ "$RUN_CELEBA"  = true ] && submit_for_dataset celeba

echo ""
echo "=============================================="
echo "Submitted $COUNT jobs"
echo "=============================================="
condor_q
