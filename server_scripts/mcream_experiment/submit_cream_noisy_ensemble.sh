#!/usr/bin/env bash
# Submit standalone CREAM experiments on the same expert graphs used in mCREAM ensemble.
# This gives the FAIR baseline comparison:
#   CREAM (one noisy graph, fully trained)  vs  mCREAM ensemble (M noisy graphs)
#
# PREREQUISITE — generate expert graphs and DAG CSVs first:
#   condor_submit server_scripts/mcream_experiment/generate_ensemble_graphs_job.sub
#   (or just let the run scripts auto-generate on first job)
#
# USAGE:
#   ./submit_cream_noisy_ensemble.sh                        # all
#   ./submit_cream_noisy_ensemble.sh --cfmnist-only
#   ./submit_cream_noisy_ensemble.sh --celeba-only
#   ./submit_cream_noisy_ensemble.sh --deletion-only
#   ./submit_cream_noisy_ensemble.sh --addition-only
#   ./submit_cream_noisy_ensemble.sh --reversal-only
#   ./submit_cream_noisy_ensemble.sh --cfmnist-only --deletion-only

set -euo pipefail
cd ~/mCREAM

RUN_CFMNIST=true
RUN_CELEBA=true
RUN_DELETION=true
RUN_ADDITION=true
RUN_REVERSAL=true

for arg in "$@"; do
    case $arg in
        --cfmnist-only)   RUN_CELEBA=false ;;
        --celeba-only)    RUN_CFMNIST=false ;;
        --deletion-only)  RUN_ADDITION=false; RUN_REVERSAL=false ;;
        --addition-only)  RUN_DELETION=false; RUN_REVERSAL=false ;;
        --reversal-only)  RUN_DELETION=false; RUN_ADDITION=false ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

echo "=============================================="
echo "CREAM Noisy Ensemble Baseline Submission"
echo "  cfmnist:   $RUN_CFMNIST"
echo "  celeba:    $RUN_CELEBA"
echo "  deletion:  $RUN_DELETION"
echo "  addition:  $RUN_ADDITION"
echo "  reversal:  $RUN_REVERSAL"
echo "=============================================="

find server_scripts/mcream_experiment/cfmnist/cream_noisy_ensemble -name "*.sh" -exec sed -i 's/\r$//' {} \;
find server_scripts/mcream_experiment/celeba/cream_noisy_ensemble  -name "*.sh" -exec sed -i 's/\r$//' {} \;
chmod +x server_scripts/mcream_experiment/cfmnist/cream_noisy_ensemble/*.sh
chmod +x server_scripts/mcream_experiment/celeba/cream_noisy_ensemble/*.sh

LEVELS="low medium high"
EXPERTS="0 1 2 3 4"
COUNT=0

submit_for_dataset() {
    local DATASET=$1
    local BASE="server_scripts/mcream_experiment/${DATASET}/cream_noisy_ensemble"

    echo ""
    echo "=== $DATASET ==="

    for LEVEL in $LEVELS; do
        for EXPERT in $EXPERTS; do
            if [ "$RUN_DELETION" = true ]; then
                condor_submit "$BASE/cream_noisy_deletion_${LEVEL}_expert${EXPERT}_job.sub"
                COUNT=$((COUNT + 1))
            fi
            if [ "$RUN_ADDITION" = true ]; then
                condor_submit "$BASE/cream_noisy_addition_${LEVEL}_expert${EXPERT}_job.sub"
                COUNT=$((COUNT + 1))
            fi
            if [ "$RUN_REVERSAL" = true ]; then
                condor_submit "$BASE/cream_noisy_reversal_${LEVEL}_expert${EXPERT}_job.sub"
                COUNT=$((COUNT + 1))
            fi
        done
    done
}

[ "$RUN_CFMNIST" = true ] && submit_for_dataset cfmnist
[ "$RUN_CELEBA"  = true ] && submit_for_dataset celeba

echo ""
echo "=============================================="
echo "Submitted $COUNT jobs"
echo "  Each job = 1 standalone CREAM on 1 expert graph, 5 seeds"
echo "  Results go to experiments/*/train_cbm/mCREAM/"
echo "=============================================="
condor_q
