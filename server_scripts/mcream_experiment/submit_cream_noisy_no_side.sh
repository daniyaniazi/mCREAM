#!/usr/bin/env bash
# Submit all CREAM noisy no-side-channel experiments.
# Same flags as submit_cream_noisy_ensemble.sh
#
# USAGE:
#   ./submit_cream_noisy_no_side.sh
#   ./submit_cream_noisy_no_side.sh --cfmnist-only
#   ./submit_cream_noisy_no_side.sh --celeba-only
#   ./submit_cream_noisy_no_side.sh --deletion-only
#   ./submit_cream_noisy_no_side.sh --cfmnist-only --deletion-only

set -euo pipefail
cd ~/mCREAM

RUN_CFMNIST=true
RUN_CELEBA=true
RUN_DELETION=true
RUN_ADDITION=true
RUN_REVERSAL=true

for arg in "$@"; do
    case $arg in
        --cfmnist-only)  RUN_CELEBA=false ;;
        --celeba-only)   RUN_CFMNIST=false ;;
        --deletion-only) RUN_ADDITION=false; RUN_REVERSAL=false ;;
        --addition-only) RUN_DELETION=false; RUN_REVERSAL=false ;;
        --reversal-only) RUN_DELETION=false; RUN_ADDITION=false ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

echo "=============================================="
echo "CREAM Noisy NO SIDE CHANNEL — Submission"
echo "  cfmnist:   $RUN_CFMNIST"
echo "  celeba:    $RUN_CELEBA"
echo "  deletion:  $RUN_DELETION"
echo "  addition:  $RUN_ADDITION"
echo "  reversal:  $RUN_REVERSAL"
echo "=============================================="

fix_and_chmod() {
    find "$1" -name "*.sh" -exec sed -i 's/\r$//' {} \;
    chmod +x "$1"/*.sh 2>/dev/null || true
}

fix_and_chmod server_scripts/mcream_experiment/cfmnist/cream_noisy_ensemble_no_side
fix_and_chmod server_scripts/mcream_experiment/celeba/cream_noisy_ensemble_no_side

LEVELS="low medium high"
EXPERTS="0 1 2 3 4"
COUNT=0

submit_for_dataset() {
    local DATASET=$1
    local BASE="server_scripts/mcream_experiment/${DATASET}/cream_noisy_ensemble_no_side"

    echo ""
    echo "=== $DATASET (noisy no-side) ==="

    for LEVEL in $LEVELS; do
        for EXPERT in $EXPERTS; do
            [ "$RUN_DELETION" = true ] && condor_submit "$BASE/cream_noisy_deletion_${LEVEL}_expert${EXPERT}_no_side_job.sub" && COUNT=$((COUNT+1))
            [ "$RUN_ADDITION" = true ] && condor_submit "$BASE/cream_noisy_addition_${LEVEL}_expert${EXPERT}_no_side_job.sub" && COUNT=$((COUNT+1))
            [ "$RUN_REVERSAL" = true ] && condor_submit "$BASE/cream_noisy_reversal_${LEVEL}_expert${EXPERT}_no_side_job.sub" && COUNT=$((COUNT+1))
        done
    done
}

[ "$RUN_CFMNIST" = true ] && submit_for_dataset cfmnist
[ "$RUN_CELEBA"  = true ] && submit_for_dataset celeba

echo ""
echo "Submitted $COUNT noisy no-side jobs"
condor_q
