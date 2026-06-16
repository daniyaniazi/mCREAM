#!/usr/bin/env bash
# Submit mCREAM Graph Module Ensemble experiments for cfmnist.
# Reuses same expert graphs as mcream_ensemble experiments.
#
# PREREQUISITE:
#   condor_submit server_scripts/mcream_experiment/generate_ensemble_graphs_job.sub
#
# USAGE:
#   ./submit_graph_ensemble.sh
#   ./submit_graph_ensemble.sh --deletion-only
#   ./submit_graph_ensemble.sh --addition-only
#   ./submit_graph_ensemble.sh --reversal-only

set -euo pipefail
cd ~/mCREAM

RUN_DELETION=true
RUN_ADDITION=true
RUN_REVERSAL=true

for arg in "$@"; do
    case $arg in
        --deletion-only) RUN_ADDITION=false; RUN_REVERSAL=false ;;
        --addition-only) RUN_DELETION=false; RUN_REVERSAL=false ;;
        --reversal-only) RUN_DELETION=false; RUN_ADDITION=false ;;
        *) echo "Unknown: $arg" >&2; exit 1 ;;
    esac
done

echo "=============================================="
echo "mCREAM Graph Module Ensemble — cfmnist"
echo "  deletion: $RUN_DELETION"
echo "  addition: $RUN_ADDITION"
echo "  reversal: $RUN_REVERSAL"
echo "=============================================="

find server_scripts/mcream_experiment/cfmnist/graph_ensemble -name "*.sh" -exec sed -i 's/\r$//' {} \;
chmod +x server_scripts/mcream_experiment/cfmnist/graph_ensemble/**/*.sh

COUNT=0
BASE="server_scripts/mcream_experiment/cfmnist/graph_ensemble"

for LEVEL in low medium high; do
    [ "$RUN_DELETION" = true ] && condor_submit "$BASE/deletion/graph_ensemble_deletion_${LEVEL}_job.sub" && COUNT=$((COUNT+1))
    [ "$RUN_ADDITION" = true ] && condor_submit "$BASE/addition/graph_ensemble_addition_${LEVEL}_job.sub" && COUNT=$((COUNT+1))
    [ "$RUN_REVERSAL" = true ] && condor_submit "$BASE/reversal/graph_ensemble_reversal_${LEVEL}_job.sub" && COUNT=$((COUNT+1))
done

echo ""
echo "Submitted $COUNT jobs (each = 5 seeds)"
condor_q
