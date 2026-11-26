#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="/home/dani00003/mCREAM"
ANALYSIS_DIR="$PROJECT_ROOT/server_scripts/cream_experiment/analysis"
cd "$PROJECT_ROOT"
chmod +x "$ANALYSIS_DIR/run_edge_sensitivity_celeba.sh" "$ANALYSIS_DIR/run_edge_sensitivity_cfmnist.sh" "$ANALYSIS_DIR/run_edge_sensitivity_retrain.sh"
condor_submit "$ANALYSIS_DIR/edge_sensitivity_celeba.sub"
condor_submit "$ANALYSIS_DIR/edge_sensitivity_cfmnist.sub"
