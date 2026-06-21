#!/usr/bin/env bash
# Submit CREAM baseline on GT graph.
# Uses best cfmnist config: all_configs/best_hparams/CREAM/CREAM_best_cfmnist_soft_config.yaml
#
# USAGE:
#   ./submit_cream_baseline.sh

set -euo pipefail
cd ~/mCREAM

echo "=============================================="
echo "CREAM Baseline — GT graph (cfmnist)"
echo "=============================================="

condor_submit server_scripts/cream_experiment/cfmnist/cream_cfmnist_job.sub

echo ""
echo "Submitted 1 job"
condor_q
