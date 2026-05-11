#!/usr/bin/env bash
# Master script to submit all CREAM with noisy single graph experiments
# These compare CREAM using one noisy graph (expert_0 from M=5) vs mCREAM using all M=5 graphs

set -euo pipefail

BASE_DIR="/home/dani00003/mCREAM/server_scripts/mcream_experiment"

echo "=== Submitting CREAM Noisy Graph Experiments ==="
echo "These experiments use a single noisy graph (expert_0 from M=5 set) with standard CREAM"
echo ""

# ----- cfmnist -----
echo "--- cfmnist CREAM Noisy ---"
condor_submit "$BASE_DIR/cfmnist/cream_noisy/CREAM_noisy_low_job.sub"
condor_submit "$BASE_DIR/cfmnist/cream_noisy/CREAM_noisy_medium_job.sub"
condor_submit "$BASE_DIR/cfmnist/cream_noisy/CREAM_noisy_high_job.sub"
condor_submit "$BASE_DIR/cfmnist/cream_noisy/CREAM_noisy_structured_bias_job.sub"

# ----- celeba -----
echo "--- celeba CREAM Noisy ---"
condor_submit "$BASE_DIR/celeba/cream_noisy/CREAM_noisy_low_job.sub"
condor_submit "$BASE_DIR/celeba/cream_noisy/CREAM_noisy_medium_job.sub"
condor_submit "$BASE_DIR/celeba/cream_noisy/CREAM_noisy_high_job.sub"
condor_submit "$BASE_DIR/celeba/cream_noisy/CREAM_noisy_structured_bias_job.sub"

echo ""
echo "=== Submitted 8 CREAM noisy jobs ==="
echo "Use 'condor_q' to check job status"
