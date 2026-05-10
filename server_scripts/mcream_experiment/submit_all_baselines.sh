#!/usr/bin/env bash
# Master script to submit all baseline experiments for cfmnist and celeba
# Usage: ./submit_all_baselines.sh

set -euo pipefail
cd ~/mCREAM

echo "=============================================="
echo "Submitting ALL Baseline Experiments (M=5)"
echo "=============================================="

# Fix line endings and permissions
find server_scripts -name "*.sh" -exec sed -i 's/\r$//' {} \;
chmod +x server_scripts/mcream_experiment/cfmnist/**/*.sh
chmod +x server_scripts/mcream_experiment/celeba/**/*.sh

echo ""
echo "=== CFMNIST Baselines ==="

# Union
echo "Submitting cfmnist union..."
condor_submit server_scripts/mcream_experiment/cfmnist/baselines/union/union_M5_low_job.sub
condor_submit server_scripts/mcream_experiment/cfmnist/baselines/union/union_M5_medium_job.sub
condor_submit server_scripts/mcream_experiment/cfmnist/baselines/union/union_M5_high_job.sub
condor_submit server_scripts/mcream_experiment/cfmnist/baselines/union/union_M5_structured_bias_job.sub

# Intersection
echo "Submitting cfmnist intersection..."
condor_submit server_scripts/mcream_experiment/cfmnist/baselines/intersection/intersection_M5_low_job.sub
condor_submit server_scripts/mcream_experiment/cfmnist/baselines/intersection/intersection_M5_medium_job.sub
condor_submit server_scripts/mcream_experiment/cfmnist/baselines/intersection/intersection_M5_high_job.sub
condor_submit server_scripts/mcream_experiment/cfmnist/baselines/intersection/intersection_M5_structured_bias_job.sub

# Majority
echo "Submitting cfmnist majority..."
condor_submit server_scripts/mcream_experiment/cfmnist/baselines/majority/majority_M5_low_job.sub
condor_submit server_scripts/mcream_experiment/cfmnist/baselines/majority/majority_M5_medium_job.sub
condor_submit server_scripts/mcream_experiment/cfmnist/baselines/majority/majority_M5_high_job.sub
condor_submit server_scripts/mcream_experiment/cfmnist/baselines/majority/majority_M5_structured_bias_job.sub

echo ""
echo "=== CELEBA Baselines ==="

# Union
echo "Submitting celeba union..."
condor_submit server_scripts/mcream_experiment/celeba/baselines/union/union_M5_low_job.sub
condor_submit server_scripts/mcream_experiment/celeba/baselines/union/union_M5_medium_job.sub
condor_submit server_scripts/mcream_experiment/celeba/baselines/union/union_M5_high_job.sub
condor_submit server_scripts/mcream_experiment/celeba/baselines/union/union_M5_structured_bias_job.sub

# Intersection
echo "Submitting celeba intersection..."
condor_submit server_scripts/mcream_experiment/celeba/baselines/intersection/intersection_M5_low_job.sub
condor_submit server_scripts/mcream_experiment/celeba/baselines/intersection/intersection_M5_medium_job.sub
condor_submit server_scripts/mcream_experiment/celeba/baselines/intersection/intersection_M5_high_job.sub
condor_submit server_scripts/mcream_experiment/celeba/baselines/intersection/intersection_M5_structured_bias_job.sub

# Majority
echo "Submitting celeba majority..."
condor_submit server_scripts/mcream_experiment/celeba/baselines/majority/majority_M5_low_job.sub
condor_submit server_scripts/mcream_experiment/celeba/baselines/majority/majority_M5_medium_job.sub
condor_submit server_scripts/mcream_experiment/celeba/baselines/majority/majority_M5_high_job.sub
condor_submit server_scripts/mcream_experiment/celeba/baselines/majority/majority_M5_structured_bias_job.sub

echo ""
echo "=============================================="
echo "Submitted 24 baseline experiments!"
echo "=============================================="
condor_q
