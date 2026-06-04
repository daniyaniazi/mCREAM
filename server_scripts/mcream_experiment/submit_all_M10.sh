#!/usr/bin/env bash
# Submit all M=10 mCREAM experiments (baselines + edge + graph) for cfmnist and celeba
#
# USAGE:
#   Step 1: Generate expert graphs FIRST (submit and wait for completion):
#       condor_submit server_scripts/mcream_experiment/generate_M10_expert_graphs_job.sub
#
#   Step 2: Once graphs are generated, submit all training jobs:
#       bash server_scripts/mcream_experiment/submit_all_M10.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting all M=10 jobs..."
echo "(Make sure expert graphs are already generated!)"
echo ""

# --- CFMNIST ---
echo "=== CFMNIST ==="
for sub in "$SCRIPT_DIR"/cfmnist/baselines/intersection/intersection_M10_*_job.sub; do
    echo "  condor_submit $sub"
    condor_submit "$sub"
done
for sub in "$SCRIPT_DIR"/cfmnist/baselines/majority/majority_M10_*_job.sub; do
    echo "  condor_submit $sub"
    condor_submit "$sub"
done
for sub in "$SCRIPT_DIR"/cfmnist/baselines/union/union_M10_*_job.sub; do
    echo "  condor_submit $sub"
    condor_submit "$sub"
done
for sub in "$SCRIPT_DIR"/cfmnist/edge/edge_M10_*_job.sub; do
    echo "  condor_submit $sub"
    condor_submit "$sub"
done
for sub in "$SCRIPT_DIR"/cfmnist/graph/graph_M10_*_job.sub; do
    echo "  condor_submit $sub"
    condor_submit "$sub"
done

echo ""
echo "=== CelebA ==="
for sub in "$SCRIPT_DIR"/celeba/baselines/intersection/intersection_M10_*_job.sub; do
    echo "  condor_submit $sub"
    condor_submit "$sub"
done
for sub in "$SCRIPT_DIR"/celeba/baselines/majority/majority_M10_*_job.sub; do
    echo "  condor_submit $sub"
    condor_submit "$sub"
done
for sub in "$SCRIPT_DIR"/celeba/baselines/union/union_M10_*_job.sub; do
    echo "  condor_submit $sub"
    condor_submit "$sub"
done
for sub in "$SCRIPT_DIR"/celeba/edge/edge_M10_*_job.sub; do
    echo "  condor_submit $sub"
    condor_submit "$sub"
done
for sub in "$SCRIPT_DIR"/celeba/graph/graph_M10_*_job.sub; do
    echo "  condor_submit $sub"
    condor_submit "$sub"
done

echo ""
echo "All M=10 jobs submitted!"
