#!/usr/bin/env bash
# Generate all single-action expert graphs for ensemble experiments.
# Run this ONCE before submitting any ensemble experiment jobs.
set -euo pipefail

PROJECT_ROOT="/home/dani00003/mCREAM"
CONDA_PYTHON="/home/dani00003/miniconda3/envs/mcream/bin/python"

if [ -x "$CONDA_PYTHON" ]; then
    PYTHON_BIN="$CONDA_PYTHON"
else
    echo "ERROR: Conda env not found at $CONDA_PYTHON" >&2
    exit 127
fi

cd "$PROJECT_ROOT"

echo "=============================================="
echo "Generating ensemble expert graphs (all datasets)"
echo "=============================================="
echo "HOST=$(hostname)"
"$PYTHON_BIN" -V

"$PYTHON_BIN" generate_ensemble_expert_graphs.py --dataset all --num_experts 10

echo ""
echo "=============================================="
echo "Expert graph generation complete!"
echo "Graphs saved under:"
echo "  data/FashionMNIST/expert_graphs/ensemble/"
echo "  data/CelebA/expert_graphs/ensemble/"
echo "=============================================="
