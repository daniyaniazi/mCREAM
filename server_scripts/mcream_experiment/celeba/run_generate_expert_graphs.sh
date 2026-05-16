#!/usr/bin/env bash
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
echo "mCREAM: Generate Expert Graphs for CelebA"
echo "=============================================="
echo "HOST=$(hostname)"
"$PYTHON_BIN" -V

"$PYTHON_BIN" -c "import torch; print('torch=', torch.__version__)"

# M5/low
"$PYTHON_BIN" generate_expert_graphs.py \
    --dag_path data/CelebA/final_DAG_unfair.csv \
    --num_classes 1 \
    --num_experts 5 \
    --disagreement_level low \
    --output_dir data/CelebA/expert_graphs/M5/low

# M5/medium
"$PYTHON_BIN" generate_expert_graphs.py \
    --dag_path data/CelebA/final_DAG_unfair.csv \
    --num_classes 1 \
    --num_experts 5 \
    --disagreement_level medium \
    --output_dir data/CelebA/expert_graphs/M5/medium

# M5/high
"$PYTHON_BIN" generate_expert_graphs.py \
    --dag_path data/CelebA/final_DAG_unfair.csv \
    --num_classes 1 \
    --num_experts 5 \
    --disagreement_level high \
    --output_dir data/CelebA/expert_graphs/M5/high

# M5/structured_bias (uses expert_types instead of disagreement_level)
"$PYTHON_BIN" generate_expert_graphs.py \
    --dag_path data/CelebA/final_DAG_unfair.csv \
    --num_classes 1 \
    --expert_types conservative liberal balanced conservative liberal \
    --output_dir data/CelebA/expert_graphs/M5/structured_bias

echo ""
echo "=== Generating CREAM noisy DAGs from expert_0 ==="
for LEVEL in low medium high structured_bias; do
    "$PYTHON_BIN" generate_single_noisy_dags.py --dataset celeba --level "$LEVEL"
done

echo ""
echo "=============================================="
echo "CelebA expert graph generation complete!"
echo "=============================================="
