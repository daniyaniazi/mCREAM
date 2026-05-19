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
echo "mCREAM: Generate M=10 Expert Graphs (ALL)"
echo "=============================================="
echo "HOST=$(hostname)"
"$PYTHON_BIN" -V

# ─── CFMNIST ─────────────────────────────────────────────────────────────────
echo ""
echo "=== CFMNIST M=10 ==="

# M10/low
"$PYTHON_BIN" generate_expert_graphs.py \
    --dag_path data/FashionMNIST/Complete_Concept_FMNIST_DAG.csv \
    --num_classes 10 \
    --num_experts 10 \
    --disagreement_level low \
    --output_dir data/FashionMNIST/expert_graphs/M10/low

# M10/medium
"$PYTHON_BIN" generate_expert_graphs.py \
    --dag_path data/FashionMNIST/Complete_Concept_FMNIST_DAG.csv \
    --num_classes 10 \
    --num_experts 10 \
    --disagreement_level medium \
    --output_dir data/FashionMNIST/expert_graphs/M10/medium

# M10/high
"$PYTHON_BIN" generate_expert_graphs.py \
    --dag_path data/FashionMNIST/Complete_Concept_FMNIST_DAG.csv \
    --num_classes 10 \
    --num_experts 10 \
    --disagreement_level high \
    --output_dir data/FashionMNIST/expert_graphs/M10/high

# M10/structured_bias (10 experts: alternating types)
"$PYTHON_BIN" generate_expert_graphs.py \
    --dag_path data/FashionMNIST/Complete_Concept_FMNIST_DAG.csv \
    --num_classes 10 \
    --expert_types conservative liberal balanced conservative liberal conservative liberal balanced conservative liberal \
    --output_dir data/FashionMNIST/expert_graphs/M10/structured_bias

# ─── CelebA ──────────────────────────────────────────────────────────────────
echo ""
echo "=== CelebA M=10 ==="

# M10/low
"$PYTHON_BIN" generate_expert_graphs.py \
    --dag_path data/CelebA/final_DAG_unfair.csv \
    --num_classes 1 \
    --num_experts 10 \
    --disagreement_level low \
    --output_dir data/CelebA/expert_graphs/M10/low

# M10/medium
"$PYTHON_BIN" generate_expert_graphs.py \
    --dag_path data/CelebA/final_DAG_unfair.csv \
    --num_classes 1 \
    --num_experts 10 \
    --disagreement_level medium \
    --output_dir data/CelebA/expert_graphs/M10/medium

# M10/high
"$PYTHON_BIN" generate_expert_graphs.py \
    --dag_path data/CelebA/final_DAG_unfair.csv \
    --num_classes 1 \
    --num_experts 10 \
    --disagreement_level high \
    --output_dir data/CelebA/expert_graphs/M10/high

# M10/structured_bias (10 experts: alternating types)
"$PYTHON_BIN" generate_expert_graphs.py \
    --dag_path data/CelebA/final_DAG_unfair.csv \
    --num_classes 1 \
    --expert_types conservative liberal balanced conservative liberal conservative liberal balanced conservative liberal \
    --output_dir data/CelebA/expert_graphs/M10/structured_bias

echo ""
echo "=============================================="
echo "All M=10 expert graphs generated!"
echo "=============================================="
