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
echo "mCREAM: Generate Expert Graphs for cfmnist"
echo "=============================================="
echo "HOST=$(hostname)"
"$PYTHON_BIN" -V

"$PYTHON_BIN" -c "import torch; print('torch=', torch.__version__)"

# Generate all variations using the ONE reusable script
# M5/low
"$PYTHON_BIN" generate_expert_graphs.py \
    --dag_path data/FashionMNIST/Complete_Concept_FMNIST_DAG.csv \
    --num_classes 10 \
    --num_experts 5 \
    --disagreement_level low \
    --output_dir data/FashionMNIST/expert_graphs/M5/low

# M5/medium
"$PYTHON_BIN" generate_expert_graphs.py \
    --dag_path data/FashionMNIST/Complete_Concept_FMNIST_DAG.csv \
    --num_classes 10 \
    --num_experts 5 \
    --disagreement_level medium \
    --output_dir data/FashionMNIST/expert_graphs/M5/medium

# M5/high
"$PYTHON_BIN" generate_expert_graphs.py \
    --dag_path data/FashionMNIST/Complete_Concept_FMNIST_DAG.csv \
    --num_classes 10 \
    --num_experts 5 \
    --disagreement_level high \
    --output_dir data/FashionMNIST/expert_graphs/M5/high

# M2/medium (for ablation)
"$PYTHON_BIN" generate_expert_graphs.py \
    --dag_path data/FashionMNIST/Complete_Concept_FMNIST_DAG.csv \
    --num_classes 10 \
    --num_experts 2 \
    --disagreement_level medium \
    --output_dir data/FashionMNIST/expert_graphs/M2/medium

# M10/medium (for ablation)
"$PYTHON_BIN" generate_expert_graphs.py \
    --dag_path data/FashionMNIST/Complete_Concept_FMNIST_DAG.csv \
    --num_classes 10 \
    --num_experts 10 \
    --disagreement_level medium \
    --output_dir data/FashionMNIST/expert_graphs/M10/medium

echo ""
echo "=============================================="
echo "Expert graph generation complete!"
echo "=============================================="
