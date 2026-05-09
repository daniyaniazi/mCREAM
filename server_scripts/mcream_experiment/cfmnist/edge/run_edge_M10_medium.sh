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
echo "mCREAM: Edge-level Learning (M=10, Medium)"
echo "=============================================="
echo "HOST=$(hostname)"
"$PYTHON_BIN" -V
nvidia-smi || true

"$PYTHON_BIN" -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available())"

# Check if expert graphs exist
EXPERT_DIR="$PROJECT_ROOT/data/FashionMNIST/expert_graphs/M10/medium"
if [ ! -d "$EXPERT_DIR" ]; then
    echo "Generating expert graphs (M=10)..."
    "$PYTHON_BIN" -c "
from src.expert_graphs.generation import generate_expert_graphs_from_dag, save_expert_graphs, DISAGREEMENT_LEVELS
params = DISAGREEMENT_LEVELS['medium']
expert_u2c, expert_c2y, u2c_star, c2y_star = generate_expert_graphs_from_dag(
    dag_path='data/FashionMNIST/Complete_Concept_FMNIST_DAG.csv',
    num_classes=10, num_experts=10,
    p_del=params['p_del'], p_add=params['p_add'], p_rev=params['p_rev'],
    base_seed=42
)
save_expert_graphs(expert_u2c, expert_c2y, '$EXPERT_DIR', {'disagreement_level': 'medium', 'num_experts': 10})
print('Expert graphs saved!')
"
fi

echo ""
echo "Running mCREAM with Edge aggregation (M=10, medium)..."
"$PYTHON_BIN" mcream_main.py --config all_configs/mcream_configs/cfmnist/edge/edge_M10_medium.yaml

echo "Done!"
