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
echo "mCREAM: Graph-level Aggregation (M=5, Low)"
echo "=============================================="
echo "HOST=$(hostname)"
"$PYTHON_BIN" -V
nvidia-smi || true

"$PYTHON_BIN" -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available())"

EXPERT_DIR="$PROJECT_ROOT/data/FashionMNIST/expert_graphs/M5/low"
if [ ! -d "$EXPERT_DIR" ]; then
    echo "Generating expert graphs..."
    "$PYTHON_BIN" -c "
from src.expert_graphs.generation import generate_expert_graphs_from_dag, save_expert_graphs, DISAGREEMENT_LEVELS
params = DISAGREEMENT_LEVELS['low']
expert_u2c, expert_c2y, u2c_star, c2y_star = generate_expert_graphs_from_dag(
    dag_path='data/FashionMNIST/Complete_Concept_FMNIST_DAG.csv',
    num_classes=10, num_experts=5,
    p_del=params['p_del'], p_add=params['p_add'], p_rev=params['p_rev'],
    base_seed=42
)
save_expert_graphs(expert_u2c, expert_c2y, '$EXPERT_DIR', {'disagreement_level': 'low', 'num_experts': 5})
print('Expert graphs saved!')
"
fi

echo ""
echo "Running mCREAM with Graph aggregation (low)..."
"$PYTHON_BIN" mcream_main.py --config all_configs/mcream_configs/cfmnist/graph/graph_M5_low.yaml

echo "Done!"
