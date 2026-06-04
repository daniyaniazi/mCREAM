#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

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
echo "mCREAM: Graph Aggregation (M=5, Structured Bias) - CelebA"
echo "=============================================="
echo "HOST=$(hostname)"
"$PYTHON_BIN" -V
nvidia-smi || true

"$PYTHON_BIN" -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available())"

EXPERT_DIR="$PROJECT_ROOT/data/CelebA/expert_graphs/M5/structured_bias"
if [ ! -d "$EXPERT_DIR" ]; then
    echo "Generating structured expert graphs for CelebA..."
    "$PYTHON_BIN" -c "
from src.expert_graphs.generation import generate_structured_experts, save_expert_graphs
expert_types = ['conservative', 'conservative', 'liberal', 'liberal', 'balanced']
expert_u2c, expert_c2y, u2c_star, c2y_star = generate_structured_experts(
    dag_path='data/CelebA/final_DAG_unfair.csv',
    num_classes=1, expert_types=expert_types, base_seed=42
)
save_expert_graphs(expert_u2c, expert_c2y, '$EXPERT_DIR', {'expert_types': expert_types, 'num_experts': 5})
print('Expert graphs saved!')
"
fi

echo ""
echo "Running mCREAM with Graph aggregation (structured bias)..."
"$PYTHON_BIN" mcream_main.py --config all_configs/mcream_configs/celeba/graph/graph_M5_structured_bias.yaml

echo "Done!"
