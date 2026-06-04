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
echo "mCREAM: Edge Aggregation (M=5, Structured Bias) - CelebA"
echo "=============================================="
echo "HOST=$(hostname)"
"$PYTHON_BIN" -V
nvidia-smi || true

"$PYTHON_BIN" -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available())"

# Check/generate expert graphs with structured bias
EXPERT_DIR="$PROJECT_ROOT/data/CelebA/expert_graphs/M5/structured_bias"
if [ ! -d "$EXPERT_DIR" ]; then
    echo "Generating expert graphs with structured bias for CelebA..."
    "$PYTHON_BIN" -c "
from src.expert_graphs.generation import generate_structured_experts, save_expert_graphs, EXPERT_BIAS_TYPES
expert_types = ['conservative', 'conservative', 'liberal', 'liberal', 'balanced']
expert_u2c, expert_c2y, u2c_star, c2y_star = generate_structured_experts(
    dag_path='data/CelebA/final_DAG_unfair.csv',
    num_classes=1,
    expert_types=expert_types,
    base_seed=42
)
save_expert_graphs(expert_u2c, expert_c2y, '$EXPERT_DIR', {'disagreement_level': 'structured_bias', 'num_experts': 5, 'expert_types': expert_types})
print('Expert graphs saved!')
"
fi

echo ""
echo "Running mCREAM with Edge aggregation..."
"$PYTHON_BIN" mcream_main.py --config all_configs/mcream_configs/celeba/edge/edge_M5_structured_bias.yaml

echo "Done!"
