#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/dani00003/mCREAM"
CONDA_PYTHON="/home/dani00003/miniconda3/envs/mcream/bin/python"
CONFIG_PATH="all_configs/best_hparams/CREAM/CREAM_cub_soft_seed0_config.yaml"
HEATMAP_IMAGES=5
HEATMAP_INDICES=""
HEATMAP_IDS_FILE="/home/dani00003/mCREAM/server_scripts/cream_experiment/cub/CUB_IDs.txt"
HEATMAP_RANDOM_VAL=0
HEATMAP_TOP_K=8

if [ -x "$CONDA_PYTHON" ]; then
    PYTHON_BIN="$CONDA_PYTHON"
else
    echo "ERROR: Conda env not found at $CONDA_PYTHON" >&2
    exit 127
fi

cd "$PROJECT_ROOT"
echo "HOST=$(hostname)"
"$PYTHON_BIN" -V
nvidia-smi || true
"$PYTHON_BIN" -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available())"
"$PYTHON_BIN" -c "import pytorch_lightning, torchvision, yaml; print('deps_ok=1')"

if [ -n "$HEATMAP_IDS_FILE" ]; then
    echo "Generating CREAM CUB seed=0 heatmaps for IDs in: $HEATMAP_IDS_FILE..."
    "$PYTHON_BIN" evaluate_metrics.py \
        --config "$CONFIG_PATH" \
        --heatmap_ids "$HEATMAP_IDS_FILE" \
        --heatmap_search_splits test,val \
        --heatmap_random_val "$HEATMAP_RANDOM_VAL" \
        --heatmap_top_k "$HEATMAP_TOP_K" \
        --only_heatmaps
elif [ -n "$HEATMAP_INDICES" ]; then
    echo "Generating CREAM CUB seed=0 heatmaps for test indices: $HEATMAP_INDICES..."
    "$PYTHON_BIN" evaluate_metrics.py \
        --config "$CONFIG_PATH" \
        --heatmap_indices "$HEATMAP_INDICES" \
        --heatmap_split test \
        --heatmap_random_val "$HEATMAP_RANDOM_VAL" \
        --heatmap_top_k "$HEATMAP_TOP_K" \
        --only_heatmaps
else
    echo "Generating CREAM CUB seed=0 heatmaps for first $HEATMAP_IMAGES test samples..."
    "$PYTHON_BIN" evaluate_metrics.py \
        --config "$CONFIG_PATH" \
        --heatmap_images "$HEATMAP_IMAGES" \
        --heatmap_split test \
        --heatmap_random_val "$HEATMAP_RANDOM_VAL" \
        --heatmap_top_k "$HEATMAP_TOP_K" \
        --only_heatmaps
fi

echo "Done!"
