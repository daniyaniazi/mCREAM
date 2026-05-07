#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/dani00003/mCREAM"

# Conda embeds a real Python binary (not a symlink), so it works inside Docker
# when the home directory is mounted via +WantGPUHomeMounted = true.
CONDA_PYTHON="/home/dani00003/miniconda3/envs/mcream/bin/python"

if [ -x "$CONDA_PYTHON" ]; then
    PYTHON_BIN="$CONDA_PYTHON"
else
    echo "ERROR: Conda env not found at $CONDA_PYTHON" >&2
    echo "Set it up once on the submit node:" >&2
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" >&2
    echo "  bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3" >&2
    echo "  ~/miniconda3/bin/conda create -n mcream python=3.11 -y" >&2
    echo "  ~/miniconda3/bin/conda run -n mcream pip install -r ~/mCREAM/requirements.txt" >&2
    exit 127
fi

cd "$PROJECT_ROOT"

echo "HOST=$(hostname)"
echo "PYTHON=$PYTHON_BIN"
"$PYTHON_BIN" -V
nvidia-smi || true

"$PYTHON_BIN" -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available(), 'cuda_version=', torch.version.cuda)"
"$PYTHON_BIN" -c "import pytorch_lightning, torchvision, yaml; print('deps_ok=1')"

"$PYTHON_BIN" simple_main.py --config all_configs/best_hparams/CREAM/CREAM_best_ifmnist_soft_config.yaml
