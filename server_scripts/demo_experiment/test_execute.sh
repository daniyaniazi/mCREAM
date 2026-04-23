#!/usr/bin/env bash
set -euo pipefail

# Demo folder inside your mCREAM repo
PROJECT_ROOT="/home/dani00003/mCREAM/server_scripts/demo_experiment"

# Python from your project venv
PYTHON_BIN="/home/dani00003/.venvs/mcream/bin/python"

cd "$PROJECT_ROOT"

echo "HOST=$(hostname)"
echo "PYTHON=$PYTHON_BIN"
"$PYTHON_BIN" -V
nvidia-smi || true

"$PYTHON_BIN" -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available(), 'cuda_version=', torch.version.cuda)"
"$PYTHON_BIN" "$@"