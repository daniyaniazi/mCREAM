#!/usr/bin/env bash
set -euo pipefail

# Demo folder inside your mCREAM repo
PROJECT_ROOT="/home/dani00003/mCREAM/server_scripts/demo_experiment"

# Prefer your home venv if it exists in the container, otherwise use container python.
if [ -x "/home/dani00003/.venvs/mcream/bin/python" ]; then
	PYTHON_BIN="/home/dani00003/.venvs/mcream/bin/python"
elif command -v python3 >/dev/null 2>&1; then
	PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
	PYTHON_BIN="$(command -v python)"
else
	echo "ERROR: No python interpreter found in this container image." >&2
	exit 127
fi

cd "$PROJECT_ROOT"

echo "HOST=$(hostname)"
echo "PYTHON=$PYTHON_BIN"
"$PYTHON_BIN" -V
nvidia-smi || true

"$PYTHON_BIN" -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available(), 'cuda_version=', torch.version.cuda)"
"$PYTHON_BIN" "$@"