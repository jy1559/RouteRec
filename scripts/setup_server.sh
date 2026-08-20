#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python - <<'PY'
import sys

if not ((3, 10) <= sys.version_info[:2] < (3, 12)):
    raise SystemExit(
        f"RouteRec requires Python 3.10 or 3.11; found {sys.version.split()[0]}"
    )
print(f"Python: {sys.version.split()[0]}")
PY

if ! python -c "import torch" >/dev/null 2>&1; then
  cat >&2 <<'EOF'
PyTorch is not installed. Create the pinned Conda environment first:
  bash scripts/create_env.sh
  conda activate routerec       # or: micromamba activate routerec
EOF
  exit 1
fi

python -m pip install -e .
python - <<'PY'
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA runtime: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")
PY
python scripts/check_repo.py
python scripts/check_data.py
python scripts/check_gpu.py
python -m unittest discover -s tests

echo "RouteRec server setup checks passed."
