#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if command -v micromamba >/dev/null 2>&1; then
  manager=micromamba
elif command -v mamba >/dev/null 2>&1; then
  manager=mamba
elif command -v conda >/dev/null 2>&1; then
  manager=conda
else
  cat >&2 <<'EOF'
No Conda-compatible environment manager was found.
Install Miniforge, Conda, Mamba, or Micromamba and run this script again.
EOF
  exit 1
fi

if "$manager" env list | awk '{print $1}' | grep -qx routerec; then
  "$manager" env update --name routerec --file environment.yml --prune
else
  "$manager" env create --file environment.yml
fi

cat <<EOF
RouteRec environment is ready.

Activate it with:
  $manager activate routerec

Then validate the checkout with:
  bash scripts/setup_server.sh
EOF
