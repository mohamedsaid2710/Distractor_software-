#!/usr/bin/env bash
set -euo pipefail

# Resolve the repo root from this script's own location so the venv lookup and
# PYTHONPATH stay correct no matter where the script is invoked from.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Prefer the project venv: the system python3 on a dev machine may be a different
# version with none of the dependencies installed. CI creates .venv via `uv sync`,
# so this resolves correctly there too. Override with PYTHON=... if needed.
if [[ -n "${PYTHON:-}" ]]; then
    PY="$PYTHON"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
    PY="$ROOT/.venv/bin/python"
else
    PY=python3
fi

PYTHONPATH="$ROOT" "$PY" "$ROOT/scripts/smoke_test.py"
