#!/usr/bin/env bash
set -euo pipefail

# Prefer the project venv: the system python3 on a dev machine may be a different
# version with none of the dependencies installed. CI creates .venv via `uv sync`,
# so this resolves correctly there too. Override with PYTHON=... if needed.
if [[ -n "${PYTHON:-}" ]]; then
    PY="$PYTHON"
elif [[ -x .venv/bin/python ]]; then
    PY=.venv/bin/python
else
    PY=python3
fi

PYTHONPATH=. "$PY" scripts/smoke_test.py
