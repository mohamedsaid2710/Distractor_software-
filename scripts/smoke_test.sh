#!/usr/bin/env bash
set -euo pipefail

python -m compileall -q .

python - <<'PY'
from set_params import set_params
from input import read_input

for path in ("config/params.txt", "config/params_de.txt"):
    params = set_params(path)
    for key in ("min_delta", "min_abs", "num_to_test"):
        if key not in params:
            raise RuntimeError(f"Missing required key '{key}' in {path}")

read_input("examples/input.txt")
read_input("examples/sample.csv")

print("Smoke test passed")
PY
