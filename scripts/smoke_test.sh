#!/usr/bin/env bash
set -euo pipefail

python -m compileall -q .

python - <<'PY'
from set_params import set_params
from input import read_input

for path in ("params.txt", "params_de.txt"):
    params = set_params(path)
    for key in ("min_delta", "min_abs", "num_to_test"):
        if key not in params:
            raise RuntimeError(f"Missing required key '{key}' in {path}")

read_input("English_sample.txt")
read_input("german_sample.txt")

print("Smoke test passed")
PY
