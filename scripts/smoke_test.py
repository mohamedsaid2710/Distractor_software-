#!/usr/bin/env python3

"""Cross-platform smoke test for basic repository sanity checks."""

import compileall
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from input import read_input
from set_params import set_params


# Directories to skip during compilation (large or irrelevant for syntax checks).
_SKIP_DIRS = {'.venv', 'venv', 'env', '__pycache__', 'models', 'node_modules', '.git'}


def main() -> None:
    import re
    skip_rx = re.compile(r'[\\/](' + '|'.join(re.escape(d) for d in _SKIP_DIRS) + r')[\\/]')
    if not compileall.compile_dir(ROOT, quiet=1, rx=skip_rx):
        raise RuntimeError("Smoke test failed: Python compilation errors were found.")

    for path in ("params.txt", "params_de.txt", "params_ar.txt"):
        full_path = os.path.join(ROOT, path)
        params = set_params(full_path)
        for key in ("min_delta", "min_abs", "num_to_test"):
            if key not in params:
                raise RuntimeError(f"Missing required key '{key}' in {full_path}")

    read_input(os.path.join(ROOT, "English_sample.txt"))
    read_input(os.path.join(ROOT, "german_sample.txt"))
    read_input(os.path.join(ROOT, "arabic_sample.txt"))

    print("Smoke test passed")


if __name__ == "__main__":
    main()
