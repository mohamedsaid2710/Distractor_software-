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


def main() -> None:
    if not compileall.compile_dir(ROOT, quiet=1):
        raise RuntimeError("Smoke test failed: Python compilation errors were found.")

    for path in ("params.txt", "params_de.txt"):
        params = set_params(path)
        for key in ("min_delta", "min_abs", "num_to_test"):
            if key not in params:
                raise RuntimeError(f"Missing required key '{key}' in {path}")

    read_input("English_sample.txt")
    read_input("german_sample.txt")

    print("Smoke test passed")


if __name__ == "__main__":
    main()
