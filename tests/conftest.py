"""Shared test setup: make src/ (the package), tests/ (fake_lm) and tools/
importable,
so `pytest tests/...` works from any directory."""
import os
import sys

TESTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS)
for path in (os.path.join(ROOT, "src"), TESTS, os.path.join(ROOT, "tools")):
    if path not in sys.path:
        sys.path.insert(0, path)
