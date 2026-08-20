"""Shared test fixtures.

The repo is flat (no package): every module assumes the repo root is on
sys.path.  Tests are run from the repo root, but make that explicit here so
`pytest tests/...` works from anywhere.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
