#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=. python3 scripts/smoke_test.py
