#!/usr/bin/env python3
"""Generate Maze distractors.

    uv run python run_pipeline.py -i examples/german_sample.txt -o output_de.txt -p config/params_de.txt

Options: -f delim|ibex (output format, default delim), --allow-tagger-fallback.
The code lives in src/maze_automate/; the settings you edit live in config/.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from maze_automate.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
