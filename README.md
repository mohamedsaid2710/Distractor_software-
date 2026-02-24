# Distractor Software

Distractor Software generates Maze-style distractor stimuli for psycholinguistic experiments in English and German.

## Attribution

This project is based on the original Maze repository by Victoria Boyce:

- https://github.com/vboyce/Maze

This implementation has been adapted and now differs from the original in models, configuration, and workflow details.

## Quick Start

Python version:

- Recommended: Python `3.12` (see `.python-version`).

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Generate distractors:

```bash
python distract.py -i german_sample.txt -o output.txt -p params_de.txt -f delim
```

## Documentation

Detailed usage, configuration behavior, quality checks, and troubleshooting live in the wiki:

- https://github.com/mohamedsaid2710/Distractor_software-/wiki

## Selection Behavior (Frequency vs Surprisal)

- Candidate pools are built using target length and target frequency ranges first.
- If the pool is too small, the search widens in both directions: lower-frequency bins and higher-frequency bins.
- Final distractor choice is surprisal-driven (mode-dependent), so frequency constrains the pool but does not directly pick the final winner.
