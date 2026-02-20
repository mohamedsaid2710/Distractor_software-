# Distractor Software

Distractor Software generates Maze-style distractor stimuli for psycholinguistic experiments in English and German.

## Attribution

This project is based on the original Maze repository by Victoria Boyce:

- https://github.com/vboyce/Maze

This implementation has been adapted and now differs from the original in models, configuration, and workflow details.

## Quick Start

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
