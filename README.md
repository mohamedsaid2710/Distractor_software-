# Distractor Software

Generate Maze-style distractor stimuli for English and German using Transformer language models.

## Quick Start

Install:

```bash
pip install -r requirements.txt
```

Download models:

```bash
python download_model.py --english
python download_model.py --german
```

Run English (`delim`):

```bash
python distract.py -i examples/input.txt -o output_en.csv -p config/params.txt -f delim
```

Run English (`ibex`):

```bash
python distract.py -i examples/input.txt -o output_ibex.txt -p config/params.txt -f ibex
```

## Documentation

Full documentation is in the Wiki:

- https://github.com/mohamedsaid2710/Distractor_software-/wiki

Local wiki-ready pages are also provided in `docs/wiki/`.

To publish `docs/wiki` into the GitHub Wiki repo from this project:

```bash
./scripts/wiki_sync.sh
```
