# Distractor Software

Distractor Software generates Maze-style distractor stimuli for psycholinguistic experiments in English and German.

## Attribution

This project is based on the original Maze repository by Victoria Boyce:

- https://github.com/vboyce/Maze

This implementation has been adapted and now differs from the original in models, configuration, and workflow details.

## What This Repository Does

- Scores candidate distractors with causal Transformer language models.
- Produces experiment-ready outputs in two formats:
  - `delim` (semicolon-delimited table)
  - `ibex` (for PCIbex)
- Supports separate English and German parameter presets.
- Runs online by default (Hugging Face model IDs), with optional offline local model files.

## Models Used

| Language | Hugging Face model | Local directory | Params file |
|---|---|---|---|
| English | `openai-community/gpt2-medium` | `models/openai-community-gpt2-medium` | `params.txt` |
| German | `dbmdz/german-gpt2` | `models/dbmdz-german-gpt2` | `params_de.txt` |

## Platform Support

This repo is designed to run on Linux, macOS, and Windows.

- Linux/macOS: use `python`
- Windows: use `py` if `python` is not available

## Installation

Linux/macOS:

```bash
python -m pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
py -m pip install -r requirements.txt
```

Optional (German noun/proper-noun post-casing fallback):

```bash
python -m spacy download de_core_news_sm
```

Windows (PowerShell/CMD):

```powershell
py -m spacy download de_core_news_sm
```

## Optional: Download Model Files (Offline Use)

```bash
python download_model.py --english
python download_model.py --german
# or
python download_model.py --all
```

Only the files needed by this pipeline are downloaded (ONNX artifacts are skipped).
Use this only if you want to run without internet access.


## How to run:

Detailed instructions on how to run is in the GitHub Wiki:

- https://github.com/mohamedsaid2710/Distractor_software-/wiki

