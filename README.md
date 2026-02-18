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
  - `ibex` (Maze item lines for Ibex)
- Supports separate English and German parameter presets.
- Runs online by default (Hugging Face model IDs), with optional offline local model files.

## Models Used

| Language | Hugging Face model | Local directory | Params file |
|---|---|---|---|
| English | `openai-community/gpt2-medium` | `models/openai-community-gpt2-medium` | `params.txt` |
| German | `dbmdz/german-gpt2` | `models/dbmdz-german-gpt2` | `params_de.txt` |

## Installation

```bash
pip install -r requirements.txt
```

Optional (German noun/proper-noun post-casing fallback):

```bash
python -m spacy download de_core_news_sm
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

## Run the Pipeline

English (`delim`):

```bash
python distract.py -i English_sample.txt -o output_en.csv -p params.txt -f delim
```

German (`delim`):

```bash
python distract.py -i german_sample.txt -o output_de.csv -p params_de.txt -f delim
```

Ibex output:

```bash
python distract.py -i English_sample.txt -o output_ibex.txt -p params.txt -f ibex
```

The default params use online model IDs for both languages (`openai-community/gpt2-medium`, `dbmdz/german-gpt2`).
For offline runs, first download models and then set `hf_model_name` to local paths in `params*.txt`.

## Input Format

Expected columns:

1. `tag`
2. `id`
3. `sentence`
4. optional labels (`0 1 2 ...`)

Example:

```csv
sample;1;The cat sat on the mat.;0 1 2 3 4 5
sample;2;Maria bought flowers for Sunday.;0 1 2 3 4
```

## Documentation

Detailed documentation is maintained in the GitHub Wiki:

- https://github.com/mohamedsaid2710/Distractor_software-/wiki

## Important Git Note

Model weight files are intentionally ignored by git and must not be committed.
Use `download_model.py` to restore them locally on any machine.

## Smoke Test

Run a lightweight repository check:

```bash
./scripts/smoke_test.sh
```
