# Distractor Software

Generate Maze-style distractor sequences from sentence stimuli using Transformer language models.

The pipeline supports:
- English distractors (default): `openai-community/gpt2-medium`
- German distractors: `dbmdz/german-gpt2`

## What This Repository Does

Given an input file of sentences (optionally with token labels), the system:
1. Loads a language model adapter (`EnglishScorer` or `GermanScorer`).
2. Builds distractor candidates from frequency dictionaries.
3. Scores candidates by surprisal against sentence context.
4. Selects one distractor per labeled position with repetition controls.
5. Writes output as:
   - `delim` (semicolon-delimited table), or
   - `ibex` (Ibex Maze item lines ready to paste into an Ibex items array).

## Models Used

- English model:
  - Hugging Face repo: `openai-community/gpt2-medium`
  - Adapter: `models/english_code/model.py` (`EnglishScorer`)
- German model:
  - Hugging Face repo/local dir: `dbmdz/german-gpt2` / `models/dbmdz-german-gpt2`
  - Adapter: `models/german_code/model.py` (`GermanScorer`)

Notes:
- Large weight files are intentionally not pushed to GitHub.
- Download models locally before first use (see below).

## Installation

```bash
pip install -r requirements.txt
```

## Download Model Files (Required)

Use the downloader script:

```bash
python download_model.py --english
python download_model.py --german
# or both:
python download_model.py --all
```

This downloads to:
- English: `models/openai-community-gpt2-medium`
- German: `models/dbmdz-german-gpt2`

## Quick Start

English (default params):

```bash
python distract.py -i examples/input.txt -o output_en.csv -p config/params.txt -f delim
```

German:

```bash
python distract.py -i examples/sample.csv -o output_de.csv -p config/params_de.txt -f delim
```

Ibex output:

```bash
python distract.py -i examples/input.txt -o output_ibex.txt -p config/params.txt -f ibex
```

Ibex-focused workflow:
1. Generate `ibex` output with `-f ibex`.
2. Open `output_ibex.txt`.
3. Copy the generated lines into your Ibex `items` list in your experiment JS.
4. Keep the trailing commas (the generator writes one line per item, comma-terminated).

## Input File Format

The reader accepts comma- or semicolon-delimited CSV text with:

1. `tag` (condition label)
2. `id` (item id)
3. `sentence`
4. optional `labels` (space-separated token labels)

Example:

```csv
sample;1;The cat sat on the mat.;0 1 2 3 4 5
sample;2;Maria bought flowers for Sunday.;0 1 2 3 4
```

Behavior:
- Header rows are auto-detected.
- If labels are omitted, labels default to token positions.
- Label count must match tokenized sentence length.

## Output Formats

### `delim`
Semicolon-delimited rows:
1. tag
2. id
3. original sentence
4. distractor sentence (`x-x-x` placeholder at position 0)
5. label sequence

### `ibex`
Each line is an Ibex Maze item:

```text
[["tag", 'id'], "Maze", {s:"<sentence>", a:"<distractor sentence>"}],
```

Field semantics:
- `tag`: your condition/group label from input column 1.
- `id`: your item id from input column 2 (kept as a quoted literal).
- `s`: original sentence string.
- `a`: distractor sentence string (token-aligned with `s`, with `x-x-x` in position 0).

The output is intentionally compatible with standard Ibex Maze item syntax so it can be pasted directly into:

```javascript
var items = [
  // paste generated lines here
];
```

Operational notes for Ibex:
- The generator escapes double quotes inside sentence strings.
- Output is UTF-8; non-ASCII characters (e.g., accents/umlauts) are preserved.
- Item order is grouped by input id and sentence order in the source file.
- If your Ibex project enforces custom item wrappers, keep the generated `{s:..., a:...}` object and adapt only the outer array tuple.

## Configuration

Configuration files:
- English: `config/params.txt`
- German: `config/params_de.txt`

### Core Parameters

- `min_delta`: minimum surprisal increase target
- `min_abs`: minimum absolute surprisal target
- `num_to_test`: candidate pool size before scoring
- `model_loc`, `model_class`: model adapter module/class
- `dictionary_loc`, `dictionary_class`: candidate dictionary backend
- `threshold_loc`, `threshold_name`: threshold function
- `exclude_words`: banned distractor list
- `include_words`: optional vocabulary allow-list
- `max_repeat`: repetition cap across a set
- `hf_model_name`: Hugging Face repo id or local path

### Default English Config

Current defaults in `config/params.txt` use:
- `EnglishScorer`
- `openai-community/gpt2-medium`
- English thresholding (`get_thresholds_en`)

### Default German Config

Current defaults in `config/params_de.txt` use:
- `GermanScorer`
- local German model path (`models/dbmdz-german-gpt2`)
- German thresholding (`get_thresholds`)

## Offline / Air-Gapped Usage

If the machine cannot reach Hugging Face:
1. Download models on a networked machine using `download_model.py`.
2. Copy model folders to this repo.
3. Set explicit local paths in params:

English:

```txt
hf_model_name: "models/openai-community-gpt2-medium"
```

German:

```txt
hf_model_name: "models/dbmdz-german-gpt2"
```

## CLI Reference

```bash
python distract.py -h
```

Key options:
- `-i, --input` input file
- `-o, --output` output file
- `-p, --parameters` params file (default `config/params.txt`)
- `-f, --format` `delim` or `ibex` (use `ibex` for direct Ibex Maze item lines)

## Repository Layout

- `distract.py`: CLI entrypoint
- `main.py`: pipeline orchestration
- `input.py`: input parsing and label handling
- `sentence_set.py`: scoring and distractor selection logic
- `wordfreq_distractor.py`: dictionary backends + thresholds
- `output.py`: `delim`/`ibex` writers
- `models/english_code/model.py`: English model adapter
- `models/german_code/model.py`: German model adapter
- `download_model.py`: model downloader
- `config/`: params and exclusion lists
- `examples/`: sample inputs/outputs

## Troubleshooting

### Error: failed to load English model

If you see:
- `ERROR: Failed to load English model 'openai-community/gpt2-medium'...`

Do:
1. `python download_model.py --english`
2. Set `hf_model_name: "models/openai-community-gpt2-medium"` in `config/params.txt`

### Push rejected by GitHub due large files

Model weight files are ignored in `.gitignore`. Do not commit model binaries (`*.bin`, `*.safetensors`, etc.).

### Distractor quality issues

Tune:
- `min_delta`, `min_abs`, `num_to_test`
- `exclude_words` (add noisy terms)
- `include_words` (optional allow-list for stricter vocab control)
