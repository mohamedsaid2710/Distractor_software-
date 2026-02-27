# Distractor Software

**Maze-style distractor stimulus generator for psycholinguistic experiments in English, German, and Arabic.**

Built on Transformer-based language models (GPT-2), the pipeline selects real-word distractors that are contextually implausible while matching the target word in length and frequency range.

## Attribution

Based on the original [Maze repository](https://github.com/vboyce/Maze) by Victoria Boyce. This implementation has been adapted and now differs in models, configuration, and workflow.

## Quick Start

**Requirements:** Python 3.12 (see `.python-version`)

```bash
pip install -r requirements.txt
```

### English

```bash
python distract.py -i English_sample.txt -o output_en.txt -p params.txt -f delim
```

### German

```bash
python distract.py -i german_sample.txt -o output_de.txt -p params_de.txt -f delim
```

### Arabic

```bash
python distract.py -i arabic_sample.txt -o output_ar.txt -p params_ar.txt -f delim
```

> **Note:** Models download automatically from Hugging Face on first run. For offline use, run `python download_model.py --all` first and set `hf_model_name` in your params file to the local path.

## Features

- **Three languages** — English (`gpt2-medium`), German (`dbmdz/german-gpt2`), and Arabic (`aubmindlab/aragpt2-medium`)
- **Two selection modes** — threshold-first (Mode A) or max-implausibility ranking (Mode B)
- **Two output formats** — `delim` (semicolon-delimited table) and `ibex` (PCIbex-ready lines)
- **Length matching** — distractors match target word length
- **Frequency filtering** — candidate pools built from Zipf-frequency dictionaries with configurable floors
- **German noun casing** — automatic post-processing capitalizes German nouns (via spaCy POS tagging)
- **Arabic diacritics handling** — tashkeel stripped for consistent frequency lookups and candidate matching
- **Quality assessment** — built-in `assess_output.py` validates placeholder policy, word form, length, and surprisal margins

## CLI Usage

```
python distract.py -i INPUT -o OUTPUT [-p PARAMS] [-f {delim,ibex}]
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `-i` | yes | — | Input file (semicolon-delimited) |
| `-o` | yes | — | Output file path |
| `-p` | no | `params.txt` | Parameters file |
| `-f` | no | `delim` | Output format: `delim` or `ibex` |

## Input Format

Semicolon-delimited text with columns: `tag`, `id`, `sentence`, and optional `labels`.

```
sample;1;The cat sat on the mat.
sample;2;Die Katze saß auf der Matte.;0 1 2 3 4 5
sample;3;القطة جلست على السجادة
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_delta` | *(required)* | Surprisal margin over target word |
| `min_abs` | *(required)* | Absolute surprisal floor |
| `num_to_test` | *(required)* | Candidate pool size per position |
| `force_max_surprisal` | `False` | `True` = rank by implausibility; `False` = threshold-first |
| `enforce_length_match` | `True` | Require same-length distractors |
| `first_token_placeholder` | `True` | Use `x-x-x` placeholder for sentence-initial position |
| `apply_postcase` | auto | German noun casing post-processing (auto-enabled for German configs) |

See the [Config Reference](https://github.com/mohamedsaid2710/Distractor_software-/wiki/Config-Reference) for the full parameter catalog.

## Selection Behavior

1. Candidate pools are built using target length and frequency ranges.
2. If the pool is too small, the search widens in both directions (lower and higher frequency bins).
3. Final distractor choice is surprisal-driven (mode-dependent) — frequency constrains the pool but does not directly pick the winner.

## Quality Assessment

```bash
python assess_output.py -i English_sample.txt -o output_en.txt -p params.txt --min-delta 0 --strict
```

## Documentation

Full usage guide, configuration reference, and troubleshooting:

- [Wiki](https://github.com/mohamedsaid2710/Distractor_software-/wiki)

## License

See the original [Maze repository](https://github.com/vboyce/Maze) for licensing information.
