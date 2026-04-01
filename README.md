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
python distract.py -i English_sample.txt -o output_en.txt -p params_en.txt -f delim
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

- **Three languages** — English (`gpt2-medium`), German (`benjamin/gerpt2`), and Arabic (`aubmindlab/aragpt2-medium`)
- **Two selection modes** — threshold-first (Mode A) or max-implausibility ranking (Mode B)
- **Two output formats** — `delim` (semicolon-delimited table) and `ibex` (PCIbex-ready lines)
- **Length matching** — distractors match target word length (with optional `len_tolerance`)
- **POS-Aware Noun Matching** — (German) target nouns receive noun distractors for grammatical capitalization
- **Frequency Matching** — distractors fall within a tight Zipf frequency band (`freq_tolerance`)
- **Proper-noun filtering** — `exclude_propn_candidates` uses spaCy POS tags (English and German)
- **German noun casing** — automatic post-processing capitalizes German nouns (via spaCy POS tagging)
- **Arabic diacritics handling** — tashkeel stripped for consistent frequency lookups and candidate matching
- **Quality assessment** — built-in `assess_output.py` validates placeholder policy, word form, length, and surprisal margins
- **Set-level distractor reuse** — distractors are chosen once per label within an item-set and reused across its condition variants; this preserves controlled comparisons (e.g., Latin-square style designs) and reduces compute time

## German "Gold Standard" Configuration

For maximum accuracy and linguistic control in German experiments, use the following validated "Gold Standard" parameters in your `params_de.txt`:

- **`min_abs: 15` / `min_delta: 8`**: The proven baseline for target/distractor surprisal gaps.
- **`len_tolerance: 1`**: Allows $\pm 1$ character matching, critical for German compound words.
- **`force_max_surprisal: True`**: (Mode B) Ensures the absolute worst-fitting word is chosen as the distractor.
- **`num_to_test: 1000`**: High-quality candidate pool for better scoring.
- **`freq_tolerance: 0.2`** & **`min_zipf: 3.0`**: Stricter frequency bounds for natural-sounding distractors.
- **`max_freq_widen: 15`**: Generous fallback to prevent "wort" placeholders.

## Performance / Batch Mode

The German Maze pipeline (and future models) now supports **Parallel Batch Scoring**. Instead of calculating surprisals for one candidate word at a time, the model processes **256 candidates simultaneously** on your GPU or CPU.

### **Speed Features:**
- **Hyper-Speed Execution**: Reduces generation time from hours to minutes for large stimulus sets.
- **Focused Context Window (64 tokens)**: Intelligently limits history look-back to ensure constant high performance, even in long sentences.
- **GPU-Accelerated**: Automatically detects and utilizes NVIDIA CUDA for maximum throughput.

## Usage

```
python distract.py [INPUT] [OUTPUT] [PARAMS] [-f {delim,ibex}]
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `INPUT`  | yes | — | Input file (or use `-i`) |
| `OUTPUT` | yes | — | Output file (or use `-o`) |
| `PARAMS` | no | `params_en.txt` | Configuration file (or use `-p`) |
| `-f` | no | `delim` | Output format (`delim` or `ibex`) |

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
| `force_max_surprisal` | `False` | `True` = rank by implausibility (Mode B); `False` = threshold-first (Mode A). **Note:** Both modes now strictly respect `min_delta` and `min_abs` before selecting a candidate. |
| `enforce_length_match` | `True` | Require character-count consistency for distractors |
| `len_tolerance` | `0` | Allowed character-length difference between target and distractor (e.g. `1` allows target ±1) |
| `first_token_placeholder` | `True` | Use `x-x-x` placeholder for sentence-initial position |
| `early_position_boost` | `0` | Extra surprisal for early positions (reduces plausible distractors at sentence start) |
| `match_noun_pos` | `False` | (German-specific) If `True`, distractors for target nouns must also be nouns. |
| `apply_postcase` | auto | German noun casing post-processing (auto-enabled for German configs). |

See the [Config Reference](https://github.com/mohamedsaid2710/Distractor_software-/wiki/Config-Reference) for the full parameter catalog.

## Selection Behavior

1. Candidate pools are built using target length and frequency ranges.
2. If the pool is too small, the search widens in both directions (lower and higher frequency bins).
3. Final distractor choice is surprisal-driven (mode-dependent) — frequency constrains the pool but does not directly pick the winner.
4. Within one item-set (same `id`), distractors are selected per label once and then reused across condition rows. This is intentional to keep lexical confounds stable across conditions while lowering model-scoring cost.

## A Note on Sub-word Tokenization

GPT-2 uses Byte Pair Encoding (BPE), which often splits a single word into multiple sub-word tokens (e.g., *"unbelievable"* → `["un", "believ", "able"]`). This software handles multi-token words correctly: it computes the **joint surprisal across all sub-tokens** by summing the conditional log-probabilities of each sub-token given its full left context (including prior sub-tokens of the same word). Formally, this applies the chain rule of probability:

**−log₂ P(word | context) = −log₂ P(t₁ | ctx) − log₂ P(t₂ | ctx, t₁) − … − log₂ P(tₖ | ctx, t₁, …, tₖ₋₁)**

This means the sub-word splitting introduces **no information loss or approximation errors** in the surprisal calculation.

## Quality Assessment

```bash
python assess_output.py -i output_en.txt -o output_en_assessed.txt -p params_en.txt --min-delta 0 --strict
```

> **Tip:** The fastest way to improve distractor quality for any language is to
> expand the exclude file (`exclude_en.txt`, `exclude_de.txt`, `exclude_ar.txt`).
> Review output, spot bad words, and add them — the list is meant to grow over time.

## Documentation

Full usage guide, configuration reference, and troubleshooting:

- [Wiki](https://github.com/mohamedsaid2710/Distractor_software-/wiki)

## License

MIT License. See [LICENSE](LICENSE) for details.

Based on the original [Maze repository](https://github.com/vboyce/Maze) by Victoria Boyce.
