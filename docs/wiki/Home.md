# Distractor Software Wiki

Distractor Software generates Maze-style distractor stimuli from sentence inputs.

## Main Features

- English pipeline (default): `openai-community/gpt2-medium`
- German pipeline: `dbmdz/german-gpt2`
- Output formats:
  - `delim` (semicolon-delimited table)
  - `ibex` (Ibex Maze item lines)

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Download model files:

```bash
python download_model.py --english
python download_model.py --german
```

Generate distractors:

```bash
python distract.py -i examples/input.txt -o output_en.csv -p config/params.txt -f delim
python distract.py -i examples/input.txt -o output_ibex.txt -p config/params.txt -f ibex
```

## Documentation Pages

- [[Usage]]
- [[Ibex Integration]]
- [[Config Reference]]
- [[Troubleshooting]]
