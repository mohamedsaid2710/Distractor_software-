# Usage

## Install

```bash
pip install -r requirements.txt
```

## Download Models

```bash
python download_model.py --english
python download_model.py --german
# or
python download_model.py --all
```

Downloaded model locations:

- English: `models/openai-community-gpt2-medium`
- German: `models/dbmdz-german-gpt2`

## Run Commands

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

## Input Format

Accepted input is CSV/semicolon-delimited text with:

1. `tag`
2. `id`
3. `sentence`
4. optional labels column (`0 1 2 ...`)

Example:

```csv
sample;1;The cat sat on the mat.;0 1 2 3 4 5
sample;2;Maria bought flowers for Sunday.;0 1 2 3 4
```

Notes:

- Header detection is automatic.
- If labels are omitted, token indices are auto-generated.
- Label count must match tokenized word count.
