# Distractor Software

This repo is for generating distractors.

## Distractor Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the model used by the default distractor pipeline (English):

```bash
python download_model.py --english
```

Run distractor generation:

```bash
python distract.py -i examples/input.txt -o output.csv -p config/params.txt -f delim
```

Optional German distractors:

```bash
python download_model.py --german
python distract.py -i examples/sample.csv -o output_de.csv -p config/params_de.txt -f delim
```
