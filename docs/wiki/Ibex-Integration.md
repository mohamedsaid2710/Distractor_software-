# Ibex Integration

Yes, this project supports Ibex directly.

## Generate Ibex Items

```bash
python distract.py -i examples/input.txt -o output_ibex.txt -p config/params.txt -f ibex
```

## Output Shape

Each generated line is:

```text
[["tag", 'id'], "Maze", {s:"<sentence>", a:"<distractor sentence>"}],
```

Field meanings:

- `tag`: condition from input column 1
- `id`: item id from input column 2
- `s`: source sentence
- `a`: distractor sentence

## Paste Into Ibex

```javascript
var items = [
  // paste generated lines from output_ibex.txt
];
```

## Operational Notes

- Lines are comma-terminated by design.
- Double quotes in sentence text are escaped.
- UTF-8 text is preserved.
- `x-x-x` is used as placeholder at distractor position 0.
