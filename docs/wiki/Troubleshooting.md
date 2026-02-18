# Troubleshooting

## English Model Load Error

If you see:

`ERROR: Failed to load English model 'openai-community/gpt2-medium'...`

Do:

1. Download the model:

```bash
python download_model.py --english
```

2. If running offline, set in `config/params.txt`:

```txt
hf_model_name: "models/openai-community-gpt2-medium"
```

## GitHub Push Rejected for Large Files

Large model weights must not be committed.

- Keep `.gitignore` as-is.
- Never commit `*.bin`, `*.safetensors`, large model blobs.

## Distractor Quality Too Noisy

Tune:

- `min_delta`
- `min_abs`
- `num_to_test`
- `exclude_words`
- `include_words` (optional curated allow-list)

## Ibex Output Not Parsing

Check:

- You used `-f ibex`.
- You pasted output lines inside the `items` array.
- You kept trailing commas between item lines.
