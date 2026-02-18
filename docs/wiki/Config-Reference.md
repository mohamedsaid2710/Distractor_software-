# Config Reference

Main parameter files:

- English: `config/params.txt`
- German: `config/params_de.txt`

## Core Parameters

- `min_delta`: minimum surprisal increase target
- `min_abs`: minimum absolute surprisal target
- `num_to_test`: number of candidate distractors to evaluate
- `dictionary_loc`, `dictionary_class`: candidate dictionary source
- `threshold_loc`, `threshold_name`: thresholding function
- `model_loc`, `model_class`: model adapter module/class
- `hf_model_name`: HF repo id or local model directory
- `exclude_words`: blacklist file
- `include_words`: optional allow-list file
- `max_repeat`: max distractor repeats per set
- `apply_postcase`: enable/disable post-case processing

## Defaults in This Repo

English defaults (`config/params.txt`):

- model: `EnglishScorer`
- model id: `openai-community/gpt2-medium`
- threshold: `get_thresholds_en`

German defaults (`config/params_de.txt`):

- model: `GermanScorer`
- model path: `models/dbmdz-german-gpt2`
- threshold: `get_thresholds`
