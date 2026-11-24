import csv
import ast
import difflib


# Languages the pipeline supports.  Adding a language means adding its code
# here too (see wiki/Adding-a-Language.md).
LANGUAGES = {"en": "English", "de": "German", "ar": "Arabic"}


# Every key a params file may set, i.e. every key the pipeline reads, with
# the kind of value it takes.  A key that is not listed stops the run: a
# misspelled key (`min_pool_sise`) used to be read without complaint and never
# used, so the run went ahead on the built-in default.  A value of the wrong
# kind stops it too: the code turns switches on with bool(), which is True for
# ANY text, so `semantic_filter: no` or `"False"` meant ON.
# When you add a setting to the code, add it here too;
# tests/test_params_keys.py fails if the code reads a key missing from this set.
#
# Kinds: "bool" True/False; "int" whole number; "number" int or decimal;
# "str" text.  A trailing "?" means null/None is also allowed.
KEY_TYPES = {
    # required
    "language": "str", "min_delta": "number", "min_abs": "number",
    "num_to_test": "int",
    # wiring: optional overrides; by default the pipeline picks the classes
    # for `language` itself (pipeline.LANGUAGE_WIRING)
    "dictionary_loc": "str", "dictionary_class": "str",
    "model_loc": "str", "model_class": "str",
    "threshold_loc": "str", "threshold_name": "str",
    # language model (scoring/hf_scorer.py)
    "hf_model_name": "str", "device": "device?", "context_window": "int?",
    "model_batch_size": "int",
    
    # run control (pipeline.py)
    "seed": "int?", "write_report": "bool", 
    "max_repeat": "int", "repeat_exempt_max_len": "int",
    # selection (selection.py)
    "allow_banned_fallback": "bool",
    "chunk_size": "int",
    "early_position_boost": "number", "early_position_count": "int",
    "enforce_length_match": "bool", "exclude_propn_candidates": "bool",
    "exclude_words": "str?", "first_token_placeholder": "bool",
    "len_tolerance": "int",
    "pos_match": "int", "pos_min_share": "number",
    "min_abs_short": "number",
    "short_word_boost": "number", "short_word_max_len": "int",
    "short_word_freq_match": "bool", "short_word_band_floor": "number",
    "use_gpu": "bool",
    # candidate pools (candidates.py)
    "freq_tolerance": "number?", "freq_widen_step": "number",
    "max_freq_widen": "int", "min_pool_size": "int",
    "min_word_len": "int",
    "min_zipf": "number", "short_word_min_zipf": "number",
    "include_words": "str?",
    "lowercase_only": "bool",
    "nlp_batch_size": "int",
    
    # semantic filter (semantic_filter.py)
    "semantic_filter": "bool", "semantic_threshold": "number",
}
KNOWN_KEYS = frozenset(KEY_TYPES)

_KIND_WORDS = {
    "bool": "True or False",
    "int": "a whole number",
    "number": "a number",
    "str": "text",
    "device": "a device name (e.g. \"cuda\", \"cpu\") or a GPU index",
}


def _type_error(key, value):
    """Return a message if `value` is not the kind KEY_TYPES expects, else None."""
    kind = KEY_TYPES[key]
    nullable = kind.endswith("?")
    kind = kind.rstrip("?")
    if value is None:
        return None if nullable else f"{key} must be {_KIND_WORDS[kind]}, got null"
    is_bool = isinstance(value, bool)
    ok = {
        "bool": is_bool,
        "int": isinstance(value, int) and not is_bool,
        "number": isinstance(value, (int, float)) and not is_bool,
        "str": isinstance(value, str),
        "device": isinstance(value, str) or (isinstance(value, int) and not is_bool),
    }[kind]
    if ok:
        return None
    msg = f"{key} must be {_KIND_WORDS[kind]}" + (" or null" if nullable else "")
    msg += f", got {value!r}"
    if (kind in ("number", "int") and isinstance(value, tuple) and len(value) == 2
            and all(isinstance(x, int) for x in value)):
        msg += f" -- use a dot for decimals: {value[0]}.{value[1]}"
    elif kind in ("number", "int") and isinstance(value, str):
        try:
            float(value)
            msg += " -- remove the quotes"
        except ValueError:
            pass
    return msg


# Keys that used to exist, with what replaced them.
REMOVED_KEYS = {
    "apply_postcase": "was removed; each distractor's casing is set when it is "
                      "chosen (German nouns capitalised from their own POS)",
    "pos_overrides": "was removed; POS comes from the norms-built lexicon "
                     "(rebuild it with tools/build_lexicon.py)",
    "match_noun_pos": "was replaced by pos_match (0 off, 1 noun/non-noun, 2 class/class)",
    "match_casing_only": "was removed; German casing now follows the distractor's own POS. "
                         "Use pos_match: 1 to keep nouns and non-nouns apart",
    "force_max_surprisal": "was removed; the most implausible candidate is always chosen. "
                           "Delete the line",
    "match_surprisal": "was removed; the most implausible candidate is always chosen. "
                       "Delete the line",
    "pool_size_factor": "was removed; the rescue search now fetches min_pool_size "
                        "words. Delete the line",
    "proactive_tagging": "was removed (candidate POS comes from the lexicon). Delete the line",
    "spacy_batch_size": "was renamed nlp_batch_size",
    "batch_size": "was renamed model_batch_size",
    "json_min_zipf": "was removed; use min_zipf",
    "absolute_threshold_only": "was removed; the threshold is always "
                               "max(min_abs, target surprisal + min_delta). Delete the line",
}

# Written into a copy of params per target word by selection.py to pass
# state to the threshold function.  Any value set in a params file would be
# overwritten, so setting one there is a mistake.
RUNTIME_KEYS = frozenset({
    "target_zipf", "target_is_noun", "target_is_function_word", "target_exact_length",
})


class ParamsError(ValueError):
    """A problem with the params file itself (missing key, bad value, a
    model/dictionary/threshold name that does not resolve).  Subclasses
    ValueError so existing callers that catch ValueError keep working."""


def _parse_value(raw):
    """Parse one params value.

    ast.literal_eval is tried first because Python's own tokenizer strips a
    trailing `# comment` for us, which is how `device: "cuda"  # ...` has
    always worked.  The true/false/null spellings are NOT Python literals, so
    they used to be compared against the whole raw string and an inline comment
    silently turned them into the string "null    # ...".  Every other value
    type accepted comments; these three did not.
    """
    try:
        return ast.literal_eval(raw)
    except Exception:
        pass

    # Not a Python literal: strip an inline comment and try the JSON-ish
    # spellings, then a plain string.
    stripped = raw.split('#', 1)[0].strip()
    low = stripped.lower()
    if low == 'true':
        return True
    if low == 'false':
        return False
    if low in ('null', 'none'):
        return None
    try:
        return ast.literal_eval(stripped)
    except Exception:
        return stripped.strip('"')


def set_params(file):
    """Takes a colon delimited file specifying various parameters,
    returns dictionary format of those parameters"""
    params = {}
    seen_on = {}
    with open(file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=":", quotechar='"')
        for row in reader:
            lineno = reader.line_num
            if not row:
                continue

            key = row[0].strip()
            if (not key) or key.startswith('#'):
                continue

            where = f"{file}, line {lineno}"
            if len(row) < 2:
                # Used to be a warning, so `min_pool_size 30` (no colon) was
                # skipped and the default used.
                raise ParamsError(f"{where}: expected 'key: value', got {':'.join(row)!r}")
            if key in RUNTIME_KEYS:
                raise ParamsError(
                    f"{where}: '{key}' is set by the pipeline for each target word "
                    f"and cannot be set in a params file")
            if key in REMOVED_KEYS:
                raise ParamsError(f"{where}: '{key}' {REMOVED_KEYS[key]}")
            if key not in KNOWN_KEYS:
                close = difflib.get_close_matches(key, KNOWN_KEYS, n=1, cutoff=0.75)
                hint = f" (did you mean '{close[0]}'?)" if close else ""
                raise ParamsError(f"{where}: unknown key '{key}'{hint}")
            if key in seen_on:
                # The later value used to win silently.
                raise ParamsError(
                    f"{where}: '{key}' is already set on line {seen_on[key]}")
            seen_on[key] = lineno

            # Keep content after first ":" so values can contain colons.
            raw = ":".join(row[1:]).strip()
            value = _parse_value(raw)
            problem = _type_error(key, value)
            if problem:
                raise ParamsError(f"{where}: {problem}")
            params[key] = value
    # Check required parameters
    for key in ('min_delta', 'min_abs', 'num_to_test'):
        if params.get(key, None) is None:
            raise ParamsError(f"{file}: required key '{key}' is missing or empty")

    # `language` used to be optional, and the five places that read it each
    # fell back to something different (en, de, de, "" and a guess from the
    # class names), so a params file without it ran with the pipeline
    # disagreeing with itself about the language.
    lang = params.get('language', None)
    if lang is None:
        raise ParamsError(f"{file}: required key 'language' is missing "
                          f"(one of: {', '.join(LANGUAGES)})")
    lang = str(lang).strip().lower()
    if lang not in LANGUAGES:
        raise ParamsError(f"{file}: language must be one of "
                          f"{', '.join(LANGUAGES)}, got {params['language']!r}")
    params['language'] = lang

    if params.get('pos_match', 0) not in (0, 1, 2):
        raise ParamsError(f"{file}: pos_match must be 0 (off), 1 (noun/non-noun) "
                          f"or 2 (class/class), got {params['pos_match']!r}")
    if not 0 <= params.get('pos_min_share', 0) <= 1:
        raise ParamsError(f"{file}: pos_min_share must be between 0 and 1, "
                          f"got {params['pos_min_share']!r}")

    # Word lists named in a params file are found next to it first, so
    # config/params_de.txt with `exclude_words: "exclude_de.txt"` works from
    # any working directory.
    from maze_automate.utils import resolve_repo_path
    import os
    base = os.path.dirname(os.path.abspath(file))
    for key in ("exclude_words", "include_words"):
        if params.get(key):
            params[key] = resolve_repo_path(params[key], base=base)
    return params
