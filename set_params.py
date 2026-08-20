import csv
import logging
import ast


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
    with open(file, 'r') as f:
        reader = csv.reader(f, delimiter=":", quotechar='"')
        for lineno, row in enumerate(reader, start=1):
            if not row:
                continue

            key = row[0].strip()
            if (not key) or key.startswith('#'):
                continue

            if len(row) < 2:
                logging.warning("Skipping malformed params line %d in %s: %r", lineno, file, row)
                continue

            # Keep content after first ":" so values can contain colons.
            raw = ":".join(row[1:]).strip()
            params[key] = _parse_value(raw)
    # Check required parameters
    if params.get('min_delta', None) is None:
        logging.error("Min delta must be provided")
        raise ValueError
    if params.get('min_abs', None) is None:
        logging.error("Min abs must be provided")
        raise ValueError
    if params.get('num_to_test', None) is None:
        logging.error("num to test must be provided")
        raise ValueError
    return params
