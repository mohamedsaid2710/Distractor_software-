import csv
import logging
import ast

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
            low = raw.lower()
            # Accept common non-Python literals (true/false/null) used in params files
            if low == 'true':
                params[key] = True
            elif low == 'false':
                params[key] = False
            elif low in ('null', 'none'):
                params[key] = None
            else:
                try:
                    params[key] = ast.literal_eval(raw)
                except Exception:
                    # Fallback: keep as raw string (no surrounding quotes expected)
                    params[key] = raw.strip('"')
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
