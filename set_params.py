import csv
import logging
import ast

def set_params(file):
    """Takes a colon delimited file specifying various parameters,
    returns dictionary format of those parameters"""
    params = {}
    with open(file, 'r') as f:
        reader = csv.reader(f, delimiter=":", quotechar='"')
        for row in reader:
            if row != []:
                if row[0].startswith('#'):
                    pass
                else:
                    raw = row[1].strip()
                    low = raw.lower()
                    # Accept common non-Python literals (true/false/null) used in params files
                    if low == 'true':
                        params[row[0]] = True
                    elif low == 'false':
                        params[row[0]] = False
                    elif low in ('null', 'none'):
                        params[row[0]] = None
                    else:
                        try:
                            params[row[0]] = ast.literal_eval(raw)
                        except Exception:
                            # Fallback: keep as raw string (no surrounding quotes expected)
                            params[row[0]] = raw.strip('"')
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
