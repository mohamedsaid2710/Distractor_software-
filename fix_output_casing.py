#!/usr/bin/env python3
"""Post-process to enforce German noun capitalization on distractors.

Rules:
- If a distractor word is a noun (identified during dictionary construction
  via 3-context majority-vote spaCy tagging), Titlecase it.
- Otherwise, lowercase the distractor.
- Preserve x-placeholders (`x`, `x-x`, `x-x-x`, ...) and punctuation.
"""
import os
import re
import tempfile
import shutil

from wordfreq_distractor import wordfreq_German_zipf_dict

X_PLACEHOLDER_RE = re.compile(r"^x(?:-x)*$", re.IGNORECASE)


def split_punct(token):
    m = re.match(r"^(?P<prefix>[^\wÄÖÜäöüß]*)(?P<body>[\wÄÖÜäöüß'-]+)(?P<suffix>[^\wÄÖÜäöüß]*)$", token, re.UNICODE)
    if not m:
        return '', token, ''
    return m.group('prefix'), m.group('body'), m.group('suffix')


def process_file(infile='test_output.txt', outfile='test_output_fixed.txt'):
    """Safely post-process a delim output file, enforcing German noun casing.

    - Supports in-place operation when infile == outfile by writing to a temp
      file and atomically replacing the destination.
    - Capitalizes distractor words that are nouns (per the dictionary's
      case_map, built via spaCy majority-vote tagging during dictionary init).
    - Preserves x-placeholders and punctuation.
    """
    # Load dictionary — its __init__ batch-tags all words with spaCy
    # to build the noun case_map.
    d = wordfreq_German_zipf_dict({'exclude_words': os.path.join(os.path.dirname(__file__), 'exclude_de.txt'),
                                    'include_words': None})

    # Decide output path (avoid truncating when doing in-place process)
    same_path = os.path.abspath(infile) == os.path.abspath(outfile)
    temp_path = None
    out_path = outfile
    if same_path:
        fd, temp_path = tempfile.mkstemp(prefix='maze_casing_', suffix='.tmp', dir=os.path.dirname(outfile) or None)
        os.close(fd)
        out_path = temp_path


    with open(infile, 'r', encoding='utf-8') as fin, open(out_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.rstrip('\n')
            if not line.strip():
                fout.write('\n')
                continue
            parts = line.split(';')
            if len(parts) < 4:
                fout.write(line + '\n')
                continue

            distractor_field = parts[3]
            toks = distractor_field.split()
            newtoks = []
            for tok in toks:
                prefix, body, suffix = split_punct(tok)
                if body and X_PLACEHOLDER_RE.fullmatch(body):
                    newtoks.append(prefix + body.lower() + suffix)
                    continue
                if not body:
                    newtoks.append(tok)
                    continue

                # Decide casing based on the DISTRACTOR word's own POS.
                # Primary source: pos_cache (built by SpaCy large batch
                # tagging during get_potential_distractors).
                # Secondary: get_titlecase_variant (lazy single-word eval).
                if body.upper() in {'USA', 'EU', 'UN', 'CDU', 'SPD', 'FDP', 'NATO', 'EZB'}:
                    new_body = body.upper()
                else:
                    is_noun = False
                    b_lower = body.lower()
                    # Primary: check SpaCy batch-tagged pos_cache
                    if hasattr(d, 'pos_cache') and b_lower in d.pos_cache:
                        is_noun = d.pos_cache[b_lower] in ('NOUN', 'PROPN')
                    else:
                        # Secondary: get_titlecase_variant (lazy eval)
                        tv = d.get_titlecase_variant(body)
                        is_noun = tv is not None and isinstance(tv, str)
                    if is_noun:
                        # Noun → Titlecase
                        new_body = body[0].upper() + body[1:].lower()
                    else:
                        # Not a noun → lowercase
                        new_body = body.lower()
                newtoks.append(prefix + new_body + suffix)
            parts[3] = ' '.join(newtoks)
            fout.write(';'.join(parts) + '\n')

    # If in-place was requested, atomically replace
    if same_path and temp_path is not None:
        try:
            shutil.move(temp_path, outfile)
        finally:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--infile', default='test_output.txt')
    p.add_argument('--outfile', default='test_output_fixed.txt')
    args = p.parse_args()
    process_file(args.infile, args.outfile)
    print('Wrote', args.outfile)
