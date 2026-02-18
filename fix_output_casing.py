#!/usr/bin/env python3
"""Post-process to enforce noun capitalization on distractors.

Rules:
- If a distractor is POS-tagged as NOUN or PROPN (via spaCy), Titlecase it.
- Otherwise, lowercase the distractor.
- Preserve `x-x-x` and punctuation.
"""
import os
import re
import tempfile
import shutil
try:
    import spacy
except Exception:
    spacy = None

from wordfreq_distractor import wordfreq_German_zipf_dict


def split_punct(token):
    m = re.match(r"^(?P<prefix>[^\wÄÖÜäöüß]*)(?P<body>[\wÄÖÜäöüß'-]+)(?P<suffix>[^\wÄÖÜäöüß]*)$", token, re.UNICODE)
    if not m:
        return '', token, ''
    return m.group('prefix'), m.group('body'), m.group('suffix')


def process_file(infile='test_output.txt', outfile='test_output_fixed.txt'):
    """Safely post-process a delim output file, enforcing German noun casing.

    - Supports in-place operation when infile == outfile by writing to a temp
      file and atomically replacing the destination.
    - Uses spaCy POS (NOUN/PROPN) to Titlecase; otherwise lowercases.
    - Preserves x-x-x and punctuation; prefers dictionary Titlecase variants.
    """
    # spaCy-only backend. If spaCy model missing, continue without POS.
    nlp_sp = None
    if spacy is not None:
        try:
            try:
                nlp_sp = spacy.load('de_core_news_sm')
            except Exception:
                nlp_sp = spacy.load('de_core_news_md')
        except Exception:
            nlp_sp = None
    # load dictionary to obtain titlecase variants when available
    d = wordfreq_German_zipf_dict({'exclude_words': os.path.join(os.path.dirname(__file__), 'exclude.txt'),
                                    'include_words': os.path.join(os.path.dirname(__file__), 'german_data', 'wordfreq_de.tsv')})

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
                if tok == 'x-x-x':
                    newtoks.append(tok)
                    continue
                prefix, body, suffix = split_punct(tok)
                if not body:
                    newtoks.append(tok)
                    continue
                # tag the body with spaCy if available
                # IMPORTANT: tag the LOWERCASE version to avoid SpaCy misclassifying capitalized verbs as proper nouns (e.g., "Kommst" → PROPN instead of VERB)
                 
                upos = None
                if nlp_sp is not None:
                    try:
                        doc = nlp_sp(body.lower())
                        if doc and len(doc) > 0:
                            upos = doc[0].pos_
                    except Exception:
                        upos = None
                # decide casing: POS-first to avoid over-capitalizing function words
                # Handle common acronyms and proper names that should stay uppercase
                if body.upper() in {'USA', 'EU', 'UN', 'CDU', 'SPD', 'FDP', 'NATO', 'EZB'}:
                    new_body = body.upper()
                elif upos in ('NOUN', 'PROPN'):
                    # prefer dictionary titlecase if available, else Titlecase
                    try:
                        tv = d.get_titlecase_variant(body)
                    except Exception:
                        tv = None
                    new_body = tv if tv else body.lower().capitalize()
                else:
                    # Not a noun according to spaCy, OR spaCy is missing.
                    if nlp_sp is None:
                        # Fallback when no POS tagger: rely on the probability dictionary.
                        # If the dictionary says the most frequent form is Titlecased (e.g. "Euro", "März"), use it.
                        # This handles obvious nouns even without spaCy.
                        try:
                            tv = d.get_titlecase_variant(body)
                            new_body = tv if tv else body
                        except Exception:
                            new_body = body  # Leave as is if we can't check
                    else:
                        # We have POS tags, and this is NOT a noun.
                        # Force lowercase to correct any erroneously capitalized non-nouns.
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
