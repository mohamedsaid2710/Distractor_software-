import logging
import csv
import re
import os
from sentence_set import Sentence, Sentence_Set


def read_input(filename):
    """Read input file and return a dict of `Sentence_Set` objects keyed by item id.

    Tokenization uses a Unicode-aware regex to capture word tokens (including German characters)
    so that punctuation (commas, periods) does not create extra tokens and label counts match.
    """
    all_sentences = {}
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    # auto-detect delimiter (semicolon or comma) and optional header
    with open(filename, 'r', encoding='utf-8') as f:
        sample = f.read(4096)
        f.seek(0)
        delim = ';'
        has_header = False
        try:
            # prefer semicolon when present, but allow the sniffer to choose
            dialect = csv.Sniffer().sniff(sample, delimiters=';,')
            delim = dialect.delimiter
            has_header = csv.Sniffer().has_header(sample)
        except Exception:
            # fallback heuristic
            delim = ';' if ';' in sample else ','
            try:
                has_header = csv.Sniffer().has_header(sample)
            except Exception:
                has_header = False

        reader = csv.reader(f, delimiter=delim, quotechar='"')
        
        # Smart header skipping: don't trust Sniffer blindly.
        # If Sniffer says header, we check the first row. If the 'id' column is a digit,
        # it's likely a false positive (data row), so we process it.
        # We handle this by not using next(reader) but rather iterating and checking row 1.
        
        first_row_skipped = False
        if has_header:
            # Peek at the first row without consuming it irrevocably from the loop
            try:
                pos = f.tell()
                first_row = next(reader)
                if len(first_row) >= 2 and first_row[1].strip().isdigit():
                    # It looks like data (ID is a number). Reset and don't skip.
                    f.seek(pos)
                    # We need to re-create reader because seek resets file ptr but reader might have buffer
                    # Note: csv.reader doesn't support seek directly well, usually better to re-create
                    reader = csv.reader(f, delimiter=delim, quotechar='"')
                    first_row_skipped = False
                else:
                    # It looks like a real header (not a digit). We successfully skipped it.
                    first_row_skipped = True
            except StopIteration:
                pass
            except Exception:
                 # If we can't read or seek, just proceed
                 pass
                 
        for ln_no, row in enumerate(reader, start=1):
            if len(row) < 3:
                logging.error("Bad input line %d: %s", ln_no, row)
                raise ValueError(f"Bad input line {ln_no}: expected >=3 fields, got {len(row)}")
            tag = row[0]
            id = row[1]
            word_sentence = row[2]
            # Tokenize: capture contiguous letter sequences (including ÄÖÜäöüß and accented letters)
            words = re.findall(r"[A-Za-zÄÖÜäöüßÀ-ÖØ-öø-ÿ]+", word_sentence, flags=re.UNICODE)
            if len(row) > 3 and row[3].strip() != "":
                label_sentence = row[3]
                labels = label_sentence.split()
                if len(labels) != len(words):
                    if len(labels) == 0:
                        labels = list(range(0, len(words)))
                    else:
                        logging.error("Labels are wrong length for sentence %s (line %d)", word_sentence, ln_no)
                        raise ValueError(f"Labels are wrong length for sentence on line {ln_no}")
            else:
                labels = list(range(0, len(words)))
            if id not in all_sentences:
                all_sentences[id] = Sentence_Set(id)
            all_sentences[id].add(Sentence(words, labels, id, tag, word_sentence))
    return all_sentences
