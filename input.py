import logging
import csv
import os
from sentence_set import Sentence, Sentence_Set
from utils import strip_punct


def detect_delimiter(sample):
    """Choose the field delimiter for an input file.

    The documented format is ``tag;id;sentence;labels``.  csv.Sniffer used to
    get the last word here, and on any semicolon file whose sentences contain
    commas -- a German relative clause, an English appositive, a list -- it
    picks ``,``, splits one sentence into three fields, and every row becomes
    its own ``Sentence_Set``.  The row still has >= 3 fields so the guard below
    never fires: no error, no warning, and the one-distractor-per-label
    Latin-square invariant is destroyed.

    So a first line that parses into >= 3 semicolon fields wins outright.
    Sniffing remains the fallback for genuinely comma-delimited files.
    """
    first = next((ln for ln in sample.splitlines() if ln.strip()), '')
    if first:
        semi_fields = next(csv.reader([first], delimiter=';', quotechar='"'), [])
        if len(semi_fields) >= 3:
            return ';'
    try:
        return csv.Sniffer().sniff(sample, delimiters=';,').delimiter
    except Exception:
        return ';' if ';' in sample else ','


def tokenize(word_sentence):
    """Split a stimulus sentence into the tokens a Maze participant will see.

    Whitespace, nothing else.  This has to agree exactly with three other
    counts: the labels column (split on whitespace), the distractor string
    written to the output (space-joined, one per token), and PCIbex's own
    splitting of the sentence at run time.  The Unicode regex this replaces
    agreed with none of them on ordinary input -- a curly apostrophe (Word and
    Google Docs produce one by default) split "doesn't" into two tokens,
    "1,000" into two, and "U.S." into two -- which either raised a spurious
    "labels are wrong length" error or, with generated labels, misaligned word
    k with distractor k for the rest of the sentence.

    Punctuation stays attached to its token; `strip_punct`/`copy_punct` handle
    it downstream, which is where the pipeline already expects to deal with it.
    """
    return word_sentence.split()


def read_input(filename):
    """Read input file and return a dict of `Sentence_Set` objects keyed by item id."""
    all_sentences = {}
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    # auto-detect delimiter (semicolon or comma) and optional header
    with open(filename, 'r', encoding='utf-8') as f:
        sample = f.read(4096)
        f.seek(0)
        delim = detect_delimiter(sample)
        try:
            has_header = csv.Sniffer().has_header(sample)
        except Exception:
            has_header = False

        reader = csv.reader(f, delimiter=delim, quotechar='"')
        
        # Smart header skipping: don't trust Sniffer blindly.
        # If Sniffer says header, we check the first row. If the 'id' column is a digit,
        # it's likely a false positive (data row), so we process it.
        # We handle this by not using next(reader) but rather iterating and checking row 1.
        
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
                else:
                    # It looks like a real header (not a digit). We successfully skipped it.
                    pass
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
            words = tokenize(word_sentence)
            empty = [w for w in words if not strip_punct(w)]
            if empty:
                logging.warning(
                    "Line %d contains token(s) with no word characters (%s); "
                    "they will still consume a distractor slot.", ln_no, empty)
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
