import logging
import csv
import os
from maze_automate.selection import Sentence, Sentence_Set
from maze_automate.utils import strip_punct


class InputFormatError(ValueError):
    """The input sentence file is malformed.  The message names the file and
    the line, so the CLI can report it without a traceback.  Subclasses
    ValueError so existing callers that catch ValueError keep working."""


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


def is_header_row(row):
    """True only for a real header line: columns named like ``tag;id;sentence``.

    Header detection used to be csv.Sniffer's guess, and when it guessed
    "header" any first line whose id was not a number was dropped.  A file
    starting with ``practice;p1;...`` followed by ``exp;e1;...`` lost p1
    without a word, because the first tag differed in length from the rest.
    """
    return (len(row) >= 3
            and row[1].strip().lower() == "id"
            and row[2].strip().lower().startswith("sentence"))


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
        raise FileNotFoundError(f"input file not found: {filename}")
    # auto-detect delimiter (semicolon or comma); a header row is skipped
    # only if it is named like tag;id;sentence (see is_header_row)
    with open(filename, 'r', encoding='utf-8') as f:
        sample = f.read(4096)
        f.seek(0)
        delim = detect_delimiter(sample)
        reader = csv.reader(f, delimiter=delim, quotechar='"')

        for row in reader:
            # reader.line_num is the physical line in the file (a skipped
            # header counts), unlike a plain row counter.
            ln_no = reader.line_num
            if ln_no == 1 and is_header_row(row):
                continue
            if len(row) < 3:
                raise InputFormatError(
                    f"{filename}, line {ln_no}: expected at least 3 fields "
                    f"(tag;id;sentence[;labels]), got {len(row)}: {row!r}")
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
                        raise InputFormatError(
                            f"{filename}, line {ln_no}: {len(labels)} labels for "
                            f"{len(words)} words in sentence {word_sentence!r}")
            else:
                labels = list(range(0, len(words)))
            if id not in all_sentences:
                all_sentences[id] = Sentence_Set(id)
            try:
                all_sentences[id].add(Sentence(words, labels, id, tag, word_sentence))
            except ValueError as e:
                # Label-rule violations raised by selection.py; add where.
                raise InputFormatError(f"{filename}, line {ln_no}: {e}") from e
    return all_sentences
