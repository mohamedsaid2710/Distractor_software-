"""Small shared helpers: path resolution, punctuation handling, de-duplication."""

import os
import re
import unicodedata


# src/maze_automate/utils.py -> the package folder, and the repo root two up.
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(PACKAGE_DIR))
CONFIG_DIR = os.path.join(REPO_ROOT, "config")
LEXICON_DIR = os.path.join(PACKAGE_DIR, "lexicons")


def lexicon_path(lang):
    """The shipped POS lexicon for `lang` (src/maze_automate/lexicons/<lang>.json)."""
    return os.path.join(LEXICON_DIR, f"{lang}.json")


def resolve_repo_path(path, base=None):
    """Find a file named in a params file (e.g. "exclude_de.txt").

    Looked up, in order: as given (absolute, or relative to the working
    directory); next to the params file (`base`); in the repo's config/
    folder; at the repo root.  Returns *path* unchanged if none exists, so
    callers keep their own missing-file handling.
    """
    if not path:
        return path
    if os.path.isabs(path) or os.path.exists(path):
        return path
    for folder in (base, CONFIG_DIR, REPO_ROOT):
        if folder:
            candidate = os.path.join(folder, path)
            if os.path.exists(candidate):
                return candidate
    return path


def _is_word_char(ch):
    """True for anything that is part of a word, not surrounding punctuation.

    ``str.isalnum()`` alone is wrong for Arabic: harakat (tashkeel) are
    Unicode category ``Mn`` (non-spacing mark), so ``isalnum()`` is False for
    them and a vocalised word such as ``الْكِتَابُ`` looks like it ends in
    punctuation.  The trailing damma then gets stripped off the target (making
    it one character "shorter" for length matching) and grafted onto the
    distractor.  Combining marks count as word characters here.
    """
    return ch.isalnum() or unicodedata.category(ch) == 'Mn'


def _word_span(word):
    """Return (start, end) indices of the word body, or None if there is none."""
    start = None
    for i, ch in enumerate(word):
        if _is_word_char(ch):
            start = i
            break
    if start is None:
        return None
    end = start
    for j in range(len(word) - 1, start - 1, -1):
        if _is_word_char(word[j]):
            end = j
            break
    return start, end


def strip_punct(word):
    '''take a word, return word with start and end punctuation removed'''
    span = _word_span(word)
    if span is None:
        # All punctuation (or empty).  The old loop-and-slice version raised
        # UnboundLocalError here because `i`/`j` were never assigned.
        return ''
    start, end = span
    return word[start:end + 1]


def copy_punct(word, distractor):
    """Copy leading/trailing punctuation from *word* onto *distractor*.

    Casing is NOT modified — that is handled separately by the
    language-specific normalization functions.
    """
    span = _word_span(word)
    if span is None:
        return distractor
    start, end = span
    return word[:start] + distractor + word[end + 1:]


# Arabic short-vowel marks (tashkeel), Quranic marks and tatweel (U+0640).
# Only Arabic script carries them, so removing them changes nothing in
# English or German text.
_ARABIC_MARKS_RE = re.compile(
    r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8'
    r'\u06EA-\u06ED\u0640]')


def strip_arabic_marks(text):
    """`text` without Arabic diacritics and tatweel."""
    return _ARABIC_MARKS_RE.sub('', text)


def bare_form(word):
    """The word as letters: punctuation stripped, and Arabic diacritics too.

    Use this wherever a word is measured or compared -- length, "is this the
    target", lexicon lookups.  A vowelled الْكِتَابُ is the 6-letter word
    الكتاب; counted with its marks it was 10 letters long, never matched its
    own lexicon entry, and was not recognised as the target.
    """
    return strip_arabic_marks(strip_punct(word))


def ordered_unique(items):
    """De-duplicate while preserving first-seen order.

    ``list(set(...))`` on strings is ordered by hash, and CPython randomises
    string hashing per process, so any downstream ``[:n]`` slice or shuffle
    over such a list silently draws a different candidate set on every run.
    """
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
