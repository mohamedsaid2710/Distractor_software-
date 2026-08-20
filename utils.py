"""Small shared helpers: path resolution, punctuation handling, de-duplication."""

import os
import unicodedata


# utils.py lives at the repo root, so this is the repo root.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def resolve_repo_path(path):
    """Resolve a bare filename from a params file against the repo root.

    Params files name their word lists as bare filenames ("exclude_de.txt"),
    which only resolve while the process CWD happens to be the repo root.  The
    three places that used to do this by hand disagreed: one had an extra
    `dirname` and resolved to the repo's *parent*, so the exclude list was
    silently dropped -- 4.9 KB of curated German exclusions with no warning --
    and the two exclusion checks inside a single run could disagree about which
    words were banned.

    Returns *path* unchanged if it is absolute, already exists, or cannot be
    found at the repo root either (so callers keep their own missing-file
    handling).
    """
    if not path:
        return path
    if os.path.isabs(path) or os.path.exists(path):
        return path
    candidate = os.path.join(REPO_ROOT, path)
    return candidate if os.path.exists(candidate) else path


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
