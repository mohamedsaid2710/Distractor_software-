#!/usr/bin/env python3
"""Semantic dissimilarity filtering for maze distractors.

With `semantic_filter: True`, a candidate is rejected when the cosine
similarity of its fastText vector and the target's is `semantic_threshold`
or higher, so distractors come from unrelated semantic domains (e.g.
"Musikerin" -> "Stoffwechsel", not "Sängerin").  The threshold is applied as
given; it is never loosened behind the user's back.

Vectors: fastText `cc.<lang>.300.bin`, in ~/.fasttext/ (where
`tools/download_models.py` puts them).  Words are compared as they will be shown:
the vectors are case-sensitive, and in German "Weg" (way) and "weg" (away)
are different words.
"""

import os
import logging
import numpy as np

FASTTEXT_DIR = os.path.expanduser("~/.fasttext")

# Lazy-loaded embedding model
_embedding_model = None
_embedding_lang = None


class SemanticFilterError(RuntimeError):
    """`semantic_filter: True`, but the fastText vectors cannot be loaded."""


def _load_fasttext(lang):
    """The fastText model for `lang`, downloading it into ~/.fasttext if needed.

    Raises SemanticFilterError instead of returning nothing: a filter the
    params file switched on must not silently let every candidate through.
    """
    global _embedding_model, _embedding_lang
    if _embedding_model is not None and _embedding_lang == lang:
        return _embedding_model

    ft_lang = (lang or '').strip().lower()[:2]
    if not ft_lang:
        raise SemanticFilterError("no language given to the semantic filter")
    try:
        import fasttext
        import fasttext.util
    except ImportError as e:
        raise SemanticFilterError(
            f"fasttext is not installed ({e}); run `uv sync`") from e

    path = os.path.join(FASTTEXT_DIR, f"cc.{ft_lang}.300.bin")
    try:
        if not os.path.exists(path):
            logging.info("Downloading fastText vectors for '%s' into %s ...",
                         ft_lang, FASTTEXT_DIR)
            os.makedirs(FASTTEXT_DIR, exist_ok=True)
            cwd = os.getcwd()
            try:
                os.chdir(FASTTEXT_DIR)          # fasttext.util writes to the CWD
                fasttext.util.download_model(ft_lang, if_exists='ignore')
            finally:
                os.chdir(cwd)
        logging.info("Loading fastText vectors from %s ...", path)
        _embedding_model = fasttext.load_model(path)
    except Exception as e:
        raise SemanticFilterError(
            f"could not load fastText vectors {path} ({e}). Run "
            f"`uv run python tools/download_models.py -p <your params file>` while "
            f"online, or set semantic_filter: False") from e
    _embedding_lang = lang
    return _embedding_model


def require_vectors(params):
    """Load the vectors before the run starts, if the filter is on."""
    if params.get('semantic_filter', False):
        _load_fasttext(params['language'])


def filter_dissimilar(target, candidates, threshold, lang, shown=None):
    """Candidates whose cosine similarity to `target` is below `threshold`.

    `shown` maps each candidate to the string a participant will see (its
    cased form); similarity is measured on that string.
    """
    if not candidates:
        return []
    model = _load_fasttext(lang)
    shown = shown or {}
    target_vec = model.get_word_vector(target)
    cand_matrix = np.vstack([model.get_word_vector(shown.get(c, c)) for c in candidates])

    target_norm = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    cand_norms = cand_matrix / (np.linalg.norm(cand_matrix, axis=1, keepdims=True) + 1e-9)
    similarities = cand_norms @ target_norm          # cosine similarity
    return [c for c, sim in zip(candidates, similarities) if sim < threshold]


def apply_semantic_filter(target_word, candidates, params, shown=None):
    """Apply the filter as configured (`semantic_filter`, `semantic_threshold`)."""
    if not params.get('semantic_filter', False):
        return candidates
    threshold = float(params.get('semantic_threshold', 0.35))
    return filter_dissimilar(target_word, candidates, threshold,
                             params['language'], shown=shown)
