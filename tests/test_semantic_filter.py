"""semantic_filter: cosine similarity against the user's threshold, no silent
loosening, no silent switch-off, words compared as shown."""
import numpy as np
import pytest

from maze_automate import semantic_filter as sf

VEC = {"Hund": [1, 0], "Katze": [0.9, 0.1], "katze": [0, 1],
       "Tisch": [0, 1], "Wolke": [0.1, 1]}


class FakeFT:
    def get_word_vector(self, w):
        return np.array(VEC.get(w, [0.5, 0.5]), dtype=float)


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(sf, "_load_fasttext", lambda lang: FakeFT())


def test_similar_candidates_are_removed(fake):
    out = sf.filter_dissimilar("Hund", ["Katze", "Tisch", "Wolke"], 0.35, "de")
    assert out == ["Tisch", "Wolke"]


def test_threshold_is_never_loosened(fake):
    # Everything is similar: the result is empty, not a looser threshold.
    assert sf.filter_dissimilar("Hund", ["Katze"], 0.35, "de") == []


def test_compared_as_shown(fake):
    # "katze" (lowercase key) is shown as "Katze", which is similar to "Hund".
    assert sf.filter_dissimilar("Hund", ["katze"], 0.35, "de", shown={"katze": "Katze"}) == []
    assert sf.filter_dissimilar("Hund", ["katze"], 0.35, "de") == ["katze"]


def test_missing_vectors_stop_the_run(monkeypatch):
    def boom(lang):
        raise sf.SemanticFilterError("no vectors")
    monkeypatch.setattr(sf, "_load_fasttext", boom)
    with pytest.raises(sf.SemanticFilterError):
        sf.require_vectors({"semantic_filter": True, "language": "de"})
    sf.require_vectors({"semantic_filter": False, "language": "de"})   # off: fine
