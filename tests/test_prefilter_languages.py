"""The noun / non-noun pre-filter applies the same rule in every language."""
import pytest

from maze_automate.candidates import wordfreq_dict
from maze_automate.candidates import distractor

WORDS = {
    "en": {"table": "NOUN", "shirt": "NOUN", "sings": "VERB", "quiet": "ADJ", "paris": "PROPN"},
    "de": {"tisch": "NOUN", "hemde": "NOUN", "singt": "VERB", "leise": "ADJ", "paris": "PROPN"},
    "ar": {"كتاب": "NOUN", "بيوت": "NOUN", "يكتب": "VERB", "جميل": "ADJ", "مصر": "PROPN"},
}


def _dict(lang):
    d = wordfreq_dict({})
    d.lang = lang
    d.pos_cache = dict(WORDS[lang])
    d.words = [distractor(w, 10.0) for w in WORDS[lang]]
    d._build_length_index()
    return d


@pytest.mark.parametrize("lang", ["en", "de", "ar"])
def test_noun_target_gets_only_nouns(lang):
    d = _dict(lang)
    params = {"min_pool_size": 1, "num_to_test": 100, "exclude_propn_candidates": True}
    pool = d.get_potential_distractors(2, 6, None, None, params, pos_filter="NOUN")
    assert pool and all(d.pos_cache[w] == "NOUN" for w in pool)


@pytest.mark.parametrize("lang", ["en", "de", "ar"])
def test_non_noun_target_gets_only_non_nouns(lang):
    d = _dict(lang)
    params = {"min_pool_size": 1, "num_to_test": 100}
    pool = d.get_potential_distractors(2, 6, None, None, params, pos_filter="!NOUN")
    assert pool and all(d.pos_cache[w] in ("VERB", "ADJ") for w in pool)
