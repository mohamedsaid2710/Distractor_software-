"""`exclude_propn_candidates` works from the lexicon alone, whatever `pos_match` is."""
from maze_automate.selection import lexicon_says_propn


class _Norms:
    lexicon_from_norms = True
    pos_cache = {"table": "NOUN", "paris": "PROPN"}


class _TaggerBuilt:
    lexicon_from_norms = False
    pos_cache = {"table": "NOUN"}


def test_tagged_words():
    assert lexicon_says_propn(_Norms(), "Paris,")
    assert not lexicon_says_propn(_Norms(), "table")


def test_missing_from_norms_lexicon_counts_as_a_name():
    # SUBTLEX-US has no names, so "kevin" has no entry rather than PROPN.
    assert lexicon_says_propn(_Norms(), "kevin")


def test_missing_from_tagger_lexicon_is_not_assumed_a_name():
    assert not lexicon_says_propn(_TaggerBuilt(), "kevin")


class _English(_Norms):
    LANG = "en"
    pos_cache = {"who": "PRON", "table": "NOUN"}


def test_english_abbreviation_counts_as_a_name():
    # "who" is printed and scored as WHO, the organisation.
    assert lexicon_says_propn(_English(), "who")
    assert lexicon_says_propn(_English(), "NASA")
    assert not lexicon_says_propn(_English(), "table")
