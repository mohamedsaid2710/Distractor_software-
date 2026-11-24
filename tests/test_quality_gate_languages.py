"""The candidate quality gate applies each language's own letters, and no
language's rule reaches another (_SCRIPT_LETTERS)."""
from maze_automate.candidates import _is_valid_cache_word as ok


def test_arabic_words_pass():
    # The German vowel rule used to reject every Arabic word.
    for w in ("يسجل", "كتب", "الطبيب", "غادر"):
        assert ok(w, len(w), lang="ar"), w


def test_u_only_words_pass():
    for w, lang in (("bus", "en"), ("run", "en"), ("just", "en"),
                    ("und", "de"), ("hund", "de"), ("kunst", "de")):
        assert ok(w, len(w), lang=lang), w


def test_latin_junk_still_rejected():
    assert not ok("bcdfg", 5, lang="en")      # no vowel
    assert not ok("xxxxx", 5, lang="de")      # one letter repeated


def test_german_letters_do_not_leak_into_english():
    # "ä" is a German vowel, not an English one.
    assert not ok("bzzä", 4, lang="en")
