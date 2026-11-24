"""Arabic: diacritic and punctuation handling, and the clitic filter."""
import json


from maze_automate.utils import strip_punct, copy_punct, lexicon_path
from maze_automate.candidates import strip_arabic_diacritics, _load_clitic_words


# --- diacritics (C4) -----------------------------------------------------

VOCALIZED = "الْكِتَابُ"        # 10 code points, 6 orthographic letters
UNVOCALIZED = "الكتاب"


def test_diacritics_are_not_punctuation():
    """Harakat are Unicode Mn, so isalnum() is False for them.

    strip_punct used to eat the trailing damma, making the target look one
    character shorter for length matching.
    """
    assert strip_punct(VOCALIZED) == VOCALIZED


def test_trailing_diacritic_is_not_grafted_onto_the_distractor():
    assert copy_punct(VOCALIZED, "XYZ") == "XYZ"


def test_real_punctuation_around_a_vocalized_word_still_moves():
    assert copy_punct("«" + VOCALIZED + "»،", "XYZ") == "«XYZ»،"


def test_diacritic_stripping_is_still_available_for_lookups():
    assert strip_arabic_diacritics(VOCALIZED) == UNVOCALIZED


# --- clitic filter -------------------------------------------------------

def test_clitic_words_are_read_from_the_lexicon(tmp_path):
    p = tmp_path / "lex.json"
    p.write_text(json.dumps({"tags": {}, "clitics": ["والكتاب", "بالبيت"]}),
                 encoding="utf-8")
    assert _load_clitic_words(str(p)) == {"والكتاب", "بالبيت"}


def test_shipped_arabic_lexicon_marks_clitic_forms():
    clitics = _load_clitic_words(lexicon_path("ar"))
    assert {"والكتاب", "بالبيت", "للطالب", "وقال"} <= clitics
    for stem in ("وزير", "لكن", "كتاب"):       # start with و/ل/ك but are stems
        assert stem not in clitics


# --- common web misspellings (lexical_norms.arabic_misspellings) ---------
def test_arabic_misspellings_found_by_frequency():
    from maze_automate.lexicons.norms import arabic_misspellings
    freq = {"فتاه": 1e-6, "فتاة": 5e-5, "هذة": 1e-6, "هذه": 1e-3,
            "بى": 1e-6, "بي": 1e-4, "راية": 2e-5, "رؤية": 2e-4,
            "كتاب": 1e-4}.get
    f = lambda w: freq(w) or 0.0
    out = arabic_misspellings(["فتاه", "هذة", "بى", "راية", "كتاب"], freq=f)
    assert out == {"فتاه": "فتاة", "هذة": "هذه", "بى": "بي"}
    # راية (flag) is a real word: ا is never swapped for ؤ.


def test_madda_words_are_not_misspellings():
    from maze_automate.lexicons.norms import arabic_misspellings
    freq = {"آثر": 1e-5, "أثر": 1e-3}.get
    assert arabic_misspellings(["آثر"], freq=lambda w: freq(w) or 0.0) == {}
