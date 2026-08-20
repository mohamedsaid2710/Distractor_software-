"""Arabic: Farasa tag parsing and diacritic handling.

Every Arabic word came back 'X' from the tagger: farasapy wraps its output in
S/S ... E/E sentinels and separates tokens with spaces, while the parser split
on '+' alone -- so it read the trailing 'E' as the tag.  'X' enters no
candidate pool, so Arabic ran with no POS information and
`exclude_propn_candidates` did nothing at all.
"""
import pytest

from utils import strip_punct, copy_punct
from wordfreq_distractor import _farasa_tag_to_upos, strip_arabic_diacritics


# --- Farasa parsing ------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("S/S كتاب/NOUN-MS E/E", "NOUN"),          # feature suffix stripped
    ("S/S في/PREP E/E", "ADP"),
    ("S/S يكتب/V E/E", "VERB"),
    ("S/S مدرس +ة/NOUN+NSUFF-FS E/E", "NOUN"),  # NSUFF is a clitic
    ("S-و/CONJ+ال/DET+قمر/NOUN", "NOUN"),        # the older sentinel-less format
])
def test_lexical_head_is_found(raw, expected):
    assert _farasa_tag_to_upos(raw) == expected


def test_possessive_suffix_does_not_make_a_noun_a_pronoun():
    """Arabic writes possessives as suffixes; the head is still the noun.

    The shipped lexicon holds 7,166 PRON entries because of this.
    """
    assert _farasa_tag_to_upos("S/S كتاب +ه/NOUN+PRON E/E") == "NOUN"


def test_standalone_pronoun_is_still_a_pronoun():
    """Skipping clitics must not erase a word that IS a function word."""
    assert _farasa_tag_to_upos("S/S هو/PRON E/E") == "PRON"


def test_unparseable_input_is_X_not_a_crash():
    assert _farasa_tag_to_upos("") == "X"
    assert _farasa_tag_to_upos("S/S E/E") == "X"
    assert _farasa_tag_to_upos("nonsense") == "X"


def test_unknown_tag_becomes_X():
    assert _farasa_tag_to_upos("S/S x/WEIRDTAG E/E") == "X"


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
