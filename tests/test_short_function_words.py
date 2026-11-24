"""Short function-word targets get no frequency band, only a floor
(`short_word_freq_match`, `short_word_band_floor`)."""
import math

from maze_automate.candidates import _freq_band
from maze_automate.selection import target_is_function_word

Z = math.log(10)   # Zipf -> natural-log units


def _zipf(band):
    return tuple(round(b / Z, 3) for b in band)


def test_short_function_word_gets_floor_only():
    p = {"freq_tolerance": 1.0, "short_word_freq_match": False,
         "short_word_band_floor": 4.0, "target_is_function_word": True}
    assert _zipf(_freq_band([7.0 * Z], p, lengths=[3])) == (4.0, 9.0)


def test_short_content_word_keeps_its_band():
    p = {"freq_tolerance": 1.0, "short_word_freq_match": False,
         "target_is_function_word": False}
    assert _zipf(_freq_band([4.5 * Z], p, lengths=[3])) == (3.5, 5.5)


def test_long_function_word_keeps_its_band():
    p = {"freq_tolerance": 1.0, "short_word_freq_match": False,
         "target_is_function_word": True, "short_word_max_len": 3}
    assert _zipf(_freq_band([6.0 * Z], p, lengths=[5])) == (5.0, 7.0)


def test_switch_on_restores_the_band():
    p = {"freq_tolerance": 1.0, "short_word_freq_match": True,
         "target_is_function_word": True}
    assert _zipf(_freq_band([7.0 * Z], p, lengths=[3])) == (6.0, 8.0)


class _Lex:
    pos_cache = {"der": "DET", "tee": "NOUN"}


def test_function_word_detection_uses_context_tag_then_lexicon():
    assert target_is_function_word(["DET"], ["den"], _Lex())
    assert target_is_function_word([None], ["der"], _Lex())          # lexicon fallback
    assert not target_is_function_word(["NOUN"], ["Tee"], _Lex())
    assert not target_is_function_word([None], ["xyz"], _Lex())       # unknown
    assert not target_is_function_word(["DET", "NOUN"], ["die", "Tee"], _Lex())
