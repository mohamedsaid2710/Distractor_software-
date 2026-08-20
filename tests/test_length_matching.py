"""Length matching for targets at the edges of the lexicon's range.

The threshold functions clamped the length they searched with, while
candidate_length_ok compared against the true length -- so with
`len_tolerance: 0` a 1- or 2-letter English target rejected every candidate its
own pool could offer and fell through the entire retrieval cascade.  17% of
non-initial tokens in English_sample.txt are 1-2 characters.
"""
import pytest

from wordfreq_distractor import (
    effective_target_length,
    get_thresholds_en,
    get_thresholds_de,
    get_thresholds_ar,
)

EN = {"min_word_len": 2}


@pytest.mark.parametrize("word,expected", [
    ("a", 2), ("at", 2), ("cat", 3), ("mountain", 8),
])
def test_effective_length_respects_min_word_len(word, expected):
    assert effective_target_length(len(word), EN) == expected


def test_effective_length_caps_very_long_targets():
    assert effective_target_length(40, EN) == 15
    assert effective_target_length(40, EN, max_len=None) == 40


def test_missing_param_keeps_the_old_floor():
    assert effective_target_length(1, {}) == 3
    assert effective_target_length(1, None) == 3


@pytest.mark.parametrize("word", ["a", "at", "in", "is", "on", "it", "he"])
def test_threshold_length_matches_the_clamped_length(word):
    """The pool the thresholds ask for must be one the length test accepts."""
    min_len, max_len, _, _ = get_thresholds_en([word], EN)
    eff = effective_target_length(len(word), EN)
    assert min_len == max_len == eff


@pytest.mark.parametrize("fn", [get_thresholds_en, get_thresholds_de, get_thresholds_ar])
def test_all_languages_use_the_same_floor(fn):
    min_len, _, _, _ = fn(["ab"], {"min_word_len": 4})
    assert min_len == 4


def test_two_letter_english_pool_is_non_empty():
    """`min_word_len: 3` left words_by_len[2] empty, so nothing could match."""
    from set_params import set_params
    from wordfreq_distractor import wordfreq_English_zipf_dict
    params = set_params("params_en.txt")
    params["semantic_filter"] = False
    d = wordfreq_English_zipf_dict(params)
    eff = effective_target_length(2, params)
    assert len(d.words_by_len.get(eff, [])) > 50


# --- the mirror-image case: long German compounds -------------------------

LONG_COMPOUND = "Dämmerungsbeleuchtung"   # 21 characters


def test_german_does_not_cap_long_compounds():
    """German thresholds are uncapped, so the length test must be too.

    Capping only one side rejects every candidate the pool can offer -- the
    same failure as short English targets, mirrored.
    """
    de = {"min_word_len": 2}
    min_len, max_len, _, _ = get_thresholds_de([LONG_COMPOUND], de)
    assert min_len == max_len == len(LONG_COMPOUND)
    assert effective_target_length(len(LONG_COMPOUND), de, max_len=None) == len(LONG_COMPOUND)


@pytest.mark.parametrize("fn,cap", [
    (get_thresholds_en, 15),
    (get_thresholds_ar, 15),
    (get_thresholds_de, None),
])
def test_threshold_and_length_test_agree_at_both_extremes(fn, cap):
    """Whatever the clamp, both sides of the comparison must apply it."""
    params = {"min_word_len": 2}
    for word in ["a", "at", "mountain", LONG_COMPOUND]:
        min_len, max_len, _, _ = fn([word], params)
        assert min_len == max_len
        assert effective_target_length(len(word), params, max_len=cap) == min_len
