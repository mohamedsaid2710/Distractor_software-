"""Frequency matching: the band must be centred on the target, and the
neighbourhood search must return a neighbourhood rather than the tail.

Both of these failed before the fix, for every English and Arabic run.
"""
import math

import pytest
import wordfreq

from wordfreq_distractor import (
    _freq_band,
    get_thresholds_en,
    get_thresholds_de,
    get_thresholds_ar,
    get_frequency_en,
)

LN10 = math.log(10)

# Content words whose Zipf is <= 4.78, i.e. inside the natural-log range
# [3, 11] where the old legacy branch collapsed to zero width.
MID_FREQUENCY_TARGETS = ["mountain", "rocky", "sentence", "trail", "quiet"]


@pytest.mark.parametrize("word", MID_FREQUENCY_TARGETS)
def test_default_band_is_not_degenerate(word):
    """Regression: min_freq == max_freq starved the tight-band search."""
    _, _, min_freq, max_freq = get_thresholds_en([word], {})
    assert max_freq - min_freq > 0.0


@pytest.mark.parametrize("word", MID_FREQUENCY_TARGETS)
def test_default_band_is_centred_on_the_target(word):
    _, _, min_freq, max_freq = get_thresholds_en([word], {})
    target = get_frequency_en(word)
    midpoint = (min_freq + max_freq) / 2
    assert midpoint == pytest.approx(target, abs=1e-9)
    assert min_freq < target < max_freq


def test_freq_tolerance_sets_the_half_width():
    _, _, min_freq, max_freq = get_thresholds_en(["mountain"], {"freq_tolerance": 0.5})
    assert (max_freq - min_freq) == pytest.approx(2 * 0.5 * LN10)


def test_explicit_null_tolerance_restores_a_wide_band():
    """`freq_tolerance: null` opts out, but must still be a real interval."""
    _, _, min_freq, max_freq = get_thresholds_en(["mountain"], {"freq_tolerance": None})
    assert min_freq <= 3.0
    assert max_freq >= 11.0
    target = get_frequency_en("mountain")
    assert min_freq <= target <= max_freq


def test_wide_band_widens_to_include_an_out_of_range_target():
    # "the" is far above the wide band's upper bound.
    _, _, min_freq, max_freq = get_thresholds_en(["the"], {"freq_tolerance": None})
    assert max_freq >= get_frequency_en("the")


@pytest.mark.parametrize("fn", [get_thresholds_en, get_thresholds_de, get_thresholds_ar])
def test_all_three_languages_share_the_band_logic(fn):
    _, _, min_freq, max_freq = fn(["haus"], {"freq_tolerance": 1.0})
    assert (max_freq - min_freq) == pytest.approx(2 * LN10)


def test_freq_band_helper_directly():
    freqs = [10.0]
    assert _freq_band(freqs, {"freq_tolerance": 1.0}) == pytest.approx(
        (10.0 - LN10, 10.0 + LN10)
    )
    assert _freq_band(freqs, None)[1] > _freq_band(freqs, None)[0]
