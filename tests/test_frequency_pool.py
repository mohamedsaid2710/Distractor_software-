"""`get_best_frequency_pool` takes Zipf, and must return a band around it.

Before the fix its bisection key was a raw Zipf value searched against an
array of natural-log frequencies, so the key was always off the end of the
array and the function returned the same tail -- the rarest words of that
length -- for every target.
"""
import math

import pytest

from wordfreq_distractor import wordfreq_dict


class _W:
    def __init__(self, text, zipf):
        self.text = text
        self.freq = zipf * math.log(10)
        self.len = len(text)


@pytest.fixture
def toy_dict():
    """80 eight-letter words with Zipf descending from 6.0 to 2.05."""
    d = wordfreq_dict({})
    d.words = [_W("w%07d" % i, 6.0 - i * 0.05) for i in range(80)]
    d._build_length_index()
    return d


def _zipfs(d, pool):
    by_text = {w.text: w.freq / math.log(10) for w in d.words}
    return [by_text[t] for t in pool]


def test_pool_is_centred_on_the_requested_zipf(toy_dict):
    for target in (5.5, 4.0, 2.5):
        pool = toy_dict.get_best_frequency_pool(8, target, n=10)
        zs = _zipfs(toy_dict, pool)
        assert len(pool) == 10
        assert min(zs) <= target <= max(zs)


def test_different_targets_give_different_pools(toy_dict):
    """The exact symptom: one identical pool for every target."""
    high = toy_dict.get_best_frequency_pool(8, 5.5, n=10)
    low = toy_dict.get_best_frequency_pool(8, 2.5, n=10)
    assert high != low
    assert not set(high) & set(low)


def test_pool_is_not_the_rarest_tail(toy_dict):
    """A high-frequency target must not be served the rarest words."""
    pool = toy_dict.get_best_frequency_pool(8, 5.5, n=10)
    rarest = [w.text for w in toy_dict.words_by_len[8][-10:]]
    assert set(pool) & set(rarest) == set()


def test_window_clamps_at_both_ends(toy_dict):
    """Targets outside the lexicon's range still return n words."""
    assert len(toy_dict.get_best_frequency_pool(8, 99.0, n=10)) == 10
    assert len(toy_dict.get_best_frequency_pool(8, 0.0, n=10)) == 10


def test_small_bucket_returns_everything(toy_dict):
    assert len(toy_dict.get_best_frequency_pool(8, 4.0, n=500)) == 80


def test_missing_length_returns_empty(toy_dict):
    assert toy_dict.get_best_frequency_pool(3, 4.0) == []
