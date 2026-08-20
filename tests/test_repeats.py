"""max_repeat enforcement.

The exemption for short words was a hardcoded `len(word) > 4` inside
Repeatcounter, and a second, disagreeing threshold (`<= 3`) in the fallback
path.  So `max_repeat: 1` was silently unenforced for short distractors -- one
word appeared five times in a 10-item English run.
"""
import pytest

from limit_repeats import Repeatcounter


def test_long_word_is_banned_at_the_limit():
    r = Repeatcounter(1)
    r.increment("kitchen")
    assert "kitchen" in r.banned


def test_short_word_is_exempt_by_default():
    """Documents the default, which is the old hardcoded behaviour."""
    r = Repeatcounter(1)
    for _ in range(5):
        r.increment("owe")
    assert "owe" not in r.banned
    assert r.distractors["owe"] == 5


def test_exemption_length_is_configurable():
    r = Repeatcounter(1, exempt_max_len=0)
    r.increment("owe")
    assert "owe" in r.banned


@pytest.mark.parametrize("length,exempt,banned", [
    (4, 4, False),   # at the boundary: exempt
    (5, 4, True),    # just over: enforced
    (2, 2, False),
    (3, 2, True),
])
def test_boundary_is_inclusive(length, exempt, banned):
    r = Repeatcounter(1, exempt_max_len=exempt)
    word = "a" * length
    r.increment(word)
    assert (word in r.banned) is banned


def test_max_repeat_zero_disables_the_limit():
    r = Repeatcounter(0, exempt_max_len=0)
    for _ in range(10):
        r.increment("kitchen")
    assert r.banned == []


def test_counting_is_case_insensitive():
    r = Repeatcounter(2, exempt_max_len=0)
    r.increment("Kitchen")
    r.increment("kitchen")
    assert r.distractors["kitchen"] == 2
    assert "kitchen" in r.banned
