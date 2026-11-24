"""Surprisal-threshold boosts.

`early_position_boost` never fired in any language: the test parsed the label
ID with int(), but make_labels rewrites every ID to "1_L5" form before
choose_distractor runs, so int() raised ValueError into a bare except.
`config/params_de.txt` has carried `early_position_boost: 20` the whole time.
"""
import pytest

from maze_automate.selection import Label, surprisal_thresholds


def _label_with(positions, words=None, surprisals=None):
    lab = Label("i1", "1_L5")
    words = words or ["haus"] * len(positions)
    surprisals = surprisals or [1.0] * len(positions)
    for w, sp, pos in zip(words, surprisals, positions):
        lab.add_sentence(w, None, sp, None, position=pos)
    return lab


def _targets(lab, params):
    """The real threshold function, as choose_distractor calls it."""
    targets, is_early, _ = surprisal_thresholds(lab.words, lab.surprisals,
                                                lab.positions, params)
    return targets, is_early


PARAMS = {"min_abs": 15, "min_delta": 8,
          "early_position_boost": 20, "early_position_count": 5}


def test_label_records_its_positions():
    assert _label_with([1, 1]).positions == [1, 1]


def test_early_position_fires_on_a_renamed_label():
    """The regression: lab is "1_L5", int() cannot parse it."""
    _, is_early = _targets(_label_with([1]), PARAMS)
    assert is_early is True


def test_boost_is_actually_added():
    targets, _ = _targets(_label_with([1]), PARAMS)
    baseline, _ = _targets(_label_with([9]), PARAMS)
    assert targets[0] - baseline[0] == pytest.approx(20)


def test_late_position_is_not_boosted():
    _, is_early = _targets(_label_with([6]), PARAMS)
    assert is_early is False


def test_boundary_is_inclusive():
    assert _targets(_label_with([5]), PARAMS)[1] is True
    assert _targets(_label_with([6]), PARAMS)[1] is False


def test_position_zero_is_the_placeholder_not_an_early_position():
    assert _targets(_label_with([0]), PARAMS)[1] is False


def test_any_early_occurrence_across_condition_rows_counts():
    """One distractor serves every row, so the strictest row wins."""
    assert _targets(_label_with([9, 2]), PARAMS)[1] is True


def test_zero_boost_is_a_no_op():
    p = dict(PARAMS, early_position_boost=0)
    assert _targets(_label_with([1]), p)[0] == _targets(_label_with([9]), p)[0]


def test_short_target_uses_min_abs_short_and_its_boost():
    p = {"min_abs": 25, "min_delta": 8, "min_abs_short": 15, "short_word_boost": 5}
    targets, _, is_short = surprisal_thresholds(["the"], [2.0], [4], p)
    assert is_short and targets == [20.0]          # max(15, 10) + 5


def test_vowelled_arabic_short_word_counts_its_letters():
    # "مِنْ" is 2 letters (4 code points with its marks).
    _, _, is_short = surprisal_thresholds(["مِنْ"], [2.0], [4],
                                          {"min_abs": 20, "min_delta": 5})
    assert is_short
