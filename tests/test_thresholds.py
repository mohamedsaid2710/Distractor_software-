"""Surprisal-threshold boosts.

`early_position_boost` never fired in any language: the test parsed the label
ID with int(), but make_labels rewrites every ID to "1_L5" form before
choose_distractor runs, so int() raised ValueError into a bare except.
`params_de.txt` has carried `early_position_boost: 20` the whole time.
"""
import pytest

from sentence_set import Label


def _label_with(positions, words=None, surprisals=None):
    lab = Label("i1", "1_L5")
    words = words or ["haus"] * len(positions)
    surprisals = surprisals or [1.0] * len(positions)
    for w, sp, pos in zip(words, surprisals, positions):
        lab.add_sentence(w, None, sp, None, position=pos)
    return lab


def _targets(lab, params):
    """Run only the threshold arithmetic, as choose_distractor does."""
    early_boost = float(params.get('early_position_boost', 0))
    early_count = int(params.get('early_position_count', 2))
    is_early = any(p is not None and 1 <= p <= early_count for p in lab.positions)
    short_boost = float(params.get('short_word_boost', 0))
    short_max = int(params.get('short_word_max_len', 3))
    is_short = any(len(w) <= short_max for w in lab.words)
    out = []
    for surprisal in lab.surprisals:
        base = max(params["min_abs"], surprisal + params["min_delta"])
        if is_early:
            base += early_boost
        if is_short:
            base += short_boost
        out.append(base)
    return out, is_early


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
