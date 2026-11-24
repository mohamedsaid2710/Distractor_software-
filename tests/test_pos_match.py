"""`pos_match`: 0 off, 1 noun/non-noun, 2 class/class.

These are the pure functions every selection path goes through
(admissibility() and the first-word chooser), so they pin the rule itself.
"""
import pytest

from maze_automate.selection import pos_class, pos_mismatch, target_pos_classes


class Dict:
    def __init__(self, tags):
        self.pos_cache = tags


@pytest.mark.parametrize("tag, mode, expected", [
    ("NOUN", 1, "NOUN"), ("PROPN", 1, "NOUN"), ("VERB", 1, "NON-NOUN"),
    ("ADJ", 1, "NON-NOUN"), ("AUX", 1, "NON-NOUN"),
    ("NOUN", 2, "NOUN"), ("PROPN", 2, "NOUN"), ("AUX", 2, "VERB"),
    ("VERB", 2, "VERB"), ("ADJ", 2, "ADJ"), ("ADP", 2, "ADP"),
    ("X", 1, None), ("X", 2, None), (None, 2, None),
    ("NOUN", 0, None),
])
def test_pos_class(tag, mode, expected):
    assert pos_class(tag, mode) == expected


def test_mode_0_constrains_nothing():
    assert pos_mismatch("VERB", {"NOUN"}, 0) is None
    assert pos_mismatch(None, {"NOUN"}, 0) is None


def test_mode_1_is_both_directions():
    assert pos_mismatch("VERB", {"NOUN"}, 1) == "pos-mismatch"
    assert pos_mismatch("NOUN", {"NON-NOUN"}, 1) == "pos-mismatch"
    assert pos_mismatch("ADJ", {"NON-NOUN"}, 1) is None
    assert pos_mismatch("NOUN", {"NOUN"}, 1) is None


def test_mode_2_matches_classes():
    assert pos_mismatch("ADJ", {"VERB"}, 2) == "pos-mismatch"
    assert pos_mismatch("AUX", {"VERB"}, 2) is None
    assert pos_mismatch("PROPN", {"NOUN"}, 2) is None


def test_untagged_candidate_is_rejected_when_matching_is_on():
    assert pos_mismatch(None, {"NOUN"}, 1) == "pos-unknown"
    assert pos_mismatch("X", {"VERB"}, 2) == "pos-unknown"


def test_unknown_target_pos_enforces_nothing():
    assert pos_mismatch("VERB", set(), 1) is None


def test_target_class_from_context_tag_wins_over_lexicon():
    d = Dict({"run": "VERB"})
    assert target_pos_classes(["NOUN"], ["run"], d, 2) == {"NOUN"}


def test_target_class_falls_back_to_lexicon_when_context_tag_missing():
    d = Dict({"run": "VERB"})
    assert target_pos_classes([None], ["run."], d, 2) == {"VERB"}
    assert target_pos_classes(["X"], ["Run"], d, 1) == {"NON-NOUN"}


def test_target_classes_across_condition_rows():
    d = Dict({})
    assert target_pos_classes(["NOUN", "VERB"], ["a", "b"], d, 2) == {"NOUN", "VERB"}


def test_subordinating_and_coordinating_conjunctions_are_one_class():
    """English norms (SUBTLEX-US) have a single 'Conjunction' label."""
    assert pos_class("SCONJ", 2) == pos_class("CCONJ", 2)


def test_ambiguous_candidate_is_rejected_below_min_share():
    assert pos_mismatch("NOUN", {"NOUN"}, 1, share=0.55, min_share=0.8) == "pos-ambiguous"
    assert pos_mismatch("NOUN", {"NOUN"}, 1, share=0.95, min_share=0.8) is None


def test_unknown_share_and_zero_min_share_do_not_filter():
    assert pos_mismatch("NOUN", {"NOUN"}, 1, share=None, min_share=0.8) is None
    assert pos_mismatch("NOUN", {"NOUN"}, 1, share=0.1, min_share=0.0) is None


def test_ambiguity_filter_is_off_when_pos_match_is_off():
    assert pos_mismatch("NOUN", {"VERB"}, 0, share=0.1, min_share=0.8) is None
