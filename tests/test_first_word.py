"""Position 0 (`first_token_placeholder: False`) follows the same rules as
every other position: candidate_problem, then the same fallback steps."""
import random

from maze_automate.selection import Sentence, Sentence_Set
from maze_automate.selection import Repeatcounter


class FakeDict:
    LANG = "en"
    lexicon_from_norms = True

    def __init__(self, pool, nearby=()):
        self.pool, self.nearby = list(pool), list(nearby)
        self.pos_cache = {"paris": "PROPN", "bread": "NOUN", "cloud": "NOUN",
                          "sings": "VERB", "zzzzz": "X"}
        self.pos_share = {}

    def get_potential_distractors(self, *a, **k):
        return list(self.pool)

    def get_best_frequency_pool(self, length, zipf, n=200):
        return list(self.nearby)

    def get_emergency_pool(self, length, is_noun=False):
        return []


def _first(pool, nearby=(), **extra):
    params = {"language": "en", "first_token_placeholder": False, "pos_match": 1,
              "exclude_propn_candidates": True, "len_tolerance": 0,
              "enforce_length_match": True, "max_repeat": 1}
    params.update(extra)
    ss = Sentence_Set("1")
    s = Sentence(["Table", "stood"], [0, 1], "1", "t", "Table stood")
    s.pos_tags = ["NOUN", "VERB"]
    ss.sentences = [s]
    later = type("L", (), {"distractor": "sang", "choose_distractor": lambda *a, **k: "sang"})()
    ss.labels = {1: later}
    random.seed(0)
    ss.do_distractors(None, FakeDict(pool, nearby), lambda w, p: (5, 5, 0, 20),
                      params, Repeatcounter(1))
    return s.distractors[0]


def test_first_word_obeys_pos_propn_and_junk_rules():
    # paris: proper noun; sings: wrong class; zzzzz: junk; table: the target.
    assert _first(["paris", "sings", "zzzzz", "table", "bread"]) == "Bread"


def test_first_word_falls_back_to_nearest_frequency():
    assert _first(["paris"], nearby=["cloud"]) == "Cloud"


def test_first_word_placeholder_when_nothing_is_admissible():
    assert _first(["paris", "sings"]) == "x-x-x"
