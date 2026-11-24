"""choose_distractor's fallback: one sequence for every language.

below-threshold -> nearest-frequency -> repeats-allowed -> none-admissible,
each step taking the highest-surprisal admissible word.
"""
import pytest

from maze_automate.selection import Label

SURPRISAL = {"bread": 30.0, "cloud": 22.0, "stone": 12.0, "chair": 9.0,
             "plant": 18.0, "ghost": 26.0, "شجرة": 12.0, "حجارة": 18.0}


class FakeModel:
    model_batch_size = 8

    def get_surprisal_batch_from_hidden(self, hidden, words, batch_size=8):
        return [SURPRISAL.get(w.lower(), 0.0) for w in words]

    def get_surprisal_from_hidden(self, hidden, word):
        return SURPRISAL.get(word.lower(), 0.0)


class FakeDict:
    LANG = "en"
    lexicon_from_norms = True

    def __init__(self, pool, nearby=(), emergency=()):
        self.pool, self.nearby, self.emergency = list(pool), list(nearby), list(emergency)
        self.pos_cache = {w: "NOUN" for w in SURPRISAL}
        self.pos_share = {}
        self.last_freq_widen_steps = 0
        self.last_band = None

    def get_potential_distractors(self, *a, **k):
        return list(self.pool)

    def get_best_frequency_pool(self, length, zipf, n=200):
        return list(self.nearby)

    def get_emergency_pool(self, length, is_noun=False):
        return list(self.emergency)


def _threshold(words, params):
    return 5, 5, 0.0, 20.0


def _choose(pool, nearby=(), emergency=(), banned=(), **extra):
    params = {"language": "en", "min_abs": 20, "min_delta": 5, "pos_match": 0,
              "num_to_test": 200, "len_tolerance": 0, "enforce_length_match": True,
              "exclude_propn_candidates": False}
    params.update(extra)
    lab = Label("1", "1_L1")
    lab.add_sentence("table", None, 4.0, [], position=3)    # threshold 20
    word = lab.choose_distractor(FakeModel(), FakeDict(pool, nearby, emergency),
                                 _threshold, params, list(banned))
    return word, lab.report


def test_highest_passing_candidate_wins():
    word, rep = _choose(["cloud", "bread", "stone"])
    assert word == "bread" and rep["relaxation"] == "" and not rep["used_fallback"]


def test_below_threshold_takes_the_best_not_a_random_one():
    word, rep = _choose(["stone", "plant", "chair"])
    assert word == "plant" and rep["relaxation"] == "below-threshold"


def test_empty_pool_uses_nearest_frequency_words():
    word, rep = _choose([], nearby=["chair", "ghost"])
    assert word == "ghost" and rep["relaxation"] == "nearest-frequency"


def test_repeats_only_as_a_later_step():
    word, rep = _choose(["bread"], banned=["bread"], allow_banned_fallback=True)
    assert word == "bread" and rep["relaxation"] == "repeats-allowed"


def test_nothing_admissible_gives_the_placeholder():
    word, rep = _choose(["bread"], banned=["bread"])
    assert word == "x-x-x" and rep["relaxation"] == "none-admissible"
    assert rep["is_placeholder"]


@pytest.mark.parametrize("lang,pool,best", [
    ("en", ["stone", "plant"], "plant"),
    ("de", ["stone", "plant"], "plant"),
    ("ar", ["شجرة", "حجارة"], "حجارة"),
])
def test_same_sequence_in_every_language(lang, pool, best):
    # The fallback logic is language-independent; only the words differ.
    word, rep = _choose(pool, language=lang)
    assert rep["relaxation"] == "below-threshold"
    assert word.lower() == best


def test_each_language_keeps_its_own_letters():
    # A Latin word is never an Arabic distractor, nor an Arabic one English.
    assert _choose(["plant"], language="ar")[1]["relaxation"] == "none-admissible"
    assert _choose(["حجارة"], language="en")[1]["relaxation"] == "none-admissible"
