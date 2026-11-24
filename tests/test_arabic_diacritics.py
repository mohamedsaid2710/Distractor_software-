"""A vowelled Arabic target is measured and compared as its bare letters."""
from maze_automate.utils import bare_form
from maze_automate.candidates import get_thresholds_ar
from maze_automate.selection import Label, lexicon_tag

VOWELLED = "الْكِتَابُ،"         # with tashkeel and a comma
BARE = "الكتاب"


def test_bare_form_drops_punctuation_and_marks_only():
    assert bare_form(VOWELLED) == BARE
    assert bare_form("Hund,") == "Hund" and bare_form("Grüße") == "Grüße"
    assert bare_form("كـتـاب") == "كتاب"          # tatweel


def test_length_is_counted_in_letters():
    assert get_thresholds_ar([VOWELLED], {"min_word_len": 2})[:2] == (6, 6)


def test_lexicon_finds_a_vowelled_target():
    d = type("D", (), {"pos_cache": {BARE: "NOUN"}})()
    assert lexicon_tag(d, VOWELLED) == "NOUN"


class _Model:
    model_batch_size = 8

    def get_surprisal_batch_from_hidden(self, hidden, words, batch_size=8):
        return [30.0 for _ in words]

    def get_surprisal_from_hidden(self, hidden, word):
        return 30.0


class _Dict:
    LANG = "ar"
    lexicon_from_norms = True
    pos_cache = {BARE: "NOUN", "المدرسة": "NOUN", "الطاولة": "NOUN"}
    pos_share = {}
    last_freq_widen_steps = 0
    last_band = None

    def get_potential_distractors(self, *a, **k):
        return [BARE, "الطاولة"]           # the target itself, and a 7-letter word

    def get_best_frequency_pool(self, *a, **k):
        return []

    def get_emergency_pool(self, *a, **k):
        return []


def test_vowelled_target_is_never_its_own_distractor():
    lab = Label("1", "1_L6")
    lab.add_sentence(VOWELLED, None, 4.0, [], position=2)
    params = {"language": "ar", "min_abs": 10, "min_delta": 5, "pos_match": 0,
              "len_tolerance": 0, "enforce_length_match": True,
              "exclude_propn_candidates": False}
    word = lab.choose_distractor(_Model(), _Dict(), lambda w, p: (6, 6, 0, 20),
                                 params, [])
    # الكتاب is the target; الطاولة has 7 letters -- nothing is admissible.
    assert word == "x-x-x"
