"""German casing follows the distractor's own POS tag.

`match_casing_only` (mirror the target's casing) was removed together with
`match_noun_pos` when `pos_match` arrived.  Casing from the tag fails open on
a word with no tag (it comes out lowercase), which is why the mirroring mode
existed; with `pos_match >= 1` an untagged candidate is rejected before it can
be chosen (see tests/test_pos_match.py), so every chosen word has a tag.
"""
from maze_automate.selection import _get_german_grammatical_case as german_case


class EmptyCache:
    """A dictionary whose POS cache does not cover the candidate."""
    pos_cache = {}


class Cache:
    def __init__(self, mapping):
        self.pos_cache = mapping


def test_known_noun_is_titlecased():
    assert german_case("anwalt", Cache({"anwalt": "NOUN"}),
                       target_token="Zeuge") == "Anwalt"


def test_known_proper_noun_is_titlecased():
    assert german_case("berlin", Cache({"berlin": "PROPN"})) == "Berlin"


def test_known_verb_is_lowercased_even_for_a_capitalised_target():
    assert german_case("Sah", Cache({"sah": "VERB"}),
                       target_token="Zeuge") == "sah"


def test_punctuation_is_preserved():
    assert german_case("anwalt,", Cache({"anwalt": "NOUN"})) == "Anwalt,"


def test_sentence_initial_is_capitalised_either_way():
    assert german_case("sah", Cache({"sah": "VERB"}), is_first_word=True) == "Sah"


def test_placeholder_tokens_are_untouched():
    assert german_case("x-x-x", EmptyCache(), target_token="Zeuge") == "x-x-x"


def test_untagged_word_fails_open_to_lowercase():
    """Why pos_match >= 1 rejects untagged candidates."""
    assert german_case("anwalt", EmptyCache(), target_token="Zeuge") == "anwalt"
