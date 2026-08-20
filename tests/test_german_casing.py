"""German casing.

`match_casing_only` was documented, enabled in params_de.txt, and inert: the
function computed the target's capitalisation and then never read it, always
taking the pos_cache route instead.  On a cache miss that route defaults to
"not a noun" -> lowercase, which is where lowercased German nouns in the output
come from.  34.7% of the German lexicon has no cache entry.
"""
import pytest

from sentence_set import _get_german_grammatical_case as german_case


class EmptyCache:
    """A dictionary whose POS cache does not cover the candidate."""
    pos_cache = {}


class Cache:
    def __init__(self, mapping):
        self.pos_cache = mapping


# --- casing-only mode ----------------------------------------------------

def test_capitalised_target_yields_capitalised_distractor_on_a_cache_miss():
    """The regression: this returned 'anwalt' -- a lowercased German noun."""
    out = german_case("anwalt", EmptyCache(), target_token="Zeuge",
                      match_casing_only=True)
    assert out == "Anwalt"


def test_lowercase_target_yields_lowercase_distractor():
    out = german_case("Anwalt", EmptyCache(), target_token="sah",
                      match_casing_only=True)
    assert out == "anwalt"


def test_casing_only_does_not_consult_the_cache():
    """Mirroring must be immune to a wrong or missing cache entry."""
    wrong = Cache({"anwalt": "VERB"})
    assert german_case("anwalt", wrong, target_token="Zeuge",
                       match_casing_only=True) == "Anwalt"


def test_punctuation_is_preserved():
    assert german_case("anwalt,", EmptyCache(), target_token="Zeuge",
                       match_casing_only=True) == "Anwalt,"


def test_sentence_initial_is_capitalised_either_way():
    assert german_case("anwalt", EmptyCache(), target_token="sah",
                       is_first_word=True, match_casing_only=True) == "Anwalt"


def test_placeholder_tokens_are_untouched():
    assert german_case("x-x-x", EmptyCache(), target_token="Zeuge",
                       match_casing_only=True) == "x-x-x"


# --- POS mode (unchanged behaviour) --------------------------------------

def test_pos_mode_titlecases_a_known_noun():
    assert german_case("anwalt", Cache({"anwalt": "NOUN"}),
                       target_token="Zeuge") == "Anwalt"


def test_pos_mode_lowercases_a_known_verb():
    assert german_case("Sah", Cache({"sah": "VERB"}),
                       target_token="Zeuge") == "sah"


def test_pos_mode_still_fails_open_on_a_cache_miss():
    """Documenting why match_casing_only exists at all."""
    assert german_case("anwalt", EmptyCache(), target_token="Zeuge") == "anwalt"
