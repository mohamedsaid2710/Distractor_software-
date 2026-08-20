"""Input parsing: delimiter choice and tokenization.

Both defects here are silent -- they produce a plausible-looking run whose
items are wrong -- so they are the most dangerous ones for a real experiment.
"""
import re
import textwrap

import pytest

from input import detect_delimiter, tokenize, read_input


# --- C1: delimiter -------------------------------------------------------

GERMAN_RELATIVE_CLAUSE = "rel;i1;Der Anwalt, den der Zeuge sah, war alt."


def test_semicolon_wins_over_commas_inside_the_sentence():
    """The exact reproduction: csv.Sniffer used to answer ',' here."""
    assert detect_delimiter(GERMAN_RELATIVE_CLAUSE) == ';'


def test_english_appositive_keeps_semicolon():
    assert detect_delimiter("filler;i2;The dog, a beagle, barked.") == ';'


def test_genuine_comma_file_still_sniffs_as_comma():
    sample = "rel,i1,The lawyer saw the witness.\nrel,i2,The doctor saw the nurse.\n"
    assert detect_delimiter(sample) == ','


def test_comma_file_whose_sentence_contains_a_semicolon():
    sample = "rel,i1,The lawyer arrived; the witness did not.\n"
    assert detect_delimiter(sample) == ','


def test_latin_square_set_stays_one_item(tmp_path):
    """Two condition rows of one item must land in ONE Sentence_Set.

    Before the fix each row was split on ',' into three fields, so both rows
    parsed with a different id and the shared-distractor invariant was gone.
    """
    f = tmp_path / "stim.txt"
    f.write_text(textwrap.dedent("""\
        rel_sub;1;Der Anwalt, den der Zeuge sah, war alt.
        rel_obj;1;Der Anwalt, der den Zeugen sah, war alt.
        """), encoding='utf-8')
    sents = read_input(str(f))
    assert list(sents.keys()) == ['1']
    assert len(sents['1'].sentences) == 2


# --- C2: tokenization ----------------------------------------------------

# The regex `tokenize` replaces.  Kept here so the tests document exactly what
# the old behaviour was, rather than asserting a tautology against split().
OLD_TOKENIZER_RE = re.compile(
    r"[^\s\w0-9]*[A-Za-zÄÖÜäöüßÀ-ÖØ-öø-ÿ\u0600-\u06FF0-9]+"
    r"(?:[-'][A-Za-zÄÖÜäöüßÀ-ÖØ-öø-ÿ\u0600-\u06FF0-9]+)*"
    r"[^\s\w0-9]*[.,!?;]*", re.UNICODE)


@pytest.mark.parametrize("sentence,expected", [
    ("The dog doesn't bark.", ["The", "dog", "doesn't", "bark."]),
    ("The dog doesn\u2019t bark.", ["The", "dog", "doesn\u2019t", "bark."]),
    ("It cost 1,000 dollars.", ["It", "cost", "1,000", "dollars."]),
    ("The U.S. team won.", ["The", "U.S.", "team", "won."]),
    ("A well-known fact.", ["A", "well-known", "fact."]),
    ("Er sagte: \u201eNein\u201c heute.", ["Er", "sagte:", "\u201eNein\u201c", "heute."]),
    ("  spaced   out  ", ["spaced", "out"]),
])
def test_tokenizer_produces_the_expected_tokens(sentence, expected):
    assert tokenize(sentence) == expected


@pytest.mark.parametrize("sentence,old_count,new_count", [
    ("The dog doesn\u2019t bark.", 5, 4),      # curly apostrophe split the word
    ("It cost 1,000 dollars.", 5, 4),       # digit grouping split the number
    ("The U.S. team won.", 5, 4),           # internal periods split the acronym
])
def test_old_regex_disagreed_with_whitespace_on_ordinary_input(sentence, old_count, new_count):
    """What C2 was: the two counts feeding one output row disagreed.

    A user-supplied labels column then raised "labels are wrong length"; with
    generated labels the run continued and misaligned word k with distractor k.
    """
    assert len(OLD_TOKENIZER_RE.findall(sentence)) == old_count
    assert len(tokenize(sentence)) == new_count == len(sentence.split())


def test_supplied_labels_of_the_right_length_are_accepted(tmp_path):
    f = tmp_path / "stim.txt"
    f.write_text("rel;1;The dog doesn’t bark.;0 1 2 3\n", encoding='utf-8')
    sents = read_input(str(f))
    sentence = sents['1'].sentences[0]
    assert len(sentence.words) == 4
    assert len(sentence.labels) == 4


def test_distractor_slot_count_matches_the_sentence(tmp_path):
    """One distractor per whitespace token, or PCIbex misaligns the item."""
    f = tmp_path / "stim.txt"
    f.write_text("rel;1;It cost 1,000 dollars.\n", encoding='utf-8')
    sentence = read_input(str(f))['1'].sentences[0]
    assert sentence.words == ["It", "cost", "1,000", "dollars."]
    assert len(sentence.words) == 4
