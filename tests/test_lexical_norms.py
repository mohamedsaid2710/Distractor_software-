"""Dominant POS from lexical norms (lexicons/norms.py), on small synthetic inputs.

The real sources are multi-GB downloads (see tools/build_lexicon.py); these tests
pin the parsing and aggregation rules without them.
"""
import json

from maze_automate.lexicons.norms import load_dlexdb, _camel_upos, SUBTLEX_TO_UPOS
from maze_automate.candidates import STTS_TO_UPOS, load_lexicon_extras


def _dlex_row(form, tag, freq):
    # Columns 1, 3 and 8 are read (form, STTS tag, annotated-type frequency).
    return "\t".join([form, str(len(form)), tag, form, "x", "x", "x", str(freq), "0"])


def test_dlexdb_pools_casings_and_takes_the_dominant_reading(tmp_path):
    f = tmp_path / "typposlem.tsv"
    f.write_text("\n".join([
        _dlex_row("gut", "adjd", 600),
        _dlex_row("gut", "adja", 350),
        _dlex_row("Gut", "nn", 50),      # das Gut
        _dlex_row("von", "appr", 999),
        _dlex_row("Von", "ne", 1),
    ]) + "\n", encoding="utf-8")
    n = load_dlexdb(str(f), STTS_TO_UPOS)
    assert n["gut"] == ("ADJ", 0.95)
    assert n["von"][0] == "ADP"


def test_dlexdb_keep_set_limits_what_is_read(tmp_path):
    f = tmp_path / "typposlem.tsv"
    f.write_text(_dlex_row("Haus", "nn", 10) + "\n" + _dlex_row("Baum", "nn", 50) + "\n",
                 encoding="utf-8")
    assert set(load_dlexdb(str(f), STTS_TO_UPOS, keep={"haus"})) == {"haus"}


def test_dlexdb_extra_stts_tags_map_to_upos(tmp_path):
    f = tmp_path / "typposlem.tsv"
    f.write_text(_dlex_row("zwei", "card", 10) + "\n" + _dlex_row("darauf", "pav", 10) + "\n",
                 encoding="utf-8")
    n = load_dlexdb(str(f), STTS_TO_UPOS)
    assert n["zwei"][0] == "NUM" and n["darauf"][0] == "ADV"


def test_camel_pos_mapping():
    assert _camel_upos("noun_prop") == "PROPN"
    assert _camel_upos("prep") == "ADP"
    assert _camel_upos("part_neg") == "PART"
    assert _camel_upos("verb_pseudo") == "VERB"
    assert _camel_upos("something_new") == "X"


def test_subtlex_labels_cover_the_proper_noun_class():
    assert SUBTLEX_TO_UPOS["Name"] == "PROPN"
    assert SUBTLEX_TO_UPOS["Conjunction"] == "CCONJ"


def test_norms_artifact_reports_shares_and_source_kind(tmp_path):
    p = tmp_path / "lex.json"
    p.write_text(json.dumps({"_meta": {"source_kind": "norms"},
                             "tags": {"gut": "ADJ"}, "shares": {"gut": 0.95}}),
                 encoding="utf-8")
    shares, from_norms = load_lexicon_extras(str(p))
    assert from_norms and shares == {"gut": 0.95}


def test_tagger_artifact_is_not_treated_as_norms(tmp_path):
    p = tmp_path / "lex.json"
    p.write_text(json.dumps({"_meta": {"tagger": "HanTa"}, "tags": {"gut": "NOUN"}}),
                 encoding="utf-8")
    assert load_lexicon_extras(str(p)) == ({}, False)


def test_modern_ss_spelling_borrows_the_dominant_old_spelling(tmp_path):
    f = tmp_path / "typposlem.tsv"
    f.write_text("\n".join([
        _dlex_row("Bloss", "ne", 294),     # name in the old-spelling corpus
        _dlex_row("bloß", "adv", 13940),
        _dlex_row("Masse", "nn", 5000),    # a different word from Maße
        _dlex_row("Maße", "nn", 800),
    ]) + "\n", encoding="utf-8")
    n = load_dlexdb(str(f), STTS_TO_UPOS, keep={"bloss", "masse"})
    assert n["bloss"][0] == "ADV"
    assert n["masse"] == ("NOUN", 1.0)       # not merged


def test_too_few_tokens_leaves_the_word_untagged(tmp_path):
    f = tmp_path / "typposlem.tsv"
    f.write_text(_dlex_row("Seltenwort", "nn", 3) + "\n", encoding="utf-8")
    assert "seltenwort" not in load_dlexdb(str(f), STTS_TO_UPOS, keep={"seltenwort"})


# --- ß restoration (wordfreq writes every ß as ss) -----------------------
# Token counts from dlexDB v2 (old-spelling 20th-century corpus).
_SS_COUNTS = dict(
    dass=18783, daß=593703, muss=2546, muß=90245, fluss=103, fluß=2710,
    gross=496, groß=11555, weiss=805, weiß=38518, heissem=5, heißem=602,
    grüsse=18, grüße=1202, füsse=76, füße=3646, masse=5243, maße=7011,
    liessen=118, ließen=6399, strasse=365, straße=14441, haus=100000,
)


def test_modern_spelling_restores_a_reform_proof_eszett():
    from maze_automate.lexicons.norms import modern_spelling
    for ss, real in [("heissem", "heißem"), ("weiss", "weiß"), ("grüsse", "grüße"),
                     ("füsse", "füße"), ("liessen", "ließen"), ("strasse", "straße")]:
        assert modern_spelling(ss, _SS_COUNTS) == real


def test_modern_spelling_keeps_ss_when_both_spellings_are_common():
    from maze_automate.lexicons.norms import modern_spelling
    assert modern_spelling("masse", _SS_COUNTS) == "masse"    # Masse vs Maße
    assert modern_spelling("haus", _SS_COUNTS) == "haus"      # no ss at all


def test_modern_spelling_keeps_ss_where_the_reform_may_have_changed_it():
    # Old daß/muß/Fluß are dass/muss/Fluss today, but old groß is still groß;
    # the spelling cannot tell them apart, so wordfreq's ss form is kept.
    from maze_automate.lexicons.norms import modern_spelling
    for w in ("dass", "muss", "fluss", "gross"):
        assert modern_spelling(w, _SS_COUNTS) == w


def test_dlexdb_fills_spellings(tmp_path):
    f = tmp_path / "typposlem.tsv"
    f.write_text("\n".join([
        _dlex_row("heißem", "adja", 602), _dlex_row("heissem", "adja", 5),
        _dlex_row("daß", "kous", 593703), _dlex_row("dass", "kous", 18783),
    ]) + "\n", encoding="utf-8")
    sp = {}
    load_dlexdb(str(f), STTS_TO_UPOS, keep={"heissem", "dass"}, spellings=sp)
    assert sp == {"heissem": "heißem"}


def test_shipped_german_lexicon_has_no_misspelled_eszett_words():
    import json
    from maze_automate.utils import lexicon_path
    data = json.load(open(lexicon_path("de"), encoding="utf-8"))
    tags = data["tags"]
    for w in ("heißem", "weiß", "grüße", "gießerei", "äußern"):
        assert w in tags
    for w in ("heissem", "weiss", "grüsse"):
        assert w not in tags
    assert "dass" in tags and "gross" in tags   # undecidable: kept with ss
