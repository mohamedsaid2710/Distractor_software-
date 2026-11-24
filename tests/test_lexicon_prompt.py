"""A missing / non-norms POS lexicon is never a silent fallback (main.confirm_lexicon)."""
import io
import json
import sys

import pytest

from maze_automate.pipeline import LexiconError, confirm_lexicon

PARAMS = {"language": "de"}


def _cls(path):
    return type("D", (), {"LEXICON_PATH": str(path), "FALLBACK_TAGGER": "HanTa"})


@pytest.fixture
def norms(tmp_path):
    p = tmp_path / "lex.json"
    p.write_text(json.dumps({"_meta": {"source_kind": "norms"}, "tags": {"gut": "ADJ"}}),
                 encoding="utf-8")
    return p


def test_norms_lexicon_passes_silently(norms):
    confirm_lexicon(_cls(norms), PARAMS)        # no prompt, no error


def test_missing_lexicon_stops_a_non_interactive_run(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))      # not a tty
    with pytest.raises(LexiconError, match=r"(?s)is missing.*--allow-tagger-fallback"):
        confirm_lexicon(_cls(tmp_path / "absent.json"), PARAMS)


def test_explicit_flag_continues(tmp_path):
    confirm_lexicon(_cls(tmp_path / "absent.json"), PARAMS, allow_tagger_fallback=True)


class _Tty(io.StringIO):
    def isatty(self):
        return True


@pytest.mark.parametrize("answer, continues", [("y", True), ("yes", True),
                                               ("", False), ("n", False)])
def test_interactive_run_asks(tmp_path, monkeypatch, answer, continues):
    monkeypatch.setattr(sys, "stdin", _Tty(""))
    cls = _cls(tmp_path / "absent.json")
    if continues:
        confirm_lexicon(cls, PARAMS, ask=lambda _q: answer)
    else:
        with pytest.raises(LexiconError):
            confirm_lexicon(cls, PARAMS, ask=lambda _q: answer)


def test_tagger_built_lexicon_is_also_flagged(tmp_path, monkeypatch):
    p = tmp_path / "lex.json"
    p.write_text(json.dumps({"_meta": {"tagger": "HanTa"}, "tags": {"gut": "NOUN"}}),
                 encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    with pytest.raises(LexiconError, match="built by a tagger"):
        confirm_lexicon(_cls(p), PARAMS)
