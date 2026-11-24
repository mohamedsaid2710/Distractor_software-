"""A missing input-sentence tagger stops the run instead of leaving targets untagged."""
import builtins

import pytest

from maze_automate import selection


@pytest.mark.parametrize("lang, module", [("de", "stanza"), ("ar", "stanza"), ("en", "spacy")])
def test_missing_tagger_raises(monkeypatch, lang, module):
    monkeypatch.setattr(selection, "_nlp_model", {})
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == module:
            raise ImportError(f"No module named '{module}'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(selection.TaggerError):
        selection._get_nlp_model(lang, params={})
