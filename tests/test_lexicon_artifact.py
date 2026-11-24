"""The POS lexicon is a read-only artifact, not a self-writing cache.

The old cache was written by generation itself, append-only: "if a word is
already in the cache on disk, do NOT overwrite it".  A wrong tag was therefore
permanent, and each run's tagging fed the next run's candidate pools.
"""
import json

import pytest

from maze_automate.candidates import load_pos_lexicon


@pytest.fixture
def artifact(tmp_path):
    p = tmp_path / "lex.json"
    p.write_text(json.dumps({
        "_meta": {"tagger": "HanTa 1.1.2", "built": "2026-08-20", "language": "de"},
        "tags": {"haus": "NOUN", "alonso": "PROPN", "laufen": "VERB"},
    }), encoding="utf-8")
    return p


def test_artifact_tags_are_loaded(artifact):
    assert load_pos_lexicon(str(artifact), "de")["alonso"] == "PROPN"


def test_missing_lexicon_gives_no_tags_not_an_error(tmp_path):
    assert load_pos_lexicon(str(tmp_path / "a.json"), "de") == {}


def test_corrupt_lexicon_gives_no_tags_not_a_crash(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert load_pos_lexicon(str(bad), "de") == {}


def test_generation_never_writes_the_lexicon():
    """The run loop must not CALL save_pos_cache.

    Checked on the parse tree, not the source text, so the comment explaining
    why the call was removed does not itself trip the test.
    """
    import ast
    import inspect
    import textwrap

    from maze_automate import pipeline

    tree = ast.parse(textwrap.dedent(inspect.getsource(pipeline.run_stuff)))
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "save_pos_cache" not in called
