"""The POS lexicon is a read-only artifact, not a self-writing cache.

The old cache was written by generation itself, append-only: "if a word is
already in the cache on disk, do NOT overwrite it".  A wrong tag was therefore
permanent, and each run's tagging fed the next run's candidate pools.
"""
import json

import pytest

from wordfreq_distractor import load_pos_lexicon


@pytest.fixture
def artifact(tmp_path):
    p = tmp_path / "lex.json"
    p.write_text(json.dumps({
        "_meta": {"tagger": "HanTa 1.1.2", "built": "2026-08-20", "language": "de"},
        "tags": {"haus": "NOUN", "alonso": "PROPN", "laufen": "VERB"},
    }), encoding="utf-8")
    return p


@pytest.fixture
def legacy(tmp_path):
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps({"haus": "NOUN", "alonso": "ADV"}), encoding="utf-8")
    return p


def test_artifact_is_preferred_over_the_legacy_cache(artifact, legacy):
    tags = load_pos_lexicon(str(artifact), str(legacy), "de")
    assert tags["alonso"] == "PROPN"      # legacy says ADV


def test_legacy_is_the_fallback_when_no_artifact_exists(tmp_path, legacy):
    tags = load_pos_lexicon(str(tmp_path / "absent.json"), str(legacy), "de")
    assert tags["alonso"] == "ADV"


def test_deleting_the_artifact_reverts_cleanly(artifact, legacy):
    assert load_pos_lexicon(str(artifact), str(legacy), "de")["alonso"] == "PROPN"
    artifact.unlink()
    assert load_pos_lexicon(str(artifact), str(legacy), "de")["alonso"] == "ADV"


def test_missing_both_returns_empty_not_an_error(tmp_path):
    assert load_pos_lexicon(str(tmp_path / "a.json"), str(tmp_path / "b.json"), "de") == {}


def test_corrupt_artifact_falls_back_instead_of_crashing(tmp_path, legacy):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert load_pos_lexicon(str(bad), str(legacy), "de")["alonso"] == "ADV"


def test_artifact_without_tags_section_falls_back(tmp_path, legacy):
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"_meta": {"tagger": "x"}}), encoding="utf-8")
    assert load_pos_lexicon(str(empty), str(legacy), "de")["alonso"] == "ADV"


def test_generation_never_writes_the_lexicon():
    """The run loop must not CALL save_pos_cache.

    Checked on the parse tree, not the source text, so the comment explaining
    why the call was removed does not itself trip the test.
    """
    import ast
    import inspect
    import textwrap

    import main

    tree = ast.parse(textwrap.dedent(inspect.getsource(main.run_stuff)))
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "save_pos_cache" not in called
