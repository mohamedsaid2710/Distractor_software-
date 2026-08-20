"""Exclude lists must resolve from any working directory.

Three call sites used to resolve the same bare filename by hand and one had an
extra `dirname`, so it pointed at the repo's *parent*: the exclude list was
dropped with no warning, and the two exclusion checks in a single run could
disagree about which words were banned.
"""
import os

from utils import REPO_ROOT, resolve_repo_path


def test_repo_root_is_the_repo(tmp_path):
    assert os.path.exists(os.path.join(REPO_ROOT, "wordfreq_distractor.py"))


def test_bare_exclude_filename_resolves_from_another_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("exclude_en.txt", "exclude_de.txt", "exclude_ar.txt"):
        resolved = resolve_repo_path(name)
        assert os.path.isabs(resolved), name
        assert os.path.exists(resolved), name


def test_absolute_path_is_returned_unchanged():
    p = "/definitely/not/here.txt"
    assert resolve_repo_path(p) == p


def test_unresolvable_name_is_returned_unchanged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_repo_path("no_such_list.txt") == "no_such_list.txt"


def test_none_passes_through():
    assert resolve_repo_path(None) is None


def test_existing_relative_path_wins(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "exclude_en.txt").write_text("local\n", encoding="utf-8")
    assert resolve_repo_path("exclude_en.txt") == "exclude_en.txt"
